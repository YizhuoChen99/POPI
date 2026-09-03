# import sys
# sys.dont_write_bytecode = True
import os
os.environ['XDG_CACHE_HOME'] = ".cache/root"
os.environ['WANDB_CACHE_DIR'] = ".cache/root"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TORCH_NUM_INTEROP_THREADS"] = "1"
import torch
torch.set_num_threads(4)
import torch.nn as nn
from utils import get_local_dir, get_local_run_dir, init_distributed, get_open_port
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
from typing import Optional, Set, Tuple, List, Dict
import numpy as np
import random
import functools
import transformers
from transformers import TrainerCallback
from utils import get_block_class_from_model, disable_dropout
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from trainers import BasicTrainer, get_preference_pair_batch_metrics, freeze_model, create_tokenized_batch, get_batch_iterator, prepare_summary_prompt_from_raw, prepare_summary_prompt_from_raw_and_question, prepare_summary_prompt_from_raw_alignx_pba, prepare_summary_prompt_from_raw_alignx_ica
from trl import GRPOConfig, GRPOTrainer
import datasets
import time
import shutil
import math


OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs, datasets, phase, loss: get_local_run_dir(exp_name, local_dirs, datasets, phase, loss))


def get_a_transformer(config: DictConfig, name_or_path) -> nn.Module:
    """Get a transformer model based on the configuration.
    
    Args:
        config: A DictConfig object containing the model configuration.
    
    Returns:
        An instance of a transformer model.
    """
    model_kwargs = {"attn_implementation": "flash_attention_2"}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name_or_path, cache_dir=os.path.join(get_local_dir(config.local_dirs), 'huggingface', 'hub'), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, use_cache=False, **model_kwargs)
    disable_dropout(model)

    return model


def get_models(config: DictConfig) -> Tuple[nn.Module, Optional[nn.Module], transformers.PreTrainedTokenizer]:
    """Build the policy and reference models based on the configuration.
    Args:
        config: A DictConfig object containing the model configuration.
    
    Returns:
        A tuple of (policy_model, reference_model, tokenizer).
        If the loss is not DPO or IPO, reference_model will be None.
    """

    # policy_path = config.model if config.round == 0 else os.path.join(config.local_run_dir, 'policy')  # we will do summary before generation in first round.
    policy = get_a_transformer(config, config.policy_ckpt)

    if config.loss.name in {'dpo', 'ipo'}:
        reference_model = get_a_transformer(config, config.model)
    else:
        reference_model = None

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer, cache_dir=os.path.join(get_local_dir(config.local_dirs), 'huggingface', 'hub'))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    assert tokenizer.truncation_side == 'right'

    return policy, reference_model, tokenizer


class UploadAndDeleteCheckpointCallback(TrainerCallback):
    def __init__(self, config):
        self.config = config

    def on_save(self, args, state, control, **kwargs):

        if not state.is_world_process_zero:
            return control

        ckpt_path = f'{self.config.local_run_dir}/summary_logs/checkpoint-{state.global_step}/'
        next_step_fname = os.path.join(self.config.local_run_dir, 'ckpt_step.txt')
        with open(next_step_fname, 'w') as f:
            f.write(f'{self.config.round} summary {state.global_step}')

        # clean up the previous checkpoint
        if self.config.ckpt_step > 0:
            previous_ckpt_path = f'{self.config.local_run_dir}/summary_logs/checkpoint-{self.config.ckpt_step}/'
            shutil.rmtree(previous_ckpt_path)

        self.config.ckpt_step = state.global_step


        return control


def get_grpo_config(config: DictConfig) -> GRPOConfig:
     # figure out some of the training parameters automatically
    num_processes = 8
    num_generations = 8
    cur_steps = config.round * config.n_examples // config.batch_size // config.gradient_accumulation_steps
    max_steps = (config.round + 1) * config.n_examples // config.batch_size // config.gradient_accumulation_steps

    if config.total_rounds == 1:
        scheduler = config.scheduler
        learning_rate = config.lr
    else:
        if config.round == 0:
            scheduler = 'constant_with_warmup'
            learning_rate = config.lr
        else:
            total_steps = config.total_rounds * config.n_examples // config.batch_size // config.gradient_accumulation_steps
            scheduler = 'constant'
            if config.scheduler == 'linear':
                def lr_lambda(current_step):
                    if current_step < config.warmup_steps:
                        return float(current_step) / float(max(1, config.warmup_steps))
                    return max(
                        0.0, float(total_steps - current_step) / float(max(1, total_steps - config.warmup_steps))
                    )
            elif config.scheduler == 'cosine':
                def lr_lambda(current_step):
                    if current_step < config.warmup_steps:
                        return float(current_step) / float(max(1, config.warmup_steps))
                    progress = float(current_step - config.warmup_steps) / float(max(1, total_steps - config.warmup_steps))
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * 0.5 * progress)))
            else:
                raise ValueError(f"Unknown scheduler: {config.scheduler}")
            learning_rate = config.lr * lr_lambda(cur_steps)

    grpo_config = GRPOConfig(
        output_dir=os.path.join(config.local_run_dir, 'summary_logs'),
        seed=config.seed,
        num_generations=num_generations,
        report_to='wandb',
        max_steps=max_steps,
        run_name=config.exp_name, 
        per_device_train_batch_size=config.batch_size*num_generations//num_processes, 
        per_device_eval_batch_size=config.eval_batch_size*num_generations//num_processes, 
        dataloader_num_workers=num_processes, 
        max_prompt_length=config.max_length,
        max_completion_length=2048,
        eval_strategy='steps',
        eval_steps=config.eval_every,
        log_completions=config.sample_during_eval,
        logging_steps=config.log_every,
        num_completions_to_print=config.n_eval_model_samples,
        save_strategy='steps',
        save_steps=config.save_every,
        use_vllm=True,
        vllm_mode='colocate',
        vllm_gpu_memory_utilization=0.15,
        vllm_tensor_parallel_size=1,
        dataloader_drop_last=not config.exact,
        temperature=0.6,
        top_p=1.0,
        top_k=None,
        ignore_data_skip=False,
        shuffle_dataset=False,
        optim='adamw_torch',
        learning_rate=learning_rate,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=scheduler,
        beta=config.beta,
        reward_weights=[1.0, config.lambda_length] if config.lambda_length > 0 else [1.0],
        scale_rewards=False,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=True,
        eval_on_start=config.do_first_eval,
    )

    return grpo_config


def main_summary(config: DictConfig, policy, reference_model, tokenizer):

    name = os.path.join(get_local_dir(config.local_dirs), 'preprocessed_datasets', f'{config.datasets[0]}_train')
    train_dataset = get_batch_iterator(name=name, n_examples=None)

    if config.datasets[0].endswith('_92000'):
        test_dataset_name = config.datasets[0].replace('_92000', '')
    else:
        test_dataset_name = config.datasets[0]

    name = os.path.join(get_local_dir(config.local_dirs), 'preprocessed_datasets', f'{test_dataset_name}_test')
    eval_dataset = get_batch_iterator(name=name, n_examples=config.n_eval_examples)

    if config.condition_summary_on_question:
        if config.datasets[0].startswith('alignx'):
            raise NotImplementedError("Conditioning summary on question is not implemented for alignx datasets")
        def prepare_summary_prompt(example):
            example['prompt'] = prepare_summary_prompt_from_raw_and_question(example['raw'], example['question'])
            return example
    else:
        if config.datasets[0] in ['review_4shot', 'elix_4shot', 'roleplay_8shot']:
            def prepare_summary_prompt(example):
                example['prompt'] = prepare_summary_prompt_from_raw(example['raw'])
                return example
        elif config.datasets[0].startswith('alignx_pba'):
            def prepare_summary_prompt(example):
                example['prompt'] = prepare_summary_prompt_from_raw_alignx_pba(example['raw'])
                return example
        elif config.datasets[0].startswith('alignx_ica'):
            def prepare_summary_prompt(example):
                example['prompt'] = prepare_summary_prompt_from_raw_alignx_ica(example['raw'])
                return example
        else:
            raise ValueError(f"Unknown dataset: {config.datasets[0]}")

    train_dataset = train_dataset.map(prepare_summary_prompt, load_from_cache_file=False)
    eval_dataset = eval_dataset.map(prepare_summary_prompt, load_from_cache_file=False)

    @torch.no_grad()
    def reward_preference(prompts: List[str], completions: List[str], raw: List[str], question: List[str], my_chosen: List[str], my_rejected: List[str], **kwargs):

        # 1. Create batch.
        summaries = [completion[0]['content'] for completion in completions]
        # print(len(summaries))
        # print(f"Summaries: {summaries}, chosen: {my_chosen}")

        total_size = len(summaries)
        mini_batch_size = config.batch_size
        all_rewards = []

        for start in range(0, total_size, mini_batch_size):
            end = min(start + mini_batch_size, total_size)

            summaries_chunk = summaries[start:end]
            raw_chunk = raw[start:end]
            question_chunk = question[start:end]
            chosen_chunk = my_chosen[start:end]
            rejected_chunk = my_rejected[start:end]

            batch = create_tokenized_batch(
                summaries=summaries_chunk,
                raw=raw_chunk,
                question=question_chunk,
                chosen=chosen_chunk,
                rejected=rejected_chunk,
                tokenizer=tokenizer,
                config=config,
            )

            # Move to correct device
            batch = {
                k: v.to(trainer.accelerator.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            # Compute reward
            losses, _ = get_preference_pair_batch_metrics(
                batch,
                config.loss,
                train=False,
                policy=policy,
                reference_model=reference_model,
                accelerator=None,
                do_gather=False,
                is_final_evaluation=False,
            )

            reward = (-losses).cpu().tolist()


            all_rewards.extend(reward)

        return all_rewards
    
    trainer = GRPOTrainer(
        model=config.model,
        processing_class=tokenizer,
        args=get_grpo_config(config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[reward_preference],
        callbacks=[UploadAndDeleteCheckpointCallback(config=config)],
    )

    if trainer.accelerator.is_main_process:
        config_path = os.path.join(config.local_run_dir, 'summary_config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config, f)

    print(f"policy and reference model sent to device: {trainer.accelerator.device}")
    freeze_model(policy)
    # policy = trainer.accelerator.prepare(policy)
    policy.to(trainer.accelerator.device)
    policy.eval()
    if reference_model is not None:
        freeze_model(reference_model)
        # reference_model = trainer.accelerator.prepare(reference_model)
        reference_model.to(trainer.accelerator.device)
        reference_model.eval()

    resume_from_checkpoint = os.path.join(config.local_run_dir, 'summary_logs', f'checkpoint-{config.ckpt_step}') if config.ckpt_step > 0 else None
    print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(os.path.join(config.local_run_dir, 'summary_model'))
    trainer.accelerator.wait_for_everyone()


def main_generation(config: DictConfig, policy, reference_model, tokenizer):


    trainer = trainers.BasicTrainer(config, 
                                    policy=policy, 
                                    reference_model=reference_model, 
                                    tokenizer=tokenizer)
    trainer.train()
    if config.is_final_evaluation:
        return
    trainer.save_model(os.path.join(config.local_run_dir, 'policy'))
    trainer.accelerator.wait_for_everyone()
    

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    transformers.set_seed(config.seed)

    policy, reference_model, tokenizer = get_models(config)

    if config.phase == 'summary':
        main_summary(config, policy, reference_model, tokenizer)
    elif config.phase == 'generation':
        main_generation(config, policy, reference_model, tokenizer)


if __name__ == '__main__':
    main()