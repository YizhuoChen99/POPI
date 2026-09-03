# from unsloth import FastLanguageModel
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
    disable_dropout,
    get_open_port,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple, Set
from transformers import get_scheduler
from accelerate import Accelerator, load_checkpoint_and_dispatch
from omegaconf import OmegaConf
from accelerate.utils import FullyShardedDataParallelPlugin, DataLoaderConfiguration
from torch.nn.utils.rnn import pad_sequence
import datasets
import shutil


def freeze_model(model: nn.Module):
    """Freeze the model parameters to prevent training."""
    for param in model.parameters():
        param.requires_grad = False


def prepare_summary_prompt_from_raw_alignx_pba(context) -> str:
    prompt = (
        "# User Traits\n"
        f"{context}\n\n"
        "# Instruction\n"
        "Convert the above User Traits into a usable and concise user summary. This summary should be suitable for guiding other Large Language Models to write Reddit-style comments that mimic this user.\n\n"
    )
    # make it conversational
    prompt = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    return prompt


def prepare_generation_prompt_from_raw_alignx_pba(context, question) -> str:
    prompt = (
        "# User Traits\n"
        f"{context}\n\n"
        "# Instruction\n"
        "Write a Reddit-style comment on the post below, mimicking the user described above.\n\n"
        "# Post\n"
        f"{question}\n\n"
    )
    # make it conversational
    prompt = [{"role": "system", "content": "You are writing a Reddit-style comment in first person that mimics a given user."}, {"role": "user", "content": prompt}]
    return prompt


def prepare_summary_prompt_from_raw_alignx_ica(context) -> str:
    prompt = (
        "# User Information and Historical Behavior\n"
        f"{context}\n\n"
        "# Instruction\n"
        "Write a concise and structured summary of the User Information and Historical Behavior above. Your summary should be suitable for guiding other Large Language Models to write Reddit-style comments that mimic this user.\n\n"
    )
    # make it conversational
    prompt = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    return prompt


def prepare_generation_prompt_from_summary_alignx(summary, question) -> str:
    prompt = (
        "# User Summary\n"
        f"{summary}\n\n"
        "# Instruction\n"
        "Write a Reddit-style comment on the post below, mimicking the user described above.\n\n"
        "# Post\n"
        f"{question}\n\n"
    )
    # make it conversational
    prompt = [{"role": "system", "content": "You are writing a Reddit-style comment in first person that mimics a given user."}, {"role": "user", "content": prompt}]
    return prompt


def prepare_generation_prompt_from_raw_alignx_ica(context, question) -> str:
    prompt = (
        "# User Information and Historical Behavior\n"
        f"{context}\n\n"
        "# Instruction\n"
        "Write a Reddit-style comment on the post below, mimicking the user described above.\n\n"
        "# Post\n"
        f"{question}\n\n"
    )
    # make it conversational
    prompt = [{"role": "system", "content": "You are writing a Reddit-style comment in first person that mimics a given user."}, {"role": "user", "content": prompt}]
    return prompt


def prepare_generation_directly_prompt_alignx(question):
    prompt = (
        "# Instruction\n"
        "Write a Reddit-style comment on the post below.\n\n"
        "# Post\n"
        f"{question}\n\n"
    )
    prompt = [{"role": "system", "content": "You are writing a Reddit-style comment in first person."}, {"role": "user", "content": prompt}]
    return prompt


def prepare_summary_prompt_from_raw(context) -> str:
    prompt = (
        "# User Preference Examples\n"
        f"{context}\n\n"
        "# Instruction\n"
        "You are an expert at identifying and summarizing user preferences from texts. Analyze the examples above, where each consists of a preferred and dispreferred response to the same prompt. Based on these comparisons, write a concise and structured summary that captures the user's preferences. Your summary should generalize beyond the specific examples and be suitable for guiding the generation of future personalized responses.\n\n"
        # "# User Preference Summary\n"
    )
    # make it conversational
    prompt = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    return prompt


def prepare_summary_prompt_from_raw_and_question(context, question) -> str:
    prompt = (
        "# User Preference Examples\n"
        f"{context}\n\n"
        "# Instruction\n"
        "You are an expert at identifying and summarizing user preferences from texts. Analyze the examples above, where each consists of a preferred and dispreferred response to the same prompt. Based on these comparisons, write a concise and structured summary that captures the user's preferences. Your summary will be used to guide the generation of personalized responses for the user's upcoming prompt.\n\n"
        "# User Upcoming Prompt\n"
        f"{question}\n\n"
    )
    # make it conversational
    prompt = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    return prompt


def prepare_generation_prompt_from_summary(summary, question) -> str:
    prompt = (
        "# User Preference Summary\n"
        f"{summary}\n\n"
        "# Instruction\n"
        "Given the user preference summary above, generate a personalized response to the user prompt below.\n\n" # that strictly adheres to these preferences." # Provide a direct response to the user prompt only. Do not include meta-level remarks or explanations\n\n"
        "# User Prompt\n"
        f"{question}\n\n"
        # "# Personalized Response\n"
    )
    # make it conversational
    # prompt = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    prompt = [{"role": "user", "content": prompt}]
    return prompt


def prepare_generation_prompt_from_raw(context, question) -> str:
    prompt = (
        "# User Preference Examples\n"
        f"{context}\n\n"
        "# Instruction\n"
        "Given the examples above, generate a preferred response to the user prompt below.\n\n" # Provide a direct response to the user prompt only. Do not include meta-level remarks or explanations\n\n"
        "# User Prompt\n"
        f"{question}\n\n"
        # "# Preferred Response\n"
    )
    # make it conversational
    # prompt = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    prompt = [{"role": "user", "content": prompt}]
    return prompt


def prepare_generation_directly_prompt(question):
    # prompt = (
    #     "# Instruction\n"
    #     "Generate a response to the user prompt below.\n\n" #Provide a direct response to the user prompt only. Do not include meta-level remarks or explanations\n\n"
    #     "# User Prompt\n"
    #     f"{question}\n\n"
    #     # "# Response\n"
    # )
    prompt = question
    # prompt = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    prompt = [{"role": "user", "content": prompt}]
    return prompt


####################### rebuttal
def prepare_generation_directly_prompt_2(question):
    user_content = (
        "[Instruction]\n"
        "Craft a response to the User Prompt.\n\n"
        "[User Prompt]\n"
        f"{question}\n"
    )
    prompt = [
        {"role": "system", "content": "You are a helpful assistant who provides well-considered responses."},
        {"role": "user", "content": user_content}
    ]
    return prompt


def prepare_generation_prompt_from_summary_2(summary, question):
    user_content = (
        "[Instruction]\n"
        "Use the User Preference Summary provided below as guidance to craft a personalized response to the User Prompt.\n\n"
        "[User Preference Summary]\n"
        f"{summary}\n\n"
        "[User Prompt]\n"
        f"{question}\n"
    )
    prompt = [
        {"role": "system", "content": "You are a helpful assistant who adapts responses according to user preferences."},
        {"role": "user", "content": user_content}
    ]
    return prompt


def prepare_generation_prompt_from_raw_2(context, question):
    user_content = (
        "[Instruction]\n"
        "Use the User Preference Examples provided below as guidance to craft a personalized response to the User Prompt.\n\n"
        "[User Preference Examples]\n"
        f"{context}\n\n"
        "[User Prompt]\n"
        f"{question}\n"
    )
    prompt = [
        {"role": "system", "content": "You are a helpful assistant who adapts responses according to user preferences."},
        {"role": "user", "content": user_content}
    ]
    return prompt
#######################



def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        losses = losses * beta / 2  # scale by beta to make the loss independent of beta
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        losses = losses / beta  # scale by 1/beta to make the loss independent of beta

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards



def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:]
    logits = logits[:, :-1, :]

    per_token_logps = -torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction='none',
        ignore_index=-100,
    ).reshape(logits.size(0), logits.size(1))

    if average_log_prob:
        raise NotImplementedError("Average log probability not implemented for this function.")
    else:
        return per_token_logps.sum(-1)



def get_preference_pair_batch_metrics(batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train, policy, reference_model, accelerator, do_gather, is_final_evaluation):
    """Compute the SFT or DPO loss and other metrics for the given batch of inputs.
    
    Args:
        batch: A batch of data. Must contain the keys 'summary_chosen_input_ids', 'summary_chosen_attention_mask', 'summary_chosen_labels',
               'summary_rejected_input_ids', 'summary_rejected_attention_mask', 'summary_rejected_labels',
               'raw_chosen_input_ids', 'raw_chosen_attention_mask', 'raw_chosen_labels',
               'raw_rejected_input_ids', 'raw_rejected_attention_mask', 'raw_rejected_labels'.
        loss_config: A configuration object containing the loss type and parameters.
        train: Whether to compute training metrics (True) or evaluation metrics (False).
        policy: The policy model to use for computing the loss and metrics.
        reference_model: The reference model to use for DPO training.
        accelerator: The Accelerator object used for distributed training.

    Returns:
        A tuple containing the loss tensor and a dictionary of metrics.
        The loss tensor has shape (batch_size,) and contains the loss for each example in the batch.
        The metrics dictionary contains various metrics computed from the batch, such as rewards and log probabilities.

    """


    metrics = {}
    train_test = 'train' if train else 'eval'
        
    with torch.no_grad():
        chosen_logits = reference_model(batch[f'direct_chosen_input_ids'], attention_mask=batch[f'direct_chosen_attention_mask']).logits.to(torch.float32)
        reference_chosen_logps = _get_batch_logps(chosen_logits, batch[f'direct_chosen_labels'], average_log_prob=False)
        rejected_logits = reference_model(batch[f'direct_rejected_input_ids'], attention_mask=batch[f'direct_rejected_attention_mask']).logits.to(torch.float32)
        reference_rejected_logps = _get_batch_logps(rejected_logits, batch[f'direct_rejected_labels'], average_log_prob=False)
    
    chosen_logits = policy(batch[f'summary_chosen_input_ids'], attention_mask=batch[f'summary_chosen_attention_mask']).logits.to(torch.float32)
    policy_chosen_logps = _get_batch_logps(chosen_logits, batch[f'summary_chosen_labels'], average_log_prob=False)
    rejected_logits = policy(batch[f'summary_rejected_input_ids'], attention_mask=batch[f'summary_rejected_attention_mask']).logits.to(torch.float32)
    policy_rejected_logps = _get_batch_logps(rejected_logits, batch[f'summary_rejected_labels'], average_log_prob=False)

    if loss_config.name == 'dpo':
        loss_kwargs = {
            'beta': loss_config.beta,
            'reference_free': loss_config.reference_free,
            'label_smoothing': loss_config.label_smoothing,
            'ipo': False
        }
    elif loss_config.name == 'ipo':
        loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
    else:
        raise ValueError(f'unknown loss {loss_config.name}')

    losses, chosen_rewards, rejected_rewards = preference_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        **loss_kwargs
    )

    reward_accuracies = (chosen_rewards > rejected_rewards).float()

    # Gather metrics across all processes for logging
    if do_gather is True:
        chosen_rewards = accelerator.gather_for_metrics(chosen_rewards)
        rejected_rewards = accelerator.gather_for_metrics(rejected_rewards)
        reward_accuracies = accelerator.gather_for_metrics(reward_accuracies)
        policy_rejected_logps = accelerator.gather_for_metrics(policy_rejected_logps.detach())

        if accelerator.is_main_process:
            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().tolist()
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().tolist()

    # Log common metrics
    if do_gather is True:
        policy_chosen_logps = accelerator.gather_for_metrics(policy_chosen_logps.detach())
        all_devices_losses = accelerator.gather_for_metrics(losses.detach())

        if accelerator.is_main_process:
            metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().tolist()
            metrics[f'loss/{train_test}'] = all_devices_losses.cpu().tolist()

    return losses, metrics


def collate_fn(batch, tokenizer, config) -> Dict[str, torch.Tensor]:
    res_batch = {}
    for k in batch[0].keys():
        res_batch[k] = [ex[k] for ex in batch]
    
    summaries = res_batch['summary_with_question'] if config.condition_summary_on_question else res_batch['summary']

    res_batch = create_tokenized_batch(
        summaries=summaries,
        raw=res_batch['raw'],
        question=res_batch['question'],
        chosen=res_batch['my_chosen'],
        rejected=res_batch['my_rejected'],
        tokenizer=tokenizer,
        config=config,
    )
    
    return res_batch


def tokenize_preference_pair_batch(prompts, chosen, rejected, tokenizer, max_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """

    prompt_text = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer(prompt_text, padding=False, truncation=False, max_length=max_length, add_special_tokens=False, return_tensors=None)
    
    chosen_text = [c + tokenizer.eos_token for c in chosen]
    chosen_tokens = tokenizer(chosen_text, padding=False, truncation=False, max_length=max_length, add_special_tokens=False, return_tensors=None)

    rejected_text = [r + tokenizer.eos_token for r in rejected]
    rejected_tokens = tokenizer(rejected_text, padding=False, truncation=False, max_length=max_length, add_special_tokens=False, return_tensors=None)

    # concat the prompt tokens with the chosen/rejected tokens
    chosen_sequence_tokens = defaultdict(list)
    rejected_sequence_tokens = defaultdict(list)
    for i in range(len(prompt_tokens['input_ids'])):
        for k in ['input_ids', 'attention_mask']:
            tmp = prompt_tokens[k][i] + chosen_tokens[k][i]
            chosen_sequence_tokens[k].append(tmp)

            tmp = prompt_tokens[k][i] + rejected_tokens[k][i]
            rejected_sequence_tokens[k].append(tmp)
        
        # create labels
        tmp = [-100] * len(prompt_tokens['input_ids'][i]) + chosen_tokens['input_ids'][i]
        chosen_sequence_tokens['labels'].append(tmp)

        tmp = [-100] * len(prompt_tokens['input_ids'][i]) + rejected_tokens['input_ids'][i]
        rejected_sequence_tokens['labels'].append(tmp)

    # truncate on the right
    truncated = []
    for i in range(len(chosen_sequence_tokens['input_ids'])):
        truncated_i = False
        for k in ['input_ids', 'attention_mask', 'labels']:
            if len(chosen_sequence_tokens[k][i]) > max_length:
                print(f'Truncated chosen sequence with {len(chosen_sequence_tokens[k][i])} tokens')
                chosen_sequence_tokens[k][i] = chosen_sequence_tokens[k][i][:max_length]
                truncated_i = True
            if len(rejected_sequence_tokens[k][i]) > max_length:
                print(f'Truncated rejected sequence with {len(rejected_sequence_tokens[k][i])} tokens')
                rejected_sequence_tokens[k][i] = rejected_sequence_tokens[k][i][:max_length]
                truncated_i = True
        truncated.append(truncated_i)

    # naming
    batch = {}
    batch['prompt_text'] = prompt_text
    batch['chosen_text'] = [prompt_text[i] + chosen_text[i] for i in range(len(prompt_text))]
    batch['rejected_text'] = [prompt_text[i] + rejected_text[i] for i in range(len(prompt_text))]

    
    padding_value = {'input_ids': tokenizer.pad_token_id, 'attention_mask': 0, 'labels': -100}

    for k in ['input_ids', 'attention_mask']:
        
        to_pad = [torch.LongTensor(ex) for ex in prompt_tokens[k]]
        batch[f'prompt_{k}'] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value[k], padding_side='left')

    for k in ['input_ids', 'attention_mask', 'labels']:

        to_pad = [torch.LongTensor(ex) for ex in chosen_sequence_tokens[k]]
        batch[f'chosen_{k}'] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value[k])

        to_pad = [torch.LongTensor(ex) for ex in rejected_sequence_tokens[k]]
        batch[f'rejected_{k}'] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value[k])

    batch['truncated'] = torch.BoolTensor(truncated)

    return batch


def create_tokenized_batch(summaries: List[str], raw: List[str], question: List[str], chosen: List[str], rejected: List[str], tokenizer, config) -> List[float]:
    


    if config.datasets[0] in ['review_4shot', 'elix_4shot', 'roleplay_8shot']:
        direct_prompts = [prepare_generation_directly_prompt(q) for q in question]
    elif config.datasets[0].startswith('alignx'):
        direct_prompts = [prepare_generation_directly_prompt_alignx(q) for q in question]
    else:
        raise ValueError(f"Unknown dataset: {config.datasets[0]}")
    direct_batch = tokenize_preference_pair_batch(
        direct_prompts,
        chosen,
        rejected,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    direct_batch_element = {f'direct_{k}': v for k, v in direct_batch.items()}

    if config.datasets[0] in ['review_4shot', 'elix_4shot', 'roleplay_8shot']:
        summary_prompts = [prepare_generation_prompt_from_summary(s, q) for s, q in zip(summaries, question)]
    elif config.datasets[0].startswith('alignx_ica') or config.datasets[0].startswith('alignx_pba'):
        summary_prompts = [prepare_generation_prompt_from_summary_alignx(s, q) for s, q in zip(summaries, question)]
    else:
        raise ValueError(f"Unknown dataset: {config.datasets[0]}")
    summary_batch = tokenize_preference_pair_batch(
        summary_prompts,
        chosen,
        rejected,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    summary_batch_element = {f'summary_{k}': v for k, v in summary_batch.items()}

    
    return {**summary_batch_element, **direct_batch_element}


@torch.no_grad()
def get_samples(input_ids, attention_mask, max_length, model, tokenizer, accelerator, do_gather):
    """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

    # Unwrap the model for generation
    model = accelerator.unwrap_model(model)

    # self.accelerator.print(f'Generating samples for batch of size {len(batch["prompt"])}')
    policy_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        # max_new_tokens=16,
    )
    # self.accelerator.print(f'Generated {len(policy_output)} samples')

    # Pad and gather outputs
    if do_gather:
        policy_output = pad_to_length(policy_output, max_length, tokenizer.pad_token_id)
        policy_output = accelerator.gather_for_metrics(policy_output)
        if accelerator.is_main_process:
            policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)
        else:
            policy_output_decoded = []
    else:
        policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)

    return policy_output_decoded


def get_batch_iterator(name: str, n_examples=None) -> datasets.Dataset:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """

    dataset = datasets.Dataset.load_from_disk(name)

    if n_examples is not None:
        dataset = dataset.select(range(n_examples))

    return dataset

class BasicTrainer(object):
    def __init__(self, config, policy, reference_model, tokenizer):
        """A trainer for a language model, supporting either SFT or DPO training."""

        dataloader_config = DataLoaderConfiguration(
            split_batches=True,  # Auto-splits large batches for you
        )

        accelerator = Accelerator(
            dataloader_config=dataloader_config,
        )

        if not config.wandb.enabled:
            wandb.init = lambda *args, **kwargs: None
            wandb.log = lambda *args, **kwargs: None

        if accelerator.is_main_process:
            os.makedirs(config.local_run_dir, exist_ok=True)

            config_path = os.path.join(config.local_run_dir, 'generation_config.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config, f)

            wandb.init(
                entity=config.wandb.entity,
                project=config.wandb.project,
                config=OmegaConf.to_container(config),
                dir=get_local_dir(config.local_dirs),
                name=config.exp_name,
            )

        train_dataset_name = os.path.join(config.local_run_dir, 'dataset_train_summarized')
        test_dataset_name = os.path.join(config.local_run_dir, 'dataset_test_summarized')
        
        # simply set the training set to the test set if we are doing final evaluation. It is not used whatsoever.
        if config.is_final_evaluation:
            train_dataset_name = test_dataset_name

        train_dataset = get_batch_iterator(name=train_dataset_name, n_examples=None)
        eval_dataset = get_batch_iterator(name=test_dataset_name, n_examples=config.n_eval_examples)

        collate_fn_partial = functools.partial(collate_fn, tokenizer=tokenizer, config=config)

        accelerator.print(f'Loaded train data iterator')
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn_partial,
            num_workers=4,
            drop_last=True,
        )
        
        accelerator.print(f'Loaded eval data iterator')
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn_partial,
            num_workers=4,
            drop_last=True,
        )
            
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)

        scheduler = get_scheduler(
            config.scheduler,
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=len(train_dataloader) * config.total_rounds
        )

        self.policy, self.train_dataloader, self.eval_dataloader, self.optimizer, self.scheduler = accelerator.prepare(
            policy, train_dataloader, eval_dataloader, optimizer, scheduler
        )

        if config.loss.name in {'dpo', 'ipo'}:
            freeze_model(reference_model)
            # reference_model = accelerator.prepare(reference_model)
            reference_model.to(accelerator.device)
            reference_model.eval()

        # load checkpoint if needed
        if config.ckpt_step > 0:
            checkpoint_path = os.path.join(config.local_run_dir, "generation_logs", f"checkpoint-{config.ckpt_step}")
            accelerator.print(f'Loading checkpoint from {checkpoint_path}')
            accelerator.load_state(checkpoint_path)
            # train_dataloader = accelerator.skip_first_batches(train_dataloader, config.round * config.n_steps)

        self.reference_model = reference_model
        self.accelerator = accelerator
        self.config = config
        self.tokenizer = tokenizer


    def train_batch(self, batch, train=False, do_sample=False):
        """Train on a single batch of data, returning the loss and metrics."""
        if train:
            self.policy.train()
        else:
            self.policy.eval()

        losses, metrics = get_preference_pair_batch_metrics(
            batch, 
            self.config.loss,
            train=train, 
            policy=self.policy, 
            reference_model=self.reference_model, 
            accelerator=self.accelerator, 
            do_gather=True,
            is_final_evaluation=self.config.is_final_evaluation,
        )

        if train:
            self.accelerator.backward(losses.mean())
            grad_norm = self.accelerator.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            if self.accelerator.is_main_process:
                metrics['grad_norm'] = [grad_norm]

        return metrics, []


    @torch.no_grad()
    def eval(self):
        self.accelerator.print(f'Running evaluation after {self.example_counter} train examples')

        eval_example_counter = 0

        if self.accelerator.is_main_process:
            all_eval_metrics = defaultdict(list)
            if self.config.sample_during_eval:
                policy_text_table = wandb.Table(columns=["step", "sample"])
                all_policy_samples = []

        for eval_batch in self.eval_dataloader:
            do_sample = (self.config.sample_during_eval and eval_example_counter < self.config.n_eval_model_samples)
            with torch.no_grad():
                eval_metrics, policy_samples = self.train_batch(eval_batch, train=False, do_sample=do_sample)
            
            if self.accelerator.is_main_process:
                for k, v in eval_metrics.items():
                    all_eval_metrics[k].extend(v)
                if do_sample:
                    all_policy_samples.extend(policy_samples)
                    for sample in policy_samples:
                        policy_text_table.add_data(self.example_counter, sample)

            eval_example_counter += self.config.eval_batch_size

        if self.accelerator.is_main_process:
            mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
            self.accelerator.print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
            wandb.log(mean_eval_metrics, step=self.batch_counter)
            if self.config.sample_during_eval:
                self.accelerator.print(json.dumps(all_policy_samples, indent=2))                    
                wandb.log({"policy_samples": policy_text_table}, step=self.batch_counter)
            if self.config.is_final_evaluation:
                # save the final evaluation metrics to a file for easier access
                with open(f'{self.config.local_run_dir}/final_eval_metrics.txt', 'w') as f:
                    f.write(f'Final evaluation metrics: {formatted_dict(mean_eval_metrics)}\n')


    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        self.batch_counter = self.config.round * len(self.train_dataloader)
        self.example_counter = self.batch_counter * self.config.batch_size
        

        if self.config.do_first_eval:
            self.eval()

        # in final evaluation mode, we don't train the model, so we can return early
        if self.config.is_final_evaluation:
            return

        if self.accelerator.is_main_process:
            start_time = time.time()
            all_metrics = defaultdict(list)

        for batch in self.train_dataloader:
            # since the dataset is already shuffled and selected outside, we just need to go through it sequentially once

            # self.accelerator.print(f"summary_chosen_text: {batch['summary_chosen_text'][0]}")

            # with self.accelerator.accumulate(self.policy):
            metrics, _ = self.train_batch(batch, train=True, do_sample=False)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            # logging
            if self.accelerator.is_main_process:
                for k, v in metrics.items():
                    all_metrics[k].extend(v)

                finish_time = time.time()
                step_time = finish_time - start_time
                examples_per_second = self.config.batch_size / step_time
                all_metrics['examples_per_second'].append(examples_per_second)
                start_time = finish_time
                all_metrics['lr'].append(self.scheduler.get_last_lr()[0])

                if self.batch_counter % self.config.log_every == 0:
                    mean_train_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
                    mean_train_metrics['counters/examples'] = self.example_counter
                    mean_train_metrics['counters/updates'] = self.batch_counter

                    self.accelerator.print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                    wandb.log(mean_train_metrics, step=self.batch_counter)

                    all_metrics = defaultdict(list)


            if self.batch_counter % self.config.eval_every == 0:
                self.eval()

            # if self.batch_counter == self.config.n_steps:
            #     self.accelerator.print(f'Finished training after {self.example_counter} examples')
            #     break

        self.eval()
        self.save() # since generation phase is typically fast, we just save the checkpoint once at the end of training, so no need to bother with complicated checkpointing logic


    def save_model(self, output_dir: str):
        """Save the policy model to disk."""
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        self.accelerator.print(f'Saving policy model to {output_dir}')

        unwrapped_policy = self.accelerator.unwrap_model(self.policy)
        unwrapped_policy.save_pretrained(output_dir, 
                                         is_main_process=self.accelerator.is_main_process, 
                                         save_function=self.accelerator.save, 
                                         state_dict=self.accelerator.get_state_dict(self.policy), 
                                         safe_serialization=True)

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            # remove the accidentally saved corrupted model.safetensors file
            model_safetensors_path = os.path.join(output_dir, 'model.safetensors')
            self.accelerator.print(f'Removing corrupted model.safetensors file at {model_safetensors_path}')
            os.remove(model_safetensors_path)

        self.accelerator.wait_for_everyone()


    def save(self):
        """Save policy, optimizer, and scheduler state to disk, when using the Accelerator."""
        output_dir = f'{self.config.local_run_dir}/generation_logs/checkpoint-{self.batch_counter}/'

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        self.accelerator.print(f'creating checkpoint at {output_dir}')
        self.accelerator.save_state(output_dir)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            # clean up the previous checkpoint
            if self.config.ckpt_step > 0:
                previous_ckpt_path = f"{self.config.local_run_dir}/generation_logs/checkpoint-{self.config.ckpt_step}/"
                shutil.rmtree(previous_ckpt_path)

        self.accelerator.wait_for_everyone()
