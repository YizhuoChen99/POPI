#!/bin/bash
cd "$(dirname "$0")"

dataset_name=elix_4shot

model=meta-llama/Llama-3.2-3B-Instruct

summary_model_evaluated=meta-llama/Llama-3.2-3B-Instruct
policy_evaluated=meta-llama/Llama-3.2-3B-Instruct

condition_summary_on_question=false
beta=2e-3
loss_beta=0.01
total_rounds=1

loss_name=dpo

exp_name=popi

if [ "$model" = "$policy_evaluated" ]; then
    off_the_shelf=false
    n_generation_examples=512 # have some redundancy for later llmaaj
    n_samples_persona=80
    n_samples_total=400
else
    off_the_shelf=true
    n_generation_examples=256 # have some redundancy for later llmaaj
    n_samples_persona=40
    n_samples_total=200
fi


seed=42

reference_exp_name=base_model

if [[ "$dataset_name" == *_92000 ]]; then
    test_dataset_name="${dataset_name%_92000}"
else
    test_dataset_name=$dataset_name
fi


if [ "$condition_summary_on_question" = "true" ]; then
    generation_mode=generation_from_summary_with_question
    summary_mode=summary_with_question
else
    generation_mode=generation_from_summary
    summary_mode=summary
fi
generation_dataset=.cache/root/results/${exp_name}/${dataset_name}/${loss_name}/dataset_test_summarized
summary_model=.cache/root/results/${exp_name}/${dataset_name}/${loss_name}/summary_model
policy_ckpt=.cache/root/results/${exp_name}/${dataset_name}/${loss_name}/policy


# when evaluating on an unseen dataset, we need to grab the summary model and/or the policy from a source experiment
if [[ "$summary_model" == *"_demo"* || "$summary_model" == *"_pair"* || "$summary_model" == *"_ugc"* ]]; then
    summary_model="${summary_model//_demo/_arbitrary}"
    summary_model="${summary_model//_pair/_arbitrary}"
    summary_model="${summary_model//_ugc/_arbitrary}"
fi
if [[ "$policy_ckpt" == *"_demo"* || "$policy_ckpt" == *"_pair"* || "$policy_ckpt" == *"_ugc"* ]]; then
    policy_ckpt="${policy_ckpt//_demo/_arbitrary}"
    policy_ckpt="${policy_ckpt//_pair/_arbitrary}"
    policy_ckpt="${policy_ckpt//_ugc/_arbitrary}"
fi

if [ ! -z "$summary_model" ]; then
    python src/generate_vllm.py \
        --mode $summary_mode \
        --model $summary_model \
        --datasets .cache/root/preprocessed_datasets/${test_dataset_name}_test \
        --output_paths .cache/root/results/${exp_name}/${dataset_name}/${loss_name}/dataset_test_summarized \
        --tokenizer $model \
        --start_indexes 0 \
        --end_indexes 0 \
        --gpu_memory_utilization 0.9 \
        --batch_size 512 \
        --seed $seed \
        --enforce_eager
fi


if [ "$off_the_shelf" = "true" ]; then
    echo "skipping the final evaluation because it is not supported for off-the-shelf models"
else
    accelerate launch --config_file deepspeed_zero3.yaml src/train.py \
        condition_summary_on_question=$condition_summary_on_question \
        is_final_evaluation=true \
        policy_ckpt=$policy_ckpt \
        ckpt_step=0 \
        seed=$seed \
        phase=generation \
        round=0 \
        total_rounds=1 \
        model=$model \
        tokenizer=$model \
        datasets=[${dataset_name}] \
        n_examples=8 \
        do_first_eval=true \
        loss=${loss_name} \
        loss.beta=$loss_beta \
        lr=1e-6 \
        scheduler=cosine \
        exp_name=${exp_name} \
        sample_during_eval=false \
        gradient_accumulation_steps=1 \
        eval_every=99999 \
        log_every=99999 \
        save_every=99999 \
        batch_size=8 \
        eval_batch_size=16 \
        max_length=8192 \
        wandb.enabled=false
    fi
fi

if [ "$off_the_shelf" = "true" ]; then
    generated_path=.cache/root/results/${exp_name}/${dataset_name}/${loss_name}/off_the_shelf/${policy_evaluated}/dataset_test_generated
    generator_name=${exp_name}_${policy_evaluated}
    alpaca_eval_output_path=.cache/root/results/${exp_name}/${dataset_name}/${loss_name}/off_the_shelf/${policy_evaluated}
    alpaca_eval_reference_path=.cache/root/results/${reference_exp_name}/${dataset_name}/${loss_name}/off_the_shelf/${policy_evaluated}
else
    generated_path=.cache/root/results/${exp_name}/${dataset_name}/${loss_name}/dataset_test_generated
    generator_name=${exp_name}
    alpaca_eval_output_path=.cache/root/results/${exp_name}/${dataset_name}/${loss_name}
    alpaca_eval_reference_path=.cache/root/results/${reference_exp_name}/${dataset_name}/${loss_name}
fi


python src/generate_vllm.py \
    --mode $generation_mode \
    --model $policy_ckpt \
    --datasets $generation_dataset \
    --output_paths $generated_path \
    --tokenizer $model \
    --start_indexes 0 \
    --end_indexes $n_generation_examples \
    --gpu_memory_utilization 0.7 \
    --batch_size 512 \
    --seed $seed \
    --enforce_eager

python src/convert_to_alpaca_eval.py \
    --dataset $dataset_name \
    --generator $generator_name \
    --input_path $generated_path \
    --output_path $alpaca_eval_output_path \
    --n_samples_persona $n_samples_persona \
    --n_samples_total $n_samples_total \
    --column $generation_mode \
    --tokenizer $tokenizer


# if [ "$exp_name" = "$reference_exp_name" ]; then
#     echo "skipping the evaluation of the same experiment"
# else
#     alpaca_eval evaluate \
#         --model_outputs ${alpaca_eval_output_path}/generated_test_alpaca_eval.json \
#         --reference_outputs ${alpaca_eval_reference_path}/generated_test_alpaca_eval.json \
#         --annotators_config popi_customized \
#         --output_path ${alpaca_eval_output_path} \
#         --sort_by win_rate \
#         --annotation_kwargs '{"is_ordered": True}'
# fi
