#!/bin/bash
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false

dataset_name=elix_4shot

condition_summary_on_question=false
beta=2e-3
loss_beta=0.01
total_rounds=1

loss_name=dpo
n_examples=20000
n_eval_examples=100

model=meta-llama/Llama-3.2-3B-Instruct
seed=42

backend=deepspeed_zero3

batch_size=8
rl_real_batch_size=1

eval_every=250
log_every=250
save_every=500 #5000

sleep_time=10


if [ -f .cache/root/results/popi/${dataset_name}/${loss_name}/ckpt_step.txt ]; then
    read next_round next_phase ckpt_step < .cache/root/results/popi/${dataset_name}/${loss_name}/ckpt_step.txt
else
    next_round=0
    next_phase=summary
    ckpt_step=0
fi


if [ "$next_phase" != "summary" ]; then
    echo "Skipping summary phase because it has already been done."
else

    accelerate launch --config_file ${backend}.yaml src/train.py \
        lambda_length=0 \
        length_threshold=64 \
        beta=$beta \
        condition_summary_on_question=$condition_summary_on_question \
        is_final_evaluation=false \
        policy_ckpt=$model \
        ckpt_step=$ckpt_step \
        seed=$seed \
        phase=summary \
        round=0 \
        total_rounds=$total_rounds \
        model=$model \
        tokenizer=$model \
        datasets=[${dataset_name}] \
        n_examples=${n_examples} \
        n_eval_examples=${n_eval_examples} \
        do_first_eval=false \
        loss=${loss_name} \
        loss.beta=$loss_beta \
        lr=1e-6 \
        scheduler=cosine \
        exp_name=popi \
        sample_during_eval=false \
        gradient_accumulation_steps=$(($batch_size / rl_real_batch_size)) \
        eval_every=$eval_every \
        log_every=$log_every \
        save_every=$save_every \
        batch_size=$rl_real_batch_size \
        eval_batch_size=$rl_real_batch_size \
        max_length=8192 || exit 1

    sleep $sleep_time


    python src/generate_vllm.py \
        --mode summary \
        --model .cache/root/results/popi/${dataset_name}/${loss_name}/summary_model \
        --datasets .cache/root/preprocessed_datasets/${dataset_name}_train .cache/root/preprocessed_datasets/${test_dataset_name}_test \
        --output_paths .cache/root/results/popi/${dataset_name}/${loss_name}/dataset_train_summarized .cache/root/results/popi/${dataset_name}/${loss_name}/dataset_test_summarized \
        --tokenizer $model \
        --start_indexes 0 0 \
        --end_indexes $n_examples $n_eval_examples \
        --gpu_memory_utilization 0.8 \
        --seed $seed || exit 1

    sleep $sleep_time

    echo "0 generation 0" > .cache/root/results/popi/${dataset_name}/${loss_name}/ckpt_step.txt
fi

if [ "$next_phase" != "generation" ]; then
    echo "Skipping generation phase because it has already been done."
else

    accelerate launch --config_file ${backend}.yaml src/train.py \
        condition_summary_on_question=$condition_summary_on_question \
        is_final_evaluation=false \
        policy_ckpt=$model \
        ckpt_step=$ckpt_step \
        seed=$seed \
        phase=generation \
        round=0 \
        total_rounds=$total_rounds \
        model=$model \
        tokenizer=$model \
        datasets=[${dataset_name}] \
        n_examples=${n_examples} \
        n_eval_examples=${n_eval_examples} \
        do_first_eval=false \
        loss=${loss_name} \
        loss.beta=$loss_beta \
        lr=1e-6 \
        scheduler=cosine \
        exp_name=popi \
        sample_during_eval=false \
        gradient_accumulation_steps=1 \
        eval_every=$eval_every \
        log_every=$log_every \
        save_every=$save_every \
        batch_size=$batch_size \
        eval_batch_size=$batch_size \
        max_length=6144 || exit 1

    sleep $sleep_time

    ckpt_step=$((n_examples / batch_size))
    echo "1 summary $ckpt_step" > .cache/root/results/popi/${dataset_name}/${loss_name}/ckpt_step.txt
fi
