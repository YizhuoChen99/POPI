import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TORCH_NUM_INTEROP_THREADS"] = "1"
import torch
torch.set_num_threads(4)
from vllm import LLM, SamplingParams
import argparse
from datasets import Dataset
import transformers
from trainers import prepare_generation_prompt_from_summary, prepare_generation_prompt_from_raw, prepare_generation_directly_prompt, prepare_summary_prompt_from_raw, prepare_summary_prompt_from_raw_and_question, prepare_generation_directly_prompt_alignx, prepare_generation_prompt_from_raw_alignx_pba, prepare_generation_prompt_from_raw_alignx_ica, prepare_generation_prompt_from_summary_alignx, prepare_summary_prompt_from_raw_alignx_ica, prepare_summary_prompt_from_raw_alignx_pba, prepare_generation_prompt_from_summary_2, prepare_generation_prompt_from_raw_2, prepare_generation_directly_prompt_2
import numpy as np
import random
import tqdm



def parse_args():
    parser = argparse.ArgumentParser(description="Generate summaries using vLLM")
    parser.add_argument("--model", type=str, required=True, help="Path to the vLLM model")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--datasets", type=str, nargs='+', required=True, help="List of dataset names to process")
    parser.add_argument("--start_indexes", type=int, nargs='+', required=True, help="List of start indexes for dataset selection")
    parser.add_argument("--end_indexes", type=int, nargs='+', default=256, help="End index for dataset selection")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    parser.add_argument("--output_paths", type=str, nargs='+', required=True, help="List of output paths for the generated summaries")
    parser.add_argument("--mode", type=str, choices=["summary", "summary_with_question", "generation_from_summary", "generation_from_summary_with_question", "generation_from_summary_as_user_weight", "generation_from_raw", "generation_directly"], required=True, help="Mode of generation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--enforce_eager", action="store_true", help="Enforce eager execution in vLLM")
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    transformers.set_seed(args.seed)


    model_vllm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        trust_remote_code=True,
        tensor_parallel_size=4,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=1,
        top_k=-1,
        skip_special_tokens=True,
        max_tokens=2048,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, cache_dir='.cache/root')


    for dataset_name, output_path, start_index, end_index in zip(args.datasets, args.output_paths, args.start_indexes, args.end_indexes):

        dataset = Dataset.load_from_disk(dataset_name)
        
        end_index = end_index if end_index > 0 else len(dataset)

        dataset = dataset.select(range(start_index, end_index))
        
        if args.mode == "generation_from_summary":
            summaries = dataset["summary"]
            questions = dataset["question"]
            # get the base name of the dataset path
            if 'alignx' in dataset_name:
                prompts = [prepare_generation_prompt_from_summary_alignx(summary, question) for summary, question in zip(summaries, questions)]
            else:
                prompts = [prepare_generation_prompt_from_summary(summary, question) for summary, question in zip(summaries, questions)]
        elif args.mode == "generation_from_summary_with_question":
            summaries = dataset["summary_with_question"]
            questions = dataset["question"]
            if 'alignx' in dataset_name:
                raise NotImplementedError("generation_from_summary_with_question is not implemented for alignx datasets")
            prompts = [prepare_generation_prompt_from_summary(summary, question) for summary, question in zip(summaries, questions)]
        elif args.mode == "generation_from_summary_as_user_weight":
            summaries = dataset["user_token"]
            questions = dataset["question"]
            if 'alignx' in dataset_name:
                raise NotImplementedError("generation_from_summary_as_user_weight is not implemented for alignx datasets")
            prompts = [prepare_generation_prompt_from_summary(summary, question) for summary, question in zip(summaries, questions)]
        elif args.mode == "generation_from_raw":
            raws = dataset["raw"]
            questions = dataset["question"]
            if 'alignx_ica' in dataset_name:
                prompts = [prepare_generation_prompt_from_raw_alignx_ica(raw, question) for raw, question in zip(raws, questions)]
            elif 'alignx_pba' in dataset_name:
                prompts = [prepare_generation_prompt_from_raw_alignx_pba(raw, question) for raw, question in zip(raws, questions)]
            else:
                prompts = [prepare_generation_prompt_from_raw(raw, question) for raw, question in zip(raws, questions)]
        elif args.mode == "generation_directly":
            questions = dataset["question"]
            if 'alignx' in dataset_name:
                prompts = [prepare_generation_directly_prompt_alignx(question) for question in questions]
            else:
                prompts = [prepare_generation_directly_prompt(question) for question in questions]
        elif args.mode == "summary":
            raws = dataset["raw"]
            if 'alignx_ica' in dataset_name:
                prompts = [prepare_summary_prompt_from_raw_alignx_ica(raw) for raw in raws]
            elif 'alignx_pba' in dataset_name:
                prompts = [prepare_summary_prompt_from_raw_alignx_pba(raw) for raw in raws]
            else:
                prompts = [prepare_summary_prompt_from_raw(raw) for raw in raws]
        elif args.mode == "summary_with_question":
            raws = dataset["raw"]
            questions = dataset["question"]
            if 'alignx_ica' in dataset_name:
                raise NotImplementedError("summary_with_question is not implemented for alignx_ica datasets")
            prompts = [prepare_summary_prompt_from_raw_and_question(raw, question) for raw, question in zip(raws, questions)]
        else:
            raise ValueError("Invalid mode")
        
        responses = []

        for i in range(0, len(prompts), args.batch_size):
            end = min(i + args.batch_size, len(prompts))
            batch_prompts = prompts[i:end]
            batch_prompts = tokenizer.apply_chat_template(batch_prompts, tokenize=False, add_generation_prompt=True)
            outputs = model_vllm.generate(batch_prompts, sampling_params)
            batch_completions = [output.outputs[0].text for output in outputs]
            if args.verbose:
                for j in range(len(batch_prompts)):
                    print(f"Prompt {i * args.batch_size + j}: {batch_prompts[j]}")
                    print(f"Completion {i * args.batch_size + j}: {batch_completions[j]}")
            else:
                print(f"example prompt: {batch_prompts[0]}")
                print(f"example completion: {batch_completions[0]}")
            responses.extend(batch_completions)

        dataset = dataset.add_column(args.mode, responses)
        
        dataset.save_to_disk(output_path)

        print(f"Generated responses of {args.mode} for dataset {dataset_name} saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
