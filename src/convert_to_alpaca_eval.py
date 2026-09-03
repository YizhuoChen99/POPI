# covert to alpaca_eval format

import json
import os
import argparse
import datasets
from collections import defaultdict
import transformers

def parse_args():
    parser = argparse.ArgumentParser(description="Convert responses to Alpaca Eval format")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--generator", type=str, required=True, help="Name of the generator")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset file")
    parser.add_argument("--output_path", type=str, default="./responses_alpaca_eval", help="Directory to save the converted responses")
    parser.add_argument("--n_samples_persona", type=int, default=50, help="Number of samples per level")
    parser.add_argument("--n_samples_total", type=int, default=1000, help="Total number of samples to generate")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to use for text processing")
    parser.add_argument("--column", type=str, default="generation_from_summary", help="Column to use for generation")
    return parser.parse_args()


def get_avg_length(texts, tokenizer):
    tokenized = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
    lengths = [len(tokens) for tokens in tokenized['input_ids']]
    return sum(lengths) / len(lengths)


def convert_to_alpaca_eval(args):
    dataset = datasets.load_from_disk(args.input_path)

    data = []
    contexts = []
    persona_count = defaultdict(int)
    total_count = 0

    for item in dataset:
        persona = item['persona']

        if total_count >= args.n_samples_total:
            break

        if persona in persona_count and persona_count[persona] >= args.n_samples_persona:
            continue

        if args.column.startswith("generation_from_summary_with_question"):
            contexts.append(item['summary_with_question'])
        elif args.column.startswith("generation_from_summary_as_user_weight"):
            contexts.append(item['summary_as_user_weight'])
        elif args.column.startswith("generation_from_summary"):
            contexts.append(item['summary'])
        elif args.column.startswith("generation_from_raw"):
            contexts.append(item['raw'])
        elif args.column.startswith("generation_directly"):
            pass
        else:
            raise ValueError("No valid generation field found in the item")
        
        output = item[args.column]
        
        data.append({
            "instruction": item['question'],
            "output": output,
            "description": persona,
            "generator": args.generator,
            "dataset": args.dataset,
            "split": "test",
        })
        persona_count[persona] += 1
        total_count += 1

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    fname = os.path.join(args.output_path, "generated_test_alpaca_eval.json")
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


    ## Calculate average length of outputs
    if len(contexts) == 0:
        avg_length = 0

    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, cache_dir='.cache/root')
        avg_length = get_avg_length(contexts, tokenizer)

    fname = os.path.join(args.output_path, "avg_length.txt")
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(f"Average length of outputs: {avg_length}\n")
    print("avg_length", avg_length)
        

if __name__ == "__main__":
    args = parse_args()
    convert_to_alpaca_eval(args)


