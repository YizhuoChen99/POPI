from torch.utils.data import Dataset
import datasets
from preference_datasets import get_dataset
import argparse
import numpy as np
import random
import transformers
from huggingface_hub import snapshot_download
import json
from collections import defaultdict
import torch
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset for preference optimization")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to preprocess")
    parser.add_argument("--split", type=str, choices=["train", "test"], required=True, help="Dataset split to preprocess")
    return parser.parse_args()


features_persona = [
    ("Young", "Older"),
    ("Female", "Male"),
    ("High Neuroticism", "Low Neuroticism"),
    ("High Extraversion", "Low Extraversion"),
    ("High Openness", "Low Openness"),
    ("High Agreeableness", "Low Agreeableness"),
    ("High Conscientiousness", "Low Conscientiousness"),
    ("Likes a certain food", "Dislikes a certain food"),
    ("Likes a certain living environment", "Dislikes a certain living environment"),
    ("Likes sleep", "Dislikes sleep"),
    ("Aggressive investment", "Conservative investment"),
    ("Good at saving", "Bad at saving"),
    ("Concerned about physical safety", "Not concerned about physical safety"),
    ("Concerned about environmental safety", "Not concerned about environmental safety"),
    ("Prefers superficial interaction (casual, stress-free chat)", "Prefers deep interaction (discussing interests, emotional topics, etc.)"),
    ("Prefers direct communication to handle conflict", "Prefers avoidance, mediation, compromise to handle conflict"),
    ("Concise communication style", "Detailed communication style"),
    ("Strong need for a certain work environment", "Indifferent to work environment needs"),
    ("Strong need for recognition from others", "Indifferent to recognition from others"),
    ("Strong need for personal achievement", "Indifferent to personal achievement"),
    ("Likes a certain area of knowledge", "Dislikes a certain area of knowledge"),
    ("Likes a certain learning style", "Dislikes a certain learning style"),
    ("Likes a certain form of creative expression (e.g., art, writing, music)", "Dislikes a certain form of creative expression (e.g., art, writing, music)"),
    ("Strong need for Order (neatness, organization, avoiding chaos)", "Indifferent to orderliness"),
    ("Strong need for Retention (holding onto objects, unwilling to lose or change)", "Indifferent to retention (unconcerned about keeping objects)"),
    ("Strong need for Inviolacy (maintaining dignity and reputation)", "Indifferent to inviolacy (unconcerned with dignity or reputation)"),
    ("Strong need for Infavoidance (avoiding failure and embarrassment)", "Indifferent to Infavoidance (unconcerned with failure or embarrassment)"),
    ("Strong need for Counteraction (overcoming failure and obstacles)", "Indifferent to Counteraction (unconcerned with failure)"),
    ("Strong need for Seclusion (desire for isolation from others)", "Indifferent to Seclusion (does not care about isolation)"),
    ("Strong need for Dominance (controlling others through command or persuasion)", "Indifferent to Dominance (does not care about control)"),
    ("Strong need for Deference (following authority or rules)", "Indifferent to Deference (does not care about authority)"),
    ("Strong need for Autonomy (pursuing independence and self-reliance)", "Indifferent to Autonomy (does not care about independence)"),
    ("Strong need for Contrariance (pursuing uniqueness, opposing the norm)", "Indifferent to Contrariance (does not seek uniqueness)"),
    ("Strong need for Abasement (accepting blame, enjoying pain or misfortune)", "Indifferent to Abasement (does not accept blame or enjoy misfortune)"),
    ("Strong need for Aggression (controlling others through forceful means)", "Indifferent to Aggression (does not engage in aggression)"),
    ("Strong need for Affiliation (desiring close relationships)", "Indifferent to Affiliation (does not care about close relationships)"),
    ("Strong need for Rejection (isolating oneself from negatively evaluated people)", "Indifferent to Rejection (does not care about social exclusion)"),
    ("Strong need for Nurturance (caring for others, protecting them from danger)", "Indifferent to Nurturance (does not care about nurturing others)"),
    ("Strong need for Succorance (desiring help, love, and comfort from others)", "Indifferent to Succorance (does not rely on others for comfort)"),
    ("Strong need for Play (enjoying fun, relaxation, and laughter)", "Indifferent to Play (does not prioritize fun or relaxation)"),
    ("Concerned about harmlessness", "Indifferent about harmlessness"),
    ("Concerned about instruction-following", "Indifferent about instruction-following"),
    ("Concerned about honesty", "Indifferent about honesty"),
    ("Concerned about truthfulness", "Indifferent about truthfulness"),
    ("Concerned about helpfulness", "Indifferent about helpfulness"),
    ("Concerned about coherence", "Indifferent about coherence"),
    ("Concerned about complexity", "Indifferent about complexity"),
    ("Likes science", "Dislikes science"),
    ("Likes knowledge", "Dislikes knowledge"),
    ("Likes psychology", "Dislikes psychology"),
    ("Likes cinema", "Dislikes cinema"),
    ("Likes entertainment", "Dislikes entertainment"),
    ("Likes gaming", "Dislikes gaming"),
    ("Likes parenting", "Dislikes parenting"),
    ("Likes wild imagination", "Dislikes wild imagination"),
    ("Likes anime", "Dislikes anime"),
    ("Likes sports", "Dislikes sports"),
    ("Likes law", "Dislikes law"),
    ("Likes workplace", "Dislikes workplace"),
    ("Likes pets", "Dislikes pets"),
    ("Likes travel", "Dislikes travel"),
    ("Likes health", "Dislikes health"),
    ("Likes stories", "Dislikes stories"),
    ("Likes cars", "Dislikes cars"),
    ("Likes gourmet food", "Dislikes gourmet food"),
    ("Likes education", "Dislikes education"),
    ("Likes current events", "Dislikes current events"),
    ("Likes home decor", "Dislikes home decor"),
    ("Likes international", "Dislikes international"),
    ("Likes finance", "Dislikes finance"),
    ("Likes campus life", "Dislikes campus life"),
    ("Likes digital technology", "Dislikes digital technology"),
    ("Likes emotions", "Dislikes emotions"),
    ("Likes humor", "Dislikes humor"),
    ("Likes music", "Dislikes music"),
    ("Likes reading", "Dislikes reading"),
    ("Likes painting", "Dislikes painting"),
    ("Likes dance", "Dislikes dance"),
    ("Likes crafts", "Dislikes crafts"),
    ("Likes photography", "Dislikes photography"),
    ("Likes culture", "Dislikes culture"),
    ("Likes fitness", "Dislikes fitness"),
    ("Likes art", "Dislikes art"),
    ("Likes stationery and planners", "Dislikes stationery and planners"),
    ("Likes celebrities", "Dislikes celebrities"),
    ("Likes outdoors", "Dislikes outdoors"),
    ("Likes camping", "Dislikes camping"),
    ("Likes social sciences", "Dislikes social sciences"),
    ("Likes weddings", "Dislikes weddings"),
    ("Likes fashion", "Dislikes fashion")
]


def trans_persona(embedding):
    description = []

    for i, value in enumerate(embedding):
        if value == 1:
            description.append(features_persona[i][0])  
        elif value == 0:
            description.append(features_persona[i][1])

    result = ', '.join(description)
    return result


def get_profile(history):
    embedding = []
    for it in history:
        features = it["Preference Direction"]
        embedding.append(features)
    mean_embedding = torch.tensor(embedding).mean(dim=0, keepdim=True)
    mean_embedding_list = mean_embedding.squeeze().tolist()
    # print(mean_embedding_list)
    # breakpoint()
    binary_emb = [
        1 if x >= 0.51 else 0 if x <= 0.49 else 0.5 
        for x in mean_embedding_list
    ]

    profile = trans_persona(binary_emb)

    if profile == "":
        ########
        mean_embedding = torch.tensor(embedding).mean(dim=0, keepdim=True)
        mean_embedding = mean_embedding.squeeze()

        max_val = torch.max(mean_embedding)
        min_val = torch.min(mean_embedding)
        if 1 - max_val < min_val:
            mean_embedding[mean_embedding == max_val] = 1
        else:
            mean_embedding[mean_embedding == min_val] = 0
        
        mean_embedding = torch.where((mean_embedding != 1) & (mean_embedding != 0), torch.tensor(0.5), mean_embedding)

        binary_emb = mean_embedding.tolist()
        profile = trans_persona(binary_emb)
        ########

    return profile


def preprocess_alignx_one_file(data, is_train, external_persona):
    res = {'ica_raw': [], 'pba_raw': [], 'question': [], 'my_chosen': [], 'my_rejected': [], 'level': [], 'persona': []}
    mode_counts = {'UGC': 0, 'PAIR': 0, 'DEMO': 0, 'arbitrary': 0}
    for i, item in enumerate(data):
        if external_persona:
            persona = external_persona[i]
        else:
            persona = item['Demographic Information'] # it must exist
        level = persona

        question = item["prompt"]
        my_chosen = item['chosen']
        my_rejected = item['rejected']
        ugcs = item.get("User-Generated Content", [])
        pairs = item.get("Pair-wise Comparative Feedback", [])
        demo = item.get('Demographic Information', "")

        # for training set, we need to roughly mimic the distribution of the test sets
        if is_train:
            if len(ugcs) < 4 and len(pairs) < 4:
                mode_counts['DEMO'] += 1
                mode = 'DEMO'
            elif len(pairs) < 4:
                mode_counts['UGC'] += 1
                mode = 'UGC'
            elif len(ugcs) < 4:
                mode_counts['PAIR'] += 1
                mode = 'PAIR'
            else:
                # choose the mode with the smallest count
                mode = min(mode_counts, key=mode_counts.get)
                mode_counts[mode] += 1

            print(mode_counts)

            if mode == 'UGC':
                num_ugc_to_keep = 4
                num_pair_to_keep = 0
                demo = ""
            elif mode == 'PAIR':
                num_ugc_to_keep = 0
                num_pair_to_keep = 4
                demo = ""
            elif mode == 'DEMO':
                num_ugc_to_keep = 0
                num_pair_to_keep = 0
            elif mode == 'arbitrary':
                num_ugc_to_keep = random.randint(0, 4)
                num_pair_to_keep = random.randint(0, 4)
                if (num_ugc_to_keep > 0 or num_pair_to_keep > 0) and random.random() < 0.5:
                    demo = ""

            ugcs = random.sample(ugcs, num_ugc_to_keep)
            pairs = random.sample(pairs, num_pair_to_keep)
        else:
            pass

        ica_raw = f"## This user's demographic information:\n\n"
        if demo != "":
            ica_raw += demo + "\n\n"
        else:
            ica_raw += "Not provided.\n\n"

        ica_raw += f"## This user has commented on the following posts:\n\n"
        if ugcs != []:
            for i, it in enumerate(ugcs):
                ica_raw += f"### Post {i+1}:\n{it['prompt']}\n\n### Comment {i+1}:\n{it['comment']}\n\n"
        else:
            ica_raw += "Not provided.\n\n"

        ica_raw += f"## This user has preferred and dispreferred comments on the following posts:\n\n"
        if pairs != []:
            for i, it in enumerate(pairs):
                ica_raw += f"### Post {i+1}:\n{it['prompt']}\n\n### Preferred Comment {i+1}:\n{it['chosen']}\n\n### Dispreferred Comment {i+1}:\n{it['rejected']}\n\n"
        else:
            ica_raw += "Not provided.\n\n"
            
        if not is_train:
            profile = item['profile']
        else:
            history = ugcs + pairs
            if demo != "":
                history.append({'Preference Direction': item['Preference Direction']})
            profile = get_profile(history)

        pba_raw = f"{profile}\n" # no need to stuff anything, the profile is pretty clean

        res['ica_raw'].append(ica_raw)
        res['pba_raw'].append(pba_raw)
        res['question'].append(question)
        res['my_chosen'].append(my_chosen)
        res['my_rejected'].append(my_rejected)
        res['level'].append(level)
        res['persona'].append(persona)

        # print('===============')
        # print(f'ica_raw: {ica_raw}')
        # print(f'pba_raw: {pba_raw}')
        # print(f'question: {question}')
        # print(f'my_chosen: {my_chosen}')
        # print(f'my_rejected: {my_rejected}')
        # print(f'level: {level}')
        # print(f'persona: {persona}')

    res_ica = {'raw': res['ica_raw'], 'question': res['question'], 'my_chosen': res['my_chosen'], 'my_rejected': res['my_rejected'], 'level': res['level'], 'persona': res['persona']}
    res_pba = {'raw': res['pba_raw'], 'question': res['question'], 'my_chosen': res['my_chosen'], 'my_rejected': res['my_rejected'], 'level': res['level'], 'persona': res['persona']}

    return res_ica, res_pba


def preprocess_alignx_train_datasets(args):
    
    source_dir = ".cache/root/AlignX"
    cache_dir = ".cache/root"

    # download the dataset
    # repo_name = 'JinaLeejnl/AlignX'
    # snapshot_download(repo_name, cache_dir=source_dir, local_dir=source_dir, repo_type="dataset")

    # load all parquet files and sample a subset
    # parquet_files = [f for f in os.listdir(source_dir) if f.endswith('.parquet')]

    # data_list = []
    # print(len(parquet_files))

    # for idx, file in enumerate(parquet_files):
    #     file_path = os.path.join(source_dir, file)
    #     data = pd.read_parquet(file_path)
    #     data_list.extend(data.to_dict(orient='records'))
    #     print(idx)
        
    # print(len(data_list))

    # # randomly select 92000 samples
    # data_list = random.sample(data_list, 92000)

    # # convert ndarray to list
    # def ndarrays_to_lists(obj):
    #     if isinstance(obj, dict):
    #         return {k: ndarrays_to_lists(v) for k, v in obj.items()}
    #     elif isinstance(obj, list):
    #         return [ndarrays_to_lists(v) for v in obj]
    #     elif isinstance(obj, tuple):
    #         return tuple(ndarrays_to_lists(v) for v in obj)
    #     elif isinstance(obj, np.ndarray):
    #         return ndarrays_to_lists(obj.tolist())
    #     else:
    #         return obj

    # data_list = ndarrays_to_lists(data_list)

    # # save the data_list to a json file
    # with open(f"{source_dir}/train_92000.json", 'w') as f:
    #     json.dump(data_list, f, indent=4)

    with open(f"{source_dir}/train_92000.json", 'r') as f:
        data = json.load(f)

    res_ica, res_pba = preprocess_alignx_one_file(data, is_train=True, external_persona=None)

    # save
    dataset_ica = datasets.Dataset.from_dict(res_ica)
    dataset_pba = datasets.Dataset.from_dict(res_pba)

    dataset_ica.save_to_disk(f"{cache_dir}/preprocessed_datasets/alignx_ica_arbitrary_92000_train")
    dataset_pba.save_to_disk(f"{cache_dir}/preprocessed_datasets/alignx_pba_arbitrary_92000_train")


def preprocess_alignx_test_datasets(args):
    # this dataset is very problematic:
    # Through reverse-engineering, I found that, the existing profile in the Reddit_arbitrary split in the test set is determined as: 1. use the preference direction of ugc+pair+demo as history (the preference direction of demo is always used, no matter demo itself exists or not) 2. use 0.51 and 0.49 as threshold, not 0.6 and 0.4.
    # therefore, I will use the logic No.2 for the training set, but I will not use the logic No.1 because it is obviously flawed. I will use the existing profile for the Reddit_arbitrary split, to ensure performance compatibility.

    # for llmaaj, we use the DEMO as the gt persona because it is mostly close to the ground truth, and it is always available

    source_dir = ".cache/root/AlignX-test"
    cache_dir = ".cache/root"

    # download the dataset
    # repo_name = 'JinaLeejnl/AlignX-test'
    # snapshot_download(repo_name, cache_dir=source_dir, local_dir=source_dir, repo_type="dataset")

    fname = f"{source_dir}/Reddit_DEMO.json"
    with open(fname, 'r') as f:
        data = json.load(f)

    indices = list(range(len(data)))
    random.shuffle(indices)

    # shuffle the data
    data = [data[i] for i in indices]

    res_ica_demo, res_pba_demo = preprocess_alignx_one_file(data, is_train=False, external_persona=None)

    # save
    dataset_ica_demo = datasets.Dataset.from_dict(res_ica_demo)
    dataset_pba_demo = datasets.Dataset.from_dict(res_pba_demo)

    dataset_ica_demo.save_to_disk(f"{cache_dir}/preprocessed_datasets/alignx_ica_demo_test")
    dataset_pba_demo.save_to_disk(f"{cache_dir}/preprocessed_datasets/alignx_pba_demo_test")

    for test_variant in ["PAIR", "UGC", "arbitrary"]:
        fname = f"{source_dir}/Reddit_{test_variant}.json"
        with open(fname, 'r') as f:
            data = json.load(f)

        # shuffle the data
        data = [data[i] for i in indices]

        res_ica_variant, res_pba_variant = preprocess_alignx_one_file(data, is_train=False, external_persona=res_ica_demo['persona'])

        # save
        dataset_ica_variant = datasets.Dataset.from_dict(res_ica_variant)
        dataset_pba_variant = datasets.Dataset.from_dict(res_pba_variant)

        dataset_ica_variant.save_to_disk(f"{cache_dir}/preprocessed_datasets/alignx_ica_{test_variant.lower()}_test")
        dataset_pba_variant.save_to_disk(f"{cache_dir}/preprocessed_datasets/alignx_pba_{test_variant.lower()}_test")        


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(42)
    random.seed(42)
    transformers.set_seed(42)

    name = args.dataset_name
    split = args.split

    if name == "alignx":
        if split == "test":
            preprocess_alignx_test_datasets(args)
        else:
            preprocess_alignx_train_datasets(args)
    else:
        cache_dir = '.cache/root'

        data_dict = get_dataset(name, split, silent=False, cache_dir=cache_dir, include_level=True, include_persona=True)

        dataset = datasets.Dataset.from_dict(data_dict)

        # shuffle the dataset
        dataset = dataset.shuffle(seed=42)

        cache_path = f"{cache_dir}/preprocessed_datasets/{name}_{split}"
        dataset.save_to_disk(cache_path)