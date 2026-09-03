import datasets
import torch
from utils import TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
# from transformers import AutoTokenizer
from torch.utils.data import Dataset
import functools


def shots_to_text(
    fewshot_prompts: List[str], 
    fewshot_preferred: List[str], 
    fewshot_dispreferred: List[str], 
    include_dispreferred: bool = False, 
):
    """Convert a list of few-shot examples and a final question to a single string using markdown."""
    assert len(fewshot_prompts) == len(fewshot_preferred) == len(fewshot_dispreferred), "Few-shot examples must have the same length"
    num_shots = len(fewshot_prompts)
    fewshot_text = ""
    for i, (prompt, preferred, dispreferred) in enumerate(zip(fewshot_prompts, fewshot_preferred, fewshot_dispreferred)):
        fewshot_text += "## Example " + str(i + 1) + "\n"
        fewshot_text += "### Prompt \n" + prompt + "\n"
        if include_dispreferred:
            fewshot_text += "### Preferred Response \n" + preferred + "\n"
            fewshot_text += "### Dispreferred Response \n" + dispreferred + "\n"
        else:
            fewshot_text += "### Response \n" + preferred + "\n"
        fewshot_text += "\n"

    return fewshot_text
    

def get_elix(
    split: str, 
    silent: bool = False, 
    cache_dir: str = None, 
    num_shots: int = 4, 
    include_dispreferred: bool = True, 
    include_level: bool = True, 
    autolabel: bool = False,
    include_persona: bool = False,
    multi_user=False,
    scorer_match=True,
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading ELIX (regen) dataset ({split} split) from Huggingface...')
    if autolabel:
        dataset = datasets.load_dataset('Asap7772/elix_generations_autolabel', split=split, cache_dir=cache_dir)
    elif multi_user:
        dataset = datasets.load_dataset('Asap7772/elix_multexpert_preferences_gpt4o_pref', split=split, cache_dir=cache_dir)
    else:
        dataset = datasets.load_dataset('Asap7772/elix_generations_gpt4omini_pref', split=split, cache_dir=cache_dir)
    df = dataset.to_pandas()
    print('done')
    
    if scorer_match:
        df = df[(df['scorer_level'] == df['level_x']) | (df['scorer_level'] == df['level_y'])]
    
    data = {'raw': [], 'question': [], 'my_chosen': [], 'my_rejected': []}
    if include_level:
        data['level'] = []
    if include_persona:
        data['persona'] = []

    for _ in range(2): #((num_shots + 1)):
        for level, level_df in tqdm.tqdm(df.groupby('scorer_level'), desc='Processing levels', disable=silent):
            level_df = level_df.sample(frac=1).reset_index(drop=True)
            index_df = 0
            while index_df < len(level_df):
                start = index_df
                end = index_df + (num_shots + 1)  # Get N few-shot examples and 1 prompt
                if end > len(level_df):
                    break
                rows = level_df.iloc[start:end]

                fewshot_prompts = []
                fewshot_preferred = []
                fewshot_dispreferred = []
                for shot_idx in range(num_shots):
                    x = rows['prompt'].iloc[shot_idx].strip()
                    y1 = rows['response_x'].iloc[shot_idx].strip()
                    y2 = rows['response_y'].iloc[shot_idx].strip()

                    label = rows['label'].iloc[shot_idx]
                    yw = y1 if label == 0 else y2
                    yl = y2 if label == 0 else y1

                    fewshot_prompts.append(x)
                    fewshot_preferred.append(yw)
                    fewshot_dispreferred.append(yl)

                final_question = rows['prompt'].iloc[num_shots].strip()
                y1_last = rows['response_x'].iloc[num_shots].strip()
                y2_last = rows['response_y'].iloc[num_shots].strip()
                label_last = rows['label'].iloc[num_shots]
                final_chosen = y1_last if label_last == 0 else y2_last
                final_rejected = y2_last if label_last == 0 else y1_last

                fewshot_text = shots_to_text(
                    fewshot_prompts=fewshot_prompts,
                    fewshot_preferred=fewshot_preferred,
                    fewshot_dispreferred=fewshot_dispreferred,
                    include_dispreferred=include_dispreferred,
                )
                

                data['raw'].append(fewshot_text)
                data['question'].append(final_question)
                data['my_chosen'].append(final_chosen)
                data['my_rejected'].append(final_rejected)
                if include_level:
                    data['level'].append(level)
                if include_persona:
                    data['persona'].append(f"The user is {level}.")

                index_df = end


    return data


def get_review(
    split: str, 
    silent: bool=False, 
    cache_dir: str=None, 
    num_shots=4, 
    include_dispreferred=True, 
    include_level=True, 
    autolabel=False,
    include_persona=False,
    do_interpolation=False,
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading Review dataset ({split} split) from Huggingface...')
    if autolabel:
        dataset = datasets.load_dataset('Asap7772/steered_reviews_full_autolabel', split=split, cache_dir=cache_dir)
    else:
        dataset = datasets.load_dataset('Asap7772/steered_reviews_full_autolabel_gpt4o_pref', split=split, cache_dir=cache_dir)
    df = dataset.to_pandas()
    print('done')
    

    data = {'raw': [], 'question': [], 'my_chosen': [], 'my_rejected': []}
    if include_level:
        data['level'] = []
    if include_persona:
        data['persona'] = []

    for _ in range(1): #((num_shots + 1)):
        for level, level_df in tqdm.tqdm(df.groupby('scorer_level'), desc='Processing levels', disable=silent):
            if not do_interpolation:
                if level not in ['positive', 'negative', 'concise', 'verbose']:
                    continue

            level_df = level_df.sample(frac=1).reset_index(drop=True)
            index_df = 0
            while index_df < len(level_df):
                start = index_df
                end = index_df + (num_shots + 1)  # Get N few-shot examples and 1 prompt
                if end > len(level_df):
                    break
                rows = level_df.iloc[start:end]
                
                fewshot_prompts = []
                fewshot_preferred = []
                fewshot_dispreferred = []
                for shot_idx in range(num_shots):
                    x = rows['prompt'].iloc[shot_idx].strip()
                    y1 = rows['response_x'].iloc[shot_idx].strip() 
                    y2 = rows['response_y'].iloc[shot_idx].strip()
                                        
                    label = rows['label'].iloc[shot_idx]
                    yw = y1 if label == 0 else y2
                    yl = y2 if label == 0 else y1
                    
                    fewshot_prompts.append(x)
                    fewshot_preferred.append(yw)
                    fewshot_dispreferred.append(yl)
                
                mapped_level = {
                    'negative': 'This user prefers negative reviews.',
                    'positive': 'This user prefers positive reviews.',
                    'concise': 'This user prefers concise reviews.',
                    'verbose': 'This user prefers verbose reviews.',
                    'positive+concise': 'This user prefers positive and concise reviews.',
                    'negative+concise': 'This user prefers negative and concise reviews.',
                    'positive+verbose': 'This user prefers positive and verbose reviews.',
                    'negative+verbose': 'This user prefers negative and verbose reviews.',
                }
                persona_scorer = mapped_level[level]
                final_question = rows['prompt'].iloc[num_shots].strip()
                y1_last = rows['response_x'].iloc[num_shots].strip()
                y2_last = rows['response_y'].iloc[num_shots].strip()
                label_last = rows['label'].iloc[num_shots]
                final_chosen = y1_last if label_last == 0 else y2_last
                final_rejected = y2_last if label_last == 0 else y1_last
                
                fewshot_text = shots_to_text(
                    fewshot_prompts=fewshot_prompts,
                    fewshot_preferred=fewshot_preferred,
                    fewshot_dispreferred=fewshot_dispreferred,
                    include_dispreferred=include_dispreferred,
                )

                data['raw'].append(fewshot_text)
                data['question'].append(final_question)
                data['my_chosen'].append(final_chosen)
                data['my_rejected'].append(final_rejected)
                if include_level:
                    data['level'].append(level)
                if include_persona:
                    data['persona'].append(persona_scorer)
                
                index_df = end
    return data


def get_roleplay(
    split: str, 
    silent: bool=False, 
    cache_dir: str=None, 
    num_shots=8, 
    include_dispreferred=True, 
    include_persona=False,
    include_level=False
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading roleplay dataset ({split} split) from Huggingface...')
    dataset_name = f"sher222/persona-iterative-responses"
    dataset = datasets.load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    print("loading data from", dataset_name)
    df = dataset.to_pandas()
    print('done')

    data = {'raw': [], 'question': [], 'my_chosen': [], 'my_rejected': []}
    if include_level:
        data['level'] = []
    if include_persona:
        data['persona'] = []

    for _ in range(1): #((num_shots + 1)):
        for level, level_df in tqdm.tqdm(df.groupby('level'), desc='Processing levels', disable=silent):
            level_df = level_df.sample(frac=1).reset_index(drop=True)
            index_df = 0
            while index_df < len(level_df):
                start = index_df
                end = index_df + (num_shots + 1)  # Get N few-shot examples and 1 prompt
                if end > len(level_df):
                    break
                rows = level_df.iloc[start:end]
                
                fewshot_prompts = []
                fewshot_preferred = []
                fewshot_dispreferred = []

                for shot_idx in range(num_shots):
                    x = rows['x'].iloc[shot_idx].strip()
                    yw = rows['yw'].iloc[shot_idx].strip()
                    yl = rows['yl'].iloc[shot_idx].strip()

                    fewshot_prompts.append(x)
                    fewshot_preferred.append(yw)
                    fewshot_dispreferred.append(yl)

                persona_scorer = str(rows['score_persona'].iloc[num_shots])
                final_question = rows['x'].iloc[num_shots].strip()
                final_chosen = rows['yw'].iloc[num_shots].strip()
                final_rejected = rows['yl'].iloc[num_shots].strip()

                fewshot_text = shots_to_text(
                    fewshot_prompts=fewshot_prompts,
                    fewshot_preferred=fewshot_preferred,
                    fewshot_dispreferred=fewshot_dispreferred,
                    include_dispreferred=include_dispreferred,
                )

                data['raw'].append(fewshot_text)
                data['question'].append(final_question)
                data['my_chosen'].append(final_chosen)
                data['my_rejected'].append(final_rejected)
                if include_level:
                    data['level'].append(level)
                if include_persona:
                    data['persona'].append(persona_scorer)
                
                index_df = end
    return data


def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, include_level=False, include_persona=False):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'elix_4shot':
        data = get_elix(split, silent=silent, cache_dir=cache_dir, include_dispreferred=True, include_level=include_level, autolabel=True, num_shots=4, include_persona=include_persona)
    elif name == 'review_4shot':
        data = get_review(split, silent=silent, cache_dir=cache_dir, include_dispreferred=True, include_level=include_level, autolabel=False, num_shots=4, include_persona=include_persona)
    elif name == 'roleplay_8shot':
        data = get_roleplay(split, silent=silent, cache_dir=cache_dir, include_dispreferred=True, include_level=include_level, num_shots=8, include_persona=include_persona)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    return data

