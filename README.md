# POPI: Personalizing LLMs via Optimized Natural Language Preference Inference

## Setup
Create a venv with requirements listed in requirements.txt, ideally with python 3.12.
```bash
conda create --name POPI python=3.12
conda activate POPI
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn
```


## Run
Set API keys:
```
export OPENAI_API_KEY=xxx
export HF_TOKEN=xxx
export WANDB_API_KEY=xxx
```
Preprocess the datasets:
```
python ./src/preprocess_dataset.py --dataset_name elix_4shot --split train
python ./src/preprocess_dataset.py --dataset_name elix_4shot --split test
```
Run the training pipeline:
```
sh run.sh
```
Evaluate the trained model:
```
sh run_generate_for_evaluation.sh
```
Then the results can be found in ```.cache/root/results/popi/elix_4shot/dpo```