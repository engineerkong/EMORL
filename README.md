# DMORL: Distributed Multi-Objective RL Fine-Tuning Framework

## Original Paper
DMORL: Distributed Multi-Objective Reinforcement Learning Framework for Fine-Tuning Large Language Models in Counsellor Reflection Generation

## Code Overview
- `/dl_training.py`: distributed training for individual objectives (ours), uniform weighted and dynaopt training (baselines)
- `/dl_aggregation.py`: states-level aggregation (ours), parameters and logits-level aggregation (comparison) using hierarchical grid search
- `/dl_test.py`: general evaluation experiments for all models (for DMORL needs weights combination as input)
- `/utils_additional.py`: utils about load and save models, generation configs, convergence check etc.
- `/utils_lora.py`: utils about lora implementation
- `/model_empathy.py`: program to load and implement 'bert-empathy' scoring model
- `/dynaopt_lib`: original utils from 'dynaopt' repository with minimum change

## Dependency Setup
```
# Keep sure cuda and cuda toolkit installed
conda create --name myenv python=3.9
conda activate myenv
git clone https://github.com/engineerkong/DMORL.git
cd DMORL
pip install -r requirements.txt
```

## Datasets and Weights Download
Download [PAIR dataset](https://lit.eecs.umich.edu/downloads.html) and [Psch8k dataset](https://huggingface.co/datasets/EmoCareAI/Psych8k) into `DMORL/data`.

Download [reflection scoring weights](https://drive.google.com/file/d/1RPvMVLe7WS_spOvQI8FmPz6khI-MWWtA/view?usp=drive_link) into `DMORL/weights`.

## Quick Start
```
python dl_training.py
python dl_aggregation.py
```
