# DMORL: Distributed Multi-Objective RL Fine-Tuning Framework

## Original Paper
DMORL: Distributed Multi-Objective Reinforcement Learning Framework for Fine-Tuning Large Language Models in Counsellor Reflection Generation

## Documentation Structure
- '/dl_training.py':
- '/dl_aggregation.py':
- '/dl_test.py':
- '/utils_additional.py':
- '/utils_lora.py':
- '/model_empathy.py':
- '/dynaopt_lib':

## Dependency Setup
'''
% Keep sure cuda and cuda toolkit installed
conda create --name myenv python=3.9
conda activate myenv
git clone https://github.com/engineerkong/DMORL.git
cd DMORL
pip install -r requirements.txt
'''

## Datasets and Weights Download
Download PAIR dataset https://lit.eecs.umich.edu/downloads.html and Psch8k dataset https://huggingface.co/datasets/EmoCareAI/Psych8k into 'DMORL/data'.

Download reflection scoring weights https://drive.google.com/file/d/1RPvMVLe7WS_spOvQI8FmPz6khI-MWWtA/view?usp=drive_link into 'DMORL/weights'.

## Quick Start
'''
python dl_training.py
python dl_aggregation.py
'''