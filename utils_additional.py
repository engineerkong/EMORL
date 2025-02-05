import numpy as np
import random
import torch
import os
import csv
from scipy import stats
import time
import os
import json
from itertools import combinations, product
from typing import List, Dict, Tuple
from transformers import T5Tokenizer, T5ForConditionalGeneration

def set_seed(seed: int = 42) -> None:
    """
    Function for setting seed, which influences the randomness
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def get_model(model_path: str, max_seq_length: int = 90, lora = False, max_output_length: int = 90):
    """
    Function for loading tokenizer and model (T5-base), as well as set the configurations
    """
    print("Load model: ", model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.config.max_length = max_seq_length
    tokenizer.model_max_length = max_seq_length
    model.config.max_output_length = max_output_length
    return model, tokenizer

def get_data(data_path: str):
    """
    Function for loading dataset (PAIR) and spliting it into train/dev/test
    """
    print("Load data: ", data_path)
    data_split = [0.8, 0.1, 0.1]
    data = read_pair(data_path, True) + read_pair(data_path, False) 
    new_data = []
    prompts = set()
    for d in data:
        if d["prompt"] not in prompts:
            new_data.append(d)
            prompts.add(d["prompt"])
    data = new_data
    train_data = data[:int(len(data)*data_split[0])]
    dev_data = data[int(len(data)*data_split[0]):int(len(data)*(data_split[0]+data_split[1]))]
    test_data = data[int(len(data)*(data_split[0]+data_split[1])):]

    return train_data, dev_data, test_data

def get_esconv(data_path: str):
    print("Load data: ", data_path)
    data_split = [0.8, 0.1, 0.1]
    new_data = []
    with open(data_path, 'r') as f:
        data = json.load(f)
    for d in data:
        data_dict = {"prompt": d["situation"], "response": d["problem_type"]}
        new_data.append(data_dict)
    data = new_data
    train_data = data[:int(len(data)*data_split[0])]
    dev_data = data[int(len(data)*data_split[0]):int(len(data)*(data_split[0]+data_split[1]))]
    test_data = data[int(len(data)*(data_split[0]+data_split[1])):]
    return train_data, dev_data, test_data

def get_p8k(data_path: str):
    print("Load data: ", data_path)
    data_split = [0.8, 0.1, 0.1]
    new_data = []
    with open(data_path, 'r') as f:
        data = json.load(f)
    for d in data:
        data_dict = {"prompt": d["input"], "response": d["output"]}
        new_data.append(data_dict)
    data = new_data
    train_data = data[:int(len(data)*data_split[0])]
    dev_data = data[int(len(data)*data_split[0]):int(len(data)*(data_split[0]+data_split[1]))]
    test_data = data[int(len(data)*(data_split[0]+data_split[1])):]
    return train_data, dev_data, test_data

def read_pair(data_path: str, balanced_sampling=False):
    """
    Function for processing raw PAIR data into RL training format
    """
    id2header = { 0: "prompt", 1:"hq1", 2:"hq2", 3:"mq1", 4:"lq1", 5:"lq2", 6:"lq3", 7:"lq4", 8:"lq5"}
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    new_data = []
    for row in data:
        dic = {}
        for i,r in enumerate(row):
            dic[id2header[i]] = r
        new_data.append(dic)
    res_data = []
    for dat in new_data:
        pair_dat = generate_combs_pair(dat,balanced_sampling)
        res_data += pair_dat
    return res_data

def generate_combs_pair(dic, balanced_sampling=False):
    """
    Function for processing raw PAIR data into RL training format
    """
    prompt = [dic["prompt"]]    
    hq = combinations([dic["hq1"], dic["hq2"]],1)    
    mq = combinations([dic["mq1"]],1)    
    lq = combinations([dic["lq1"],dic["lq2"], dic["lq3"], dic["lq4"], dic["lq5"] ],1)    
    hq_products = [ list(flatten(prod)) for prod in product(prompt, hq) ]    
    mq_products = [ list(flatten(prod)) for prod in product(prompt, mq) ]
    lq_products = [ list(flatten(prod)) for prod in product(prompt, lq) ]
    hq_dics = [ {"prompt": prod[0], "response": prod[1], "level":2, "anti_response": random.sample(lq_products, 1)[0][1], "anti_level":0} for prod in hq_products ]
    mq_dics = [ {"prompt": prod[0], "response": prod[1], "level":1, "anti_response": random.sample(lq_products, 1)[0][1], "anti_level":0} for prod in mq_products ]
    lq_dics = [ {"prompt": prod[0], "response": prod[1], "level":0, "anti_response": random.sample(hq_products, 1)[0][1], "anti_level":2} for prod in lq_products ]
    if balanced_sampling:
        hq_dics = random.sample(hq_dics, 1) 
        mq_dics = random.sample(mq_dics, 1) 
        lq_dics = random.sample(lq_dics, len(hq_dics))
    return hq_dics + mq_dics + lq_dics

def flatten(l, ltypes=(list, tuple)):
    """
    Flatten Function
    """    
    ltype = type(l)    
    l = list(l)    
    i = 0    
    while i < len(l):    
        while isinstance(l[i], ltypes):    
            if not l[i]:    
                l.pop(i)    
                i -= 1     
                break    
            else:    
                l[i:i + 1] = l[i]    
        i += 1    
    return ltype(l)

def collate_fn(batch):
    """
    Function for collating dataset and setting seperator token
    """
    prompts = [item["prompt"] + " [SEP] " for item in batch]
    responses = [item["response"] for item in batch]
    return {"prompts": prompts, "responses": responses}
    
def save_args(args, func_name, save_dir):
    """
    Function for saving arguments as log file
    """
    timestamp = time.strftime("%Y%m%d_%H%M")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"args_{func_name}_{timestamp}.txt")
    with open(filename, 'w') as f:
        f.write(f"Command Line Arguments for {func_name}:\n")
        f.write("----------------------\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    print(f"Saved arguments to {filename}")

def check_convergence(rewards, window_size=20, std_threshold=0.015):
    """
    Function 1 for checking convergence of rewards, default threshold is 0.1
    """
    if len(rewards) < window_size:
        return False
        
    recent_rewards = rewards[-window_size:]
    std = np.std(recent_rewards)
    mean = np.mean(recent_rewards)
    
    cv = std / mean
    print(f"std/mean cv value:{cv}")
    return cv < std_threshold

def slope_convergence(rewards, window_size=50, slope_threshold=0.001):
    """
    Function 2 for checking convergence of rewards, default threshold is 0.001
    """
    if len(rewards) < window_size:
        return False
        
    recent_rewards = rewards[-window_size:]
    x = np.arange(window_size)
    slope, _, _, _, _ = stats.linregress(x, recent_rewards)
    
    return abs(slope) < slope_threshold
