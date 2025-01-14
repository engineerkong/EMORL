import numpy as np
import random
import torch
import os
import json
import csv
from itertools import combinations, product
from typing import List, Dict
from typing import Optional, List, Dict, Tuple
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BartModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


# TODO: use QLORA
def get_model(model_path: str, max_seq_length: int = 90, lora = False, max_output_length: int = 90):

    print("="*30)
    print("Load model:", model_path)
    print("="*30)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.config.max_length = max_seq_length
    tokenizer.model_max_length = max_seq_length
    model.config.max_output_length = max_output_length
    # if lora:                                                                                                                           
    #     if hasattr(model, "enable_input_require_grads"):
    #         model.enable_input_require_grads()
    #     else:
    #         def make_inputs_require_grad(module, input, output):
    #             output.requires_grad_(True)
    #         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    #     model = make_lora_model(model, lora_r = 8, lora_alpha = 16, \
    #             lora_dropout = 0.05, lora_target_modules = [ "q_proj", "v_proj"]) 
    return model, tokenizer

def get_data(data_path: str):

    print("="*30)
    print("Load data:", data_path)
    print("="*30)
    data_split = [0.5, 0.1, 0.4]
    data = read_umich_pair(data_path, True) + read_umich_pair(data_path, False) 
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

def read_umich_pair(data_path: str, balanced_sampling=False):
    """
    Function for processing raw UMICH data into MMLE/RL training format
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

def collate_fn(batch):
    prompts = [item["prompt"] + " [SEP] " for item in batch]
    responses = [item["response"] for item in batch]
    return {"prompts": prompts, "responses": responses}
    
def save_args(args, func_name, save_dir):
   # Create timestamp for filename
   timestamp = time.strftime("%Y%m%d_%H%M%S")
   # Create directory if it doesn't exist
   os.makedirs(save_dir, exist_ok=True)
   # Create filename with timestamp
   filename = os.path.join(save_dir, f"args_{timestamp}.txt")
   # Save arguments to file
   with open(filename, 'w') as f:
       f.write(f"Command Line Arguments for {func_name}:\n")
       f.write("----------------------\n")
       for arg, value in vars(args).items():
           f.write(f"{arg}: {value}\n")
   print(f"Saved arguments to {filename}")

# TODO: should it not be 2-1,1-0,2-0?
def generate_combs_pair(dic, balanced_sampling=False):
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