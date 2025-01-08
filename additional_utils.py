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

class ReinforceCriterion:
    def __init__(self, model, tokenizer, optimizer, scaler, \
                 reward_shaping="default", ref_model=None, ref_tokenizer=None, ref_optimizer=None, kl_coeff=0.05):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.eos_id = self.tokenizer.eos_token_id
        self.scaler = scaler
        self.reward_shaping = reward_shaping
        self.ref_model = ref_model
        self.kl_coeff = kl_coeff
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")
    def __call__(self, prompt_inputs, decoded_tokens, rewards, train_model=True, sampled_actions=None, freeze_responder=False):
        if not train_model:
            return 0.0
        assert len(prompt_inputs)==len(decoded_tokens), "There's a shape mismatch between inputs and outputs %d != %d" % (len(prompt_inputs), len(decoded_tokens))
        encoded_prompt= self.tokenizer(prompt_inputs, return_tensors="pt", padding="longest", truncation=True, max_length=512)
        encoded_prompt = {k: v.to(self.model.device) for k, v in encoded_prompt.items()}
        decoded_tokens_tensor = decoded_tokens.to(self.model.device)
        output = self.model(input_ids=encoded_prompt['input_ids'], \
                attention_mask=encoded_prompt['attention_mask'], labels = decoded_tokens_tensor)
        logits = output.logits
        decoded_tokens = decoded_tokens_tensor.tolist()
        selected_logprobs = select_logprobs(logits, decoded_tokens, self.eos_id)
        if self.ref_model is not None:
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids=encoded_prompt['input_ids'], \
                    attention_mask=encoded_prompt['attention_mask'], labels = decoded_tokens_tensor).logits
                ref_selected_logprobs = select_logprobs(ref_logits, decoded_tokens, self.eos_id)
                kl = self.kl_loss(F.log_softmax(ref_logits, dim=-1), F.softmax(logits, dim=-1))
                kl = torch.sum(kl, dim=-1)
                kl_mask = decoded_tokens_tensor != self.tokenizer.pad_token_id
                kl = reduce_mean(kl, kl_mask)
        if self.ref_model is not None:
            loss = torch.mean(rewards * (selected_logprobs + self.kl_coeff * kl))
            wandb.log({"KL term": torch.mean(rewards * self.kl_coeff * kl).item()})
            wandb.log({"KL": torch.mean(kl).item()})
            wandb.log({"Reward": torch.mean(rewards*selected_logprobs).item()})
        else:
            loss = torch.mean(rewards * selected_logprobs)
        return loss  