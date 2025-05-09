import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import logging
from tqdm import tqdm
import copy
import glob
import wandb
import numpy as np

from meta_mha import MetaLearner
from model_empathy import *
from dynaopt_lib import *
from utils_lora import *
from utils_additional import *

def config_scorer(meta_model, tokenizer, ref_model, device):
    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(meta_model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device=device)

    def get_scorer(obj):
        config = scorer_configs.get(obj, lambda: scorer_configs["default"](obj))()
        scorer_model = config["model"]
        if "type" in config:
            scorer_model.type = config["type"]
        return {"name": obj, "model": scorer_model, "sign": 1, "weight": 1.0, "train": True}
    
    # Define training criterion and scorer
    scorer_configs = {
        "reflection": lambda: {
            "model": ReflectionScoreDeployedCL(score_change=False, model_file="./weights/reflection_scorer_weight.pt"),
            "type": "CLM"},
        "empathy": lambda: {
            "model": EmpathyScoreDeployedCL(score_change=False),
            "type": "CLM"},
        "default": lambda o: {
            "model": Multi(type=o)}
    }
    objectives = ["reflection", "empathy", "fluency"]
    scorers = [get_scorer(obj) for obj in objectives]
    rl_crit = ReinforceCriterion(model=meta_model, tokenizer=tokenizer, optimizer=optimizer, scaler=scaler, ref_model=ref_model, kl_coeff=0.05)
    scorer = ScorerWrapper(scorers, learning_mode="weighted", scoring_method="fixed_logsum", max_batch_size=8)

    return optimizer, scaler, scorer, rl_crit

def meta_train(meta_model, tokenizer, train_dataloader, val_dataloader, optimizer, scaler, scorer, rl_crit, gens_params, 
                train_batch_size, val_batch_size, val_interval_size, num_runs, num_steps, device):
    step_count = 0
    training = True
    while step_count < num_steps and training:
        print(f"Step count:{step_count}")
        
        # Training on train dataset, batches 15
        for batch in train_dataloader:
            # Generate inputs
            prompts = batch["prompts"]
            responses = batch["responses"]
            gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, \
                return_tensors="pt", padding="longest", truncation=True)
            gen_input = {k: v.to(device) for k, v in gen_input.items()}

            # meta_model forward pass
            meta_model.train()
            generated_tokens = meta_model.forward(gen_input, num_runs, train_batch_size, max_output_length=64)

            # Generate outputs
            generateds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            generateds = [ g.split("[CLS]") for g in generateds]
            new_generateds = []
            for g_list in generateds:
                if len(g_list) <= 1:
                    new_generateds.append([g_list[0].strip()])
                else:
                    new_generateds.append([x.strip() for x in g_list[:-1]])
            generateds = new_generateds
            cls_generateds = [ [ x.strip() + " [CLS]" for x in g] for g in generateds ]
            cls_generateds = [ " ".join(g) for g in cls_generateds]
            generateds = [ " ".join(g) for g in generateds]
            generateds = [ g.replace("<pad>", "").strip() for g in generateds]
            generateds = [g.replace("[CLS]", "").strip() for g in generateds]
            gens_out = tokenizer(generateds, max_length=64, \
                return_tensors="pt", padding="max_length", truncation=True)["input_ids"]
            prompts = [p for p in prompts for _ in range(num_runs)]
            responses = [r for r in responses for _ in range(num_runs)]

            # Calculate scorers - kSCST/logsum
            scorer_returns = scorer.score(prompts, generateds, responses=responses, step_count=step_count, \
                                          bandit=None, chosen=None, extras={"reflection": 1/3, "empathy": 1/3, "fluency": 1/3})
            print(f"Reflection: {np.mean(scorer_returns['reflection_scores'])}, \
                    Empathy: {np.mean(scorer_returns['empathy_scores'])}, \
                    Fluency: {np.mean(scorer_returns['fluency_scores'])}")
            total_scores = torch.FloatTensor(scorer_returns["total_scores"]).cuda()
            batch_scores = total_scores.reshape(train_batch_size, num_runs)
            mean_scores = batch_scores.mean(dim=1)
            unlooped_mean_scores = torch.repeat_interleave(mean_scores, num_runs)
            normalized_rewards = (unlooped_mean_scores - total_scores)

            # Calculate loss with KL penalty
            loss = rl_crit(prompts, gens_out, normalized_rewards)

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=2.0, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update step_count when one batch is finished
            step_count += 1

def main(model_name, data_path, lora_path, device, train_batch_size, val_batch_size, val_interval_size, num_runs, num_steps):
    # Load data
    train_data, val_data, test_data = get_data(data_path)
    train_dataloader = DataLoader(dataset=train_data, batch_size=train_batch_size,\
        sampler=RandomSampler(train_data), drop_last=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_data, batch_size=val_batch_size,\
        sampler=RandomSampler(val_data), drop_last=True, collate_fn=collate_fn)

    # Load model and tokenizer
    model, tokenizer = get_model(
        model_name,  
        max_seq_length=128,
        max_output_length=64,
        lora=True
    )
    model.to(device)

    # Load reference model for KL divergence
    ref_model = copy.deepcopy(model)
    ref_model.to(device)
    ref_model.eval()
    ref_model.requires_grad_(False)

    # Initialize meta-learner
    meta_learner = MetaLearner(
        base_model=model,
        n_heads=8,
        embed_dim=768,
        num_models=3,
        feed_forward_hidden=512,
        normalization='batch',
        batch_size=train_batch_size,
        num_runs=num_runs,
        max_output_length=64,
        device='cuda'
    )
    # Load LoRA updates
    pattern = os.path.join(lora_path, f"lora_*.npz")
    matching_files = glob.glob(pattern)
    lora_updates = []
    for matching_file in matching_files:
        print(f"Loading LoRA parameters from {matching_file}")
        lora_params = load_lora(matching_file)
        lora_updates.append(lora_params)
    meta_learner.config_models(lora_updates)

    # Define generation parameters
    gen_params = {
        "max_new_tokens": 64,
        "early_stopping": True,
        "do_sample": True,
        "num_return_sequences": num_runs,
        "temperature": 1.0,
        "pad_token_id": tokenizer.eos_token_id  # DialoGPT uses eos_token as pad_token
    }

    # Train the meta-learner
    optimizer, scaler, scorer, rl_crit = config_scorer(meta_learner, tokenizer, ref_model, device)
    meta_train(meta_learner, tokenizer, train_dataloader, val_dataloader, optimizer, scaler, scorer, rl_crit, gen_params, 
            train_batch_size, val_batch_size, val_interval_size, num_runs, num_steps, device)

if __name__ == "__main__":
    wandb.init(project="Improved_EMORL", mode="disabled")

    # Default configurations
    model_name = "google-t5/t5-base"
    data_path = "data/PAIR/pair_data.csv"
    lora_path = "lora_results/results_example"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_batch_size = 8
    val_batch_size = 8
    val_interval_size = 8
    num_runs = 3
    num_steps = 1000

    main(model_name, data_path, lora_path, device, train_batch_size, val_batch_size, val_interval_size, num_runs, num_steps)