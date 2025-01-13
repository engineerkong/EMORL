import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import copy
import random
import numpy as np
import wandb
from additional_utils import (
    set_seed,
    get_model,
    get_data
)
from dynaopt_lib import *


def run_rl_train(
    model_path: str,
    data_path: str,
    max_seq_length: int = 128,
    max_output_length: int = 45,
    train_batch_size: int = 1,
    num_runs: int = 10,
    # rl_run_size = 20,
    # rl_validation_step = 10, 
    # rl_validation_size = 200,
    num_steps: int = 1000,
    # use_apex = True
    kl_coeff: float = 0.05,
    seed: int = 42,
):
    """
    Simplified reinforcement learning training function for text generation.
    
    Args:
        model_name: Base model name
        model_start_dir: Directory containing pretrained model
        max_seq_length: Maximum input sequence length
        max_output_length: Maximum output sequence length
        learning_rate: Learning rate for optimization
        batch_size: Training batch size
        num_runs: Number of generation runs per input
        num_steps: Total training steps
        kl_coeff: KL divergence coefficient
        seed: Random seed
    """
    # Set random seed
    set_seed(seed)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize models
    model, tokenizer = get_model(
        model_path,  
        max_seq_length=max_seq_length,
        max_output_length=max_output_length,
        lora=True
    )
    model.to(device)
    
    # Initialize reference model
    ref_model = copy.deepcopy(model)
    ref_model.to(device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    
    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() # apex scaler

    # Prepare data
    train_data, val_data, test_data = get_data(data_path)
    
    # Create data loader
    def collate_fn(batch):
        prompts = [item["prompt"] + " [SEP] " for item in batch]
        responses = [item["response"] for item in batch]
        return {"prompts": prompts, "responses": responses}
    
    train_dataloader = DataLoader(dataset=train_data, batch_size=train_batch_size,\
        sampler=RandomSampler(train_data), drop_last=True, collate_fn=collate_fn)
    # rl_val_dataloader = DataLoader(dataset=dev_data, batch_size=rl_run_size,\
    #     sampler=SequentialSampler(dev_data), drop_last=True, collate_fn=collate_fn)
  
    rl_crit = ReinforceCriterion(model, tokenizer, optimizer, scaler, ref_model=ref_model, kl_coeff=kl_coeff)
    scorers = [{"name": "perplexity", "model": Multi(type="perplexity", experiment="test"), "sign": 1, "weight": 1.0, "train": True}]
    scorer = ScorerWrapper(scorers, learning_mode = "single", \
         scoring_method="logsum", max_batch_size=12)  

    # Training loop
    step_count = 0
    model.train()
    
    while step_count < num_steps:
        print(f"step_count:{step_count}")
        for batch in train_dataloader:
            prompts = batch["prompts"]
            responses = batch["responses"]
            
            # Generate text
            gen_params = {
                "max_new_tokens": max_output_length,
                "early_stopping": True,
                "do_sample": True,
                "num_return_sequences": num_runs,
                "temperature": 1.0
            }
            
            with torch.amp.autocast('cuda'):            
            # Forward pass
                gen_input = tokenizer.batch_encode_plus(prompts, max_length=max_seq_length, \
                    return_tensors="pt", padding="longest", truncation=True)
                gen_input = {k: v.to(device) for k, v in gen_input.items()}
                gens_out = model.generate(input_ids=gen_input["input_ids"],\
                    # decoder_start_token_id=tokenizer.bos_token_id,\
                    attention_mask=gen_input["attention_mask"], **gen_params)
                # Decode generations
                generateds = tokenizer.batch_decode(gens_out, skip_special_tokens=True)
                generateds = [ [ x.strip() for x in g.split("[CLS]")[:-1]] for g in generateds]
                cls_generateds = [ [ x.strip() + " [CLS]" for x in g] for g in generateds ]
                cls_generateds = [ " ".join(g) for g in cls_generateds]
                generateds = [ " ".join(g) for g in generateds]
                generateds = [ g.replace("<pad>", "").strip() for g in generateds]
                generateds = [g.replace("[CLS]", "").strip() for g in generateds]
                gens_out = tokenizer.batch_encode_plus(cls_generateds, max_length=max_output_length, \
                    return_tensors="pt", padding="longest", truncation=True)["input_ids"]  
                prompts = [p for p in prompts for _ in range(num_runs)]
                responses = [r for r in responses for _ in range(num_runs)]
                
                # Calculate rewards - K-SCST
                scorer_returns = scorer.score(prompts, generateds, responses=responses, step_count=step_count, bandit=None, chosen=None)
                total_scores = torch.FloatTensor(scorer_returns["total_scores"]).cuda()
                batch_scores = total_scores.reshape(train_batch_size, num_runs)
                mean_scores = batch_scores.mean(dim=1)
                max_scores = torch.max(batch_scores, dim=1).values 
                unlooped_mean_scores = torch.repeat_interleave(mean_scores, num_runs)
                normalized_rewards = (unlooped_mean_scores - total_scores)
                n_diff_pos, n_diff_neg = (normalized_rewards<-0.02).long().sum().item(), (normalized_rewards>0.02).long().sum().item()
                
                # Calculate loss with KL penalty
                loss = rl_crit(prompts, gens_out, normalized_rewards)
                
                # Backward pass and optimization
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                step_count += 1
    
    return model, tokenizer

if __name__ == "__main__":
    
    wandb.init(project="dynamic weighting", mode="disabled")
    run_rl_train(model_path="/home/ubuntu/DynaDRL/models/google-t5-t5-base", data_path="/home/ubuntu/DynaDRL/data/pair_data/pair_data.csv")