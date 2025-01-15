import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import copy
import random
import numpy as np
import wandb
import argparse
import time
import os

from dynaopt_lib import *
from lora_utils import *
from additional_utils import *

def training(args):
    """
    Train multiple LoRA models for different objectives using a T5 base model.    
    The function trains separate LoRA models for each objective specified, performs 
    periodic validation, and saves the trained parameters to the specified lora_path.
    Training progress and validation metrics are logged during training.

    Args:
        - seed (int): Random seed for reproducibility
        - model_path (str): Path to the pre-trained T5 base model
        - data_path (str): Path to training data CSV file containing input-output pairs
        - lora_path (str): Directory to save trained LoRA parameters
        - objectives (list):List of objectives to train LoRA models for. Each objective 
          will have its own LoRA model trained
        - train_batch_size (int): Batch size for training
        - val_batch_size (int): Batch size for validation
        - val_interval_size (int): Number of training steps between validation checks
        - num_runs (int): Number of training runs to perform
        - num_steps (int): Number of training steps per run
        - do_wandb (bool): Whether to use Weights & Biases for logging.

    Returns:
        dict: Dictionary containing trained LoRA parameters for each objective
    
    Example:
        args = parser.parse_args()
        model = training(args)
   """

    # Load seed, device
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = time.strftime("%Y%m%d_%H%M%S")


    # Load data
    train_data, val_data, test_data = get_data(args.data_path)
    train_dataloader = DataLoader(dataset=train_data, batch_size=args.train_batch_size,\
        sampler=RandomSampler(train_data), drop_last=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_data, batch_size=args.val_batch_size,\
        sampler=SequentialSampler(val_data), drop_last=True, collate_fn=collate_fn)

    # Train on multiple objective seperately
    for objective in args.objectives:
        print(f"Start training for objective:{objective}")
        
        # Initialize wandb
        if args.do_wandb:
            wandb.init(project="DynaDRL", group="DL_TRAINING", name=f"training_{timestamp}")
            wandb.define_metric("mean_reward", step_metric="data_consuming")
        else:
            wandb.init(project="DynaDRL", mode="disabled")

        # Load models
        model, tokenizer = get_model(
            args.model_path,  
            max_seq_length=128,
            max_output_length=64,
            lora=True
        )
        model = setup_lora_config(model)
        model.to(device)
        model.train()

        # Initialize reference model
        ref_model = copy.deepcopy(model)
        ref_model.to(device)
        ref_model.eval()
        ref_model.requires_grad_(False)
        
        # Setup optimizer and scaler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')

        # Create criterion and scorer 
        rl_crit = ReinforceCriterion(model, tokenizer, optimizer, scaler, ref_model=ref_model, kl_coeff=0.05)
        if objective == "reflection":
            scorer_model = ReflectionScoreDeployedCL(score_change=False, model_file= "./weights/reflection_scorer_weight.pt")
            scorer_model.type = "CLM"
            scorers = [{"name": "reflection", "model": scorer_model, "sign": 1, "weight": 1.0, "train": True}]
        else:
            scorers = [{"name": objective, "model": Multi(type=objective), "sign": 1, "weight": 1.0, "train": True}]
        scorer = ScorerWrapper(scorers, learning_mode = "single", scoring_method="logsum", max_batch_size=12)

        # Define generation parameters
        gen_params = {
            "max_new_tokens": 64,
            "early_stopping": True,
            "do_sample": True,
            "num_return_sequences": args.num_runs,
            "temperature": 1.0
        }

        # Training loop
        step_count = 0
        rewards_history = []
        while step_count < args.num_steps:
            print(f"step_count:{step_count}")
            
            # Training on train dataset
            for batch in train_dataloader:
                
                # Generate outputs with given prompts
                prompts = batch["prompts"]
                responses = batch["responses"]
                gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, \
                    return_tensors="pt", padding="longest", truncation=True)
                gen_input = {k: v.to(device) for k, v in gen_input.items()}
                gens_out = model.generate(input_ids=gen_input["input_ids"],\
                    attention_mask=gen_input["attention_mask"], **gen_params)
                generateds = tokenizer.batch_decode(gens_out, skip_special_tokens=True)
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
                gens_out = tokenizer.batch_encode_plus(cls_generateds, max_length=128, \
                    return_tensors="pt", padding="longest", truncation=True)["input_ids"]  
                prompts = [p for p in prompts for _ in range(args.num_runs)]
                responses = [r for r in responses for _ in range(args.num_runs)]

                # Calculate scorers - kSCST
                scorer_returns = scorer.score(prompts, generateds, responses=responses, step_count=step_count, bandit=None, chosen=None)
                total_scores = torch.FloatTensor(scorer_returns["total_scores"]).cuda()
                batch_scores = total_scores.reshape(args.train_batch_size, args.num_runs)
                mean_scores = batch_scores.mean(dim=1)
                max_scores = torch.max(batch_scores, dim=1).values 
                unlooped_mean_scores = torch.repeat_interleave(mean_scores, args.num_runs)
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
                
                # Update step_count when one batch is finished
                step_count += 1

                # Validation model with val dataset and show the mean rewards
                current_scores = { k["name"]+"_scores":[] for k in scorer.scorers }
                if step_count !=0 and  step_count % args.val_interval_size == 0:
                    for batch in val_dataloader:
                        responses = batch["responses"]
                        prompts = batch["prompts"]
                        gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, \
                            return_tensors="pt", padding="longest", truncation=True)
                        gen_input = {k: v.to(device) for k, v in gen_input.items()}
                        gens_out = model.generate(input_ids=gen_input["input_ids"],\
                            attention_mask=gen_input["attention_mask"], **gen_params)
                        generateds = tokenizer.batch_decode(gens_out, skip_special_tokens=True)
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
                        prompts = [p for p in prompts for _ in range(args.num_runs)]
                        responses = [r for r in responses for _ in range(args.num_runs)]
                        scorer_returns = scorer.rl_score(prompts, generateds, responses=responses, step_count=step_count, bandit=None)
                        for k,v in scorer_returns.items():
                            if k in current_scores:
                                current_scores[k].extend(v)
                    # Record mean reward (in single objective) and check if converged
                    mean_reward = [ np.mean(v) for k,v in current_scores.items() ]
                    rewards_history.append(mean_reward[0])
                    print("Mean Rewards", mean_reward[0])
                    wandb.log({"mean_reward": mean_reward[0], "data_consuming": step_count*args.train_batch_size})
                    if slope_convergence(rewards_history):
                        print("Training converged!")
                        print(f"Data consuming:{step_count*args.train_batch_size}")

        lora_params = acquire_lora_params(model)
        
        # Save LORA parameters
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.lora_path, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        filename = f"lora_{objective}.npz"
        npz_path = os.path.join(save_dir, filename)
        save_lora(lora_params, npz_path=npz_path)
        print(f"Saved LORA parameters to {npz_path}")

        # Finish wandb
        wandb.finish()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=429023)
    parser.add_argument('--model_path', type=str, default="models/google-t5-t5-base")
    parser.add_argument('--data_path', type=str, default="data/pair_data/pair_data.csv")
    parser.add_argument('--lora_path', type=str, default="lora_results/")
    parser.add_argument('--objectives', nargs='+', default=["reflection", "coherence", "fluency"])
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--val_interval_size', type=int, default=8)
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--do_wandb', type=int, default=0)
    
    args = parser.parse_args()
    save_args(args, "DL_TRAINING", "logs/")

    # Run single objective training
    training(args)

if __name__ == "__main__":
    main()