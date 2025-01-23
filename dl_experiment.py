import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import copy
import random
import numpy as np
import wandb
import argparse
import time
import os
import copy

from dynaopt_lib import *
from utils_lora import *
from utils_additional import *

def dynaopt(
    seed: int = 429023,
    model_path: str = "models/google-t5-t5-base",
    data_path: str = "data/pair_data/pair_data.csv",
    experiment_path: str = "experiment_results/",
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    val_interval_size: int = 8,
    val_history_size: int = 10,
    num_runs: int = 10, 
    num_steps: int = 5000,
    do_wandb: int = 0
    ):
    """
    Implements dynamic optimization for RL reward modeling, based on the DynaOpt framework.

    Parameters:
        - seed (int): Random seed for reproducibility.
        - model_path (str): Path to the pre-trained T5 model.
        - data_path (str): Path to the training data CSV file.
        - max_seq_length (int): Maximum input sequence length.
        - max_output_length (int): Maximum output sequence length.
        - train_batch_size (int): Batch size for training.
        - val_batch_size (int): Batch size for validation.
        - val_interval_size (int): Number of steps between validation checks.
        - num_runs (int): Number of training runs to perform.
        - num_steps (int): Number of training steps per run.
        - do_wandb (bool): Whether to use Weights & Biases for logging.

    This is a simplified version of the original DynaOpt implementation 
    (https://github.com/MichiganNLP/dynaopt/blob/main/rl_train.py), 
    adapted to maintain consistency with DL_TRAINING settings.
    """

    # Load seed, device
    set_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Initialize wandb
    if do_wandb:
        wandb.init(project="DynaDRL", group="DL_EXPERIMENT", name=f"dynaopt_{timestamp}")
        wandb.define_metric("mean_reward", step_metric="data_consuming")
    else:
        wandb.init(project="DynaDRL", mode="disabled")

    # Load data
    train_data, val_data, test_data = get_data(data_path)
    train_dataloader = DataLoader(dataset=train_data, batch_size=train_batch_size,\
        sampler=RandomSampler(train_data), drop_last=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_data, batch_size=val_batch_size,\
        sampler=SequentialSampler(val_data), drop_last=True, collate_fn=collate_fn)

    # Load models
    model, tokenizer = get_model(
        model_path,  
        max_seq_length=128,
        max_output_length=64,
        lora=False
    )
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

    # Define generation parameters
    gen_params = {
        "max_new_tokens": 64,
        "early_stopping": True,
        "do_sample": True,
        "num_return_sequences": num_runs,
        "temperature": 1.0
    }

    # Create criterion and scorer 
    rl_crit = ReinforceCriterion(model, tokenizer, optimizer, scaler, ref_model=ref_model, kl_coeff=0.05)
    scorer_model = ReflectionScoreDeployedCL(score_change=False, model_file= "./weights/reflection_scorer_weight.pt")
    scorer_model.type = "CLM"
    scorers = [{"name": "reflection", "model": scorer_model, "sign": 1, "weight": 1.0, "train": True},
                {"name": "fluency", "model": Multi(type="fluency"), "sign": 1, "weight": 1.0, "train": True},
                {"name": "coherence", "model": Multi(type="coherence"), "sign": 1, "weight": 1.0, "train": True}]
    scorer = ScorerWrapper(scorers, learning_mode = "bandit_weighted", scoring_method="logsum", max_batch_size=12)

    # Return scores back to update bandit weights
    bandit = Exp3(len(scorers)+1)
    bandit_history = []
    bandit_weight_history = []
    bandit_arm_weight_history = []
    chosen = bandit.draw()
    last_chosen = chosen
    print("Bandit arm pulled:", chosen)
    rl_scorer_history = { k["name"]+"_scores":[] for k in scorer.scorers }
    bandit_pulls = { i:0 for i in range(len(scorer.scorers)+1) } 
    bandit_pulls[last_chosen] += 1
    bandit_history.append(last_chosen)
    bandit_arm_weight_history.append(bandit.weights.copy())
    print("Bandit Pull:", bandit_pulls)
    print("Bandit Weights:", bandit.weights)

    # Training loop
    step_count = 0
    rewards_history = []
    while step_count < num_steps:
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
            prompts = [p for p in prompts for _ in range(num_runs)]
            responses = [r for r in responses for _ in range(num_runs)]

            # Calculate scorers - kSCST
            scorer_returns = scorer.score(prompts, generateds, responses=responses, step_count=step_count, bandit=bandit, chosen=chosen)
            # chosen = None 
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

            # Update step_count when one batch is finished
            step_count += 1

            # Validation model with val dataset and show the mean rewards
            current_scores = { k["name"]+"_scores":[] for k in scorer.scorers }
            if step_count !=0 and  step_count % val_interval_size == 0:
                
                # Generate outputs with given prompts from val dataset
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
                    prompts = [p for p in prompts for _ in range(num_runs)]
                    responses = [r for r in responses for _ in range(num_runs)]
                    
                    # Calculate scores
                    scorer_returns = scorer.rl_score(prompts, generateds, responses=responses, step_count=step_count, bandit=None)
                    for k,v in scorer_returns.items():
                        if k in current_scores:
                            current_scores[k].extend(v)
                print("Mean Rewards", [ np.mean(v) for k,v in current_scores.items() ])
                scaled = []
                for k,v in rl_scorer_history.items():
                    HISTORY_SIZE = val_history_size
                    history = v[-HISTORY_SIZE:]
                    if history == []:
                        scaled.append(0.0)
                    else:
                        scaled.append(np.mean(current_scores[k])-np.mean(history))
                print("Mean Scaled Rewards", scaled)

                # Return scores back to update bandit weights
                bandit(np.mean(scaled), last_chosen) 
                bandit_arm_weight_history.append(bandit.weights.copy())
                weights = scorer.weight_bandit.weights
                weights = weights / np.sum(weights) 
                bandit_weight_history.append(weights.tolist())
                chosen = bandit.draw()
                last_chosen = chosen
                bandit_pulls[last_chosen] += 1
                bandit_history.append(last_chosen)
                print(f"Step {step_count} / Chosen arm: {chosen}")
                print("Bandit Pull:", bandit_pulls)
                print("Bandit weights:", bandit.weights)
                for k,v in current_scores.items():
                    rl_scorer_history[k].extend(v)

                # Record mean reward and check if converged
                mean_reward = np.mean([ np.mean(v) for k,v in current_scores.items() ])
                rewards_history.append(mean_reward)
                wandb.log({"mean_reward": mean_reward, "data_consuming": step_count*train_batch_size})
                if slope_convergence(rewards_history):
                    print("Training converged!")
                    print(f"Data consuming:{step_count*train_batch_size}")
            
    # Save experiment model
    os.makedirs(experiment_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_model_dynaopt_{timestamp}.pt"
    save_path = os.path.join(experiment_path, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Saved experiment model to {save_path}")

    return model

def fixed_weights(
    seed: int = 429023,
    model_path: str = "models/google-t5-t5-base",
    data_path: str = "data/pair_data/pair_data.csv",
    experiment_path: str = "experiment_results/",
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    val_interval_size: int = 8,
    val_history_size: int = 10,
    num_runs: int = 10, 
    num_steps: int = 5000,
    do_wandb: int = 0,
    weights_dict: dict = {"reflection": 0.5, "fluency":0.2, "coherence":0.3}
    ):
    """
    Implements dynamic optimization for RL reward modeling, based on the DynaOpt framework.

    Parameters:
        - seed (int): Random seed for reproducibility.
        - model_path (str): Path to the pre-trained T5 model.
        - data_path (str): Path to the training data CSV file.
        - max_seq_length (int): Maximum input sequence length.
        - max_output_length (int): Maximum output sequence length.
        - train_batch_size (int): Batch size for training.
        - val_batch_size (int): Batch size for validation.
        - val_interval_size (int): Number of steps between validation checks.
        - num_runs (int): Number of training runs to perform.
        - num_steps (int): Number of training steps per run.
        - do_wandb (bool): Whether to use Weights & Biases for logging.

    This is a simplified version of the original DynaOpt implementation 
    (https://github.com/MichiganNLP/dynaopt/blob/main/rl_train.py), 
    adapted to maintain consistency with DL_TRAINING settings.
    """

    # Load seed, device
    set_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Initialize wandb
    if do_wandb:
        wandb.init(project="DynaDRL", group="DL_EXPERIMENT", name=f"fixed_{timestamp}")
        wandb.define_metric("mean_reward", step_metric="data_consuming")
    else:
        wandb.init(project="DynaDRL", mode="disabled")

    # Load data
    train_data, val_data, test_data = get_data(data_path)
    train_dataloader = DataLoader(dataset=train_data, batch_size=train_batch_size,\
        sampler=RandomSampler(train_data), drop_last=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_data, batch_size=val_batch_size,\
        sampler=SequentialSampler(val_data), drop_last=True, collate_fn=collate_fn)

    # Load models
    model, tokenizer = get_model(
        model_path,  
        max_seq_length=128,
        max_output_length=64,
        lora=False
    )
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

    # Define generation parameters
    gen_params = {
        "max_new_tokens": 64,
        "early_stopping": True,
        "do_sample": True,
        "num_return_sequences": num_runs,
        "temperature": 1.0
    }

    # Create criterion and scorer 
    rl_crit = ReinforceCriterion(model, tokenizer, optimizer, scaler, ref_model=ref_model, kl_coeff=0.05)
    scorer_model = ReflectionScoreDeployedCL(score_change=False, model_file= "./weights/reflection_scorer_weight.pt")
    scorer_model.type = "CLM"
    scorers = [{"name": "reflection", "model": scorer_model, "sign": 1, "weight": 1.0, "train": True},
                {"name": "fluency", "model": Multi(type="fluency"), "sign": 1, "weight": 1.0, "train": True},
                {"name": "coherence", "model": Multi(type="coherence"), "sign": 1, "weight": 1.0, "train": True}]
    scorer = ScorerWrapper(scorers, learning_mode = "bandit_weighted", scoring_method="fixed_logsum", max_batch_size=12)

    # Return scores back to update bandit weights
    bandit = Exp3(len(scorers)+1)
    bandit_history = []
    bandit_weight_history = []
    bandit_arm_weight_history = []
    chosen = bandit.draw()
    last_chosen = chosen
    print("Bandit arm pulled:", chosen)
    rl_scorer_history = { k["name"]+"_scores":[] for k in scorer.scorers }
    bandit_pulls = { i:0 for i in range(len(scorer.scorers)+1) } 
    bandit_pulls[last_chosen] += 1
    bandit_history.append(last_chosen)
    bandit_arm_weight_history.append(bandit.weights.copy())
    print("Bandit Pull:", bandit_pulls)
    print("Bandit Weights:", bandit.weights)

    # Training loop
    step_count = 0
    rewards_history = []
    while step_count < num_steps:
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
            prompts = [p for p in prompts for _ in range(num_runs)]
            responses = [r for r in responses for _ in range(num_runs)]

            # Calculate scorers - kSCST
            scorer_returns = scorer.score(prompts, generateds, responses=responses, step_count=step_count, bandit=bandit, chosen=chosen, extra=weights_dict)
            # chosen = None 
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

            # Update step_count when one batch is finished
            step_count += 1

            # Validation model with val dataset and show the mean rewards
            current_scores = { k["name"]+"_scores":[] for k in scorer.scorers }
            if step_count !=0 and  step_count % val_interval_size == 0:
                
                # Generate outputs with given prompts from val dataset
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
                    prompts = [p for p in prompts for _ in range(num_runs)]
                    responses = [r for r in responses for _ in range(num_runs)]
                    
                    # Calculate scores
                    scorer_returns = scorer.rl_score(prompts, generateds, responses=responses, step_count=step_count, bandit=None)
                    for k,v in scorer_returns.items():
                        if k in current_scores:
                            current_scores[k].extend(v)
                print("Mean Rewards", [ np.mean(v) for k,v in current_scores.items() ])
                scaled = []
                for k,v in rl_scorer_history.items():
                    HISTORY_SIZE = val_history_size
                    history = v[-HISTORY_SIZE:]
                    if history == []:
                        scaled.append(0.0)
                    else:
                        scaled.append(np.mean(current_scores[k])-np.mean(history))
                print("Mean Scaled Rewards", scaled)

                # Return scores back to update bandit weights
                bandit(np.mean(scaled), last_chosen) 
                bandit_arm_weight_history.append(bandit.weights.copy())
                weights = scorer.weight_bandit.weights
                weights = weights / np.sum(weights) 
                bandit_weight_history.append(weights.tolist())
                chosen = bandit.draw()
                last_chosen = chosen
                bandit_pulls[last_chosen] += 1
                bandit_history.append(last_chosen)
                print(f"Step {step_count} / Chosen arm: {chosen}")
                print("Bandit Pull:", bandit_pulls)
                print("Bandit weights:", bandit.weights)
                for k,v in current_scores.items():
                    rl_scorer_history[k].extend(v)

                # Record mean reward and check if converged
                mean_reward = np.mean([ np.mean(v) for k,v in current_scores.items() ])
                rewards_history.append(mean_reward)
                wandb.log({"mean_reward": mean_reward, "data_consuming": step_count*train_batch_size})
                if slope_convergence(rewards_history):
                    print("Training converged!")
                    print(f"Data consuming:{step_count*train_batch_size}")
            
    # Save experiment model
    os.makedirs(experiment_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_model_dynaopt_{timestamp}.pt"
    save_path = os.path.join(experiment_path, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Saved experiment model to {save_path}")

    return model

def verify_lora(
    seed: int = 429023,
    model_path: str = "models/google-t5-t5-base",
    data_path: str = "data/pair_data/pair_data.csv",
    lora_path: str = "lora_results/results-1k-0115/",
    test_batch_size: int = 8,
    test_history_size: int = 10,
    num_runs: int = 10, 
    num_steps: int = 500,
    objective: str = "reflection"
    ):

    # Load seed, device and wandb
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Load data
    train_data, val_data, test_data = get_data(data_path)
    test_dataloader = DataLoader(dataset=test_data, batch_size=test_batch_size,\
        sampler=RandomSampler(test_data), drop_last=True, collate_fn=collate_fn)

    # Load original model 
    model, tokenizer = get_model(
        model_path,  
        max_seq_length=128,
        max_output_length=64,
        lora=True
    )
    original_params = model.state_dict()
    model.to(device)
    model.eval()

    # Load lora params to update original model
    # lora_params = load_lora(lora_path+"lora_"+objective+".npz")
    # lora_keys = lora_params.keys()
    # for key in lora_keys:
    #     start_idx = len('base_model.model.')
    #     k = key[start_idx:] + '.weight'
    #     original_params[k] = original_params[k] + (lora_params[key][1] @ lora_params[key][0]) * lora_params[key][2]
    # updated_params = original_params
    # model.load_state_dict(updated_params)

    # Hard code test weights effect
    updated_params = original_params
    lora_params = load_lora(lora_path+"lora_reflection.npz")
    lora_keys = lora_params.keys()
    for key in lora_keys:
        start_idx = len('base_model.model.')
        k = key[start_idx:] + '.weight'
        updated_params[k] = updated_params[k] + (lora_params[key][1] @ lora_params[key][0]) * lora_params[key][2]

    lora_params = load_lora(lora_path+"lora_fluency.npz")
    lora_keys = lora_params.keys()
    for key in lora_keys:
        start_idx = len('base_model.model.')
        k = key[start_idx:] + '.weight'
        updated_params[k] = updated_params[k] + (lora_params[key][1] @ lora_params[key][0]) * lora_params[key][2]
    model.load_state_dict(updated_params)

    # Define generation parameters
    gen_params = {
        "max_new_tokens": 64,
        "early_stopping": True,
        "do_sample": True,
        "num_return_sequences": num_runs,
        "temperature": 1.0
    }

    # Define scorer
    scorer_model = ReflectionScoreDeployedCL(score_change=False, model_file= "./weights/reflection_scorer_weight.pt")
    scorer_model.type = "CLM"
    scorers = [{"name": "reflection", "model": scorer_model, "sign": 1, "weight": 1.0, "train": True},
                {"name": "fluency", "model": Multi(type="fluency"), "sign": 1, "weight": 1.0, "train": True},
                {"name": "coherence", "model": Multi(type="coherence"), "sign": 1, "weight": 1.0, "train": True}]
    scorer = ScorerWrapper(scorers, learning_mode = "bandit_weighted", \
            scoring_method="logsum", max_batch_size=12)

    # Dynamic weighting loop
    step_count = 0
    rewards_history = []
    while step_count < num_steps:

        # Dynamic weighting on test dataset
        current_scores = { k["name"]+"_scores":[] for k in scorer.scorers }
        for batch in test_dataloader:

            # Generate outputs with given prompts
            responses = batch["responses"]
            prompts = batch["prompts"]
            gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, \
                return_tensors="pt", padding="longest", truncation=True)
            gen_input = {k: v.to(device) for k, v in gen_input.items()}
            gens_out = model.generate(input_ids=gen_input["input_ids"], \
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
            prompts = [p for p in prompts for _ in range(num_runs)]
            responses = [r for r in responses for _ in range(num_runs)]

            # Calculate scores
            scorer_returns = scorer.rl_score(prompts, generateds, responses=responses, step_count=step_count, bandit=None)
            for k,v in scorer_returns.items():
                if k in current_scores:
                    current_scores[k].extend(v)
        print("Mean Rewards", [ np.mean(v) for k,v in current_scores.items() ])

        # Update step_count when one dataloader is finished
        step_count += 1

    return model

def grid_search_lora(
    seed: int = 429023,
    model_path: str = "models/google-t5-t5-base",
    data_path: str = "data/pair_data/pair_data.csv",
    lora_path: str = "lora_results/results-1k-0115/",
    test_batch_size: int = 8,
    num_runs: int = 10, 
    # num_steps: int = 500,
    objectives: list = ["reflection", "fluency"]
    ):

    # Load seed, device and wandb
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Load data
    train_data, val_data, test_data = get_data(data_path)
    test_dataloader = DataLoader(dataset=test_data, batch_size=test_batch_size,\
        sampler=RandomSampler(test_data), drop_last=True, collate_fn=collate_fn)

    # Load original model 
    model, tokenizer = get_model(
        model_path,  
        max_seq_length=128,
        max_output_length=64,
        lora=True
    )
    original_params = model.state_dict()
    model.to(device)
    model.eval()

    # Define generation parameters
    gen_params = {
        "max_new_tokens": 64,
        "early_stopping": True,
        "do_sample": True,
        "num_return_sequences": num_runs,
        "temperature": 1.0
    }

    # Define scorer
    scorer_model = ReflectionScoreDeployedCL(score_change=False, model_file= "./weights/reflection_scorer_weight.pt")
    scorer_model.type = "CLM"
    scorers = [{"name": "reflection", "model": scorer_model, "sign": 1, "weight": 1.0, "train": True},
                {"name": "fluency", "model": Multi(type="fluency"), "sign": 1, "weight": 1.0, "train": True}]
                # {"name": "coherence", "model": Multi(type="coherence"), "sign": 1, "weight": 1.0, "train": True}]
    scorer = ScorerWrapper(scorers, learning_mode = "bandit_weighted", \
            scoring_method="logsum", max_batch_size=12)

    # Dynamic weighting loop
    step_count = 0
    lora_reflection_params = load_lora(lora_path+"lora_reflection.npz")
    lora_reflection_keys = lora_reflection_params.keys()
    lora_fluency_params = load_lora(lora_path+"lora_fluency.npz")
    lora_fluency_keys = lora_fluency_params.keys()
    for w1 in np.linspace(1.0, 0.0, 11):
        for w2 in np.linspace(0.0, 1.0, 11):

            print(f"LORA Weights:{w1, w2}")
            # Hard code test weights effect
            updated_params = copy.deepcopy(original_params)
            for key in lora_reflection_keys:
                start_idx = len('base_model.model.')
                k = key[start_idx:] + '.weight'
                updated_params[k] = updated_params[k] + np.dot(lora_reflection_params[key][1], lora_reflection_params[key][0]) * lora_reflection_params[key][2] * w1
            for key in lora_fluency_keys:
                start_idx = len('base_model.model.')
                k = key[start_idx:] + '.weight'
                updated_params[k] = updated_params[k] + np.dot(lora_fluency_params[key][1], lora_fluency_params[key][0]) * lora_fluency_params[key][2] * w2
            model.load_state_dict(updated_params)

            # Dynamic weighting on test dataset
            current_scores = { k["name"]+"_scores":[] for k in scorer.scorers }
            for batch in test_dataloader:

                # Generate outputs with given prompts
                responses = batch["responses"]
                prompts = batch["prompts"]
                gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, \
                    return_tensors="pt", padding="longest", truncation=True)
                gen_input = {k: v.to(device) for k, v in gen_input.items()}
                gens_out = model.generate(input_ids=gen_input["input_ids"], \
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
                prompts = [p for p in prompts for _ in range(num_runs)]
                responses = [r for r in responses for _ in range(num_runs)]

                # Calculate scores
                scorer_returns = scorer.rl_score(prompts, generateds, responses=responses, step_count=step_count, bandit=None)
                for k,v in scorer_returns.items():
                    if k in current_scores:
                        current_scores[k].extend(v)
            print("Mean Rewards", [ np.mean(v) for k,v in current_scores.items() ])

            step_count += 1

    return model

if __name__ == "__main__":
    # model = dynaopt(train_batch_size = 8, val_batch_size = 8, val_interval_size = 8, num_runs = 10, num_steps = 5000, do_wandb = 1)

    # model = fixed_weights(train_batch_size = 8, val_batch_size = 8, val_interval_size = 8, num_runs = 10, num_steps = 5000, do_wandb = 1, \
    #     weights_dict = {"reflection": 0.5, "fluency":0.2, "coherence":0.3})
    
    # model = verify_lora(objective="coherence")

    model = grid_search_lora()