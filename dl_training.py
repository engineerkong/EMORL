import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import copy
import random
import numpy as np
import wandb
import argparse
import time
import os
from model_empathy import *
from dynaopt_lib import *
from utils_lora import *
from utils_additional import *
from huggingface_hub import login

def config_training(args, objective, seed):
    # Load device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_data, val_data, test_data = get_data(args.data_path)
    train_dataloader = DataLoader(dataset=train_data, batch_size=args.train_batch_size,\
        sampler=RandomSampler(train_data), drop_last=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_data, batch_size=args.val_batch_size,\
        sampler=RandomSampler(val_data), drop_last=True, collate_fn=collate_fn)

    # Load model (lora) and tokenizer
    model, tokenizer = get_model(
        args.model_path,  
        max_seq_length=128,
        max_output_length=64,
        lora=True if args.training_mode ==  "distributed" else False
    )
    if args.training_mode ==  "distributed":
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
    learning_mode = "bandit_weighted" if args.training_mode == "centralized" else "single"
    objectives = [objective] if args.training_mode == "distributed" else objective
    scorers = [get_scorer(obj) for obj in objectives]
    rl_crit = ReinforceCriterion(model, tokenizer, optimizer, scaler, ref_model=ref_model, kl_coeff=0.05)
    scorer = ScorerWrapper(scorers, learning_mode, scoring_method="fixed_logsum" if args.training_mode=="fixed" else "logsum", max_batch_size=12)
   
    # Create validation criterion and scorer
    val_objectives = ["reflection", "empathy", "fluency", "coherence", "specificity"]
    val_scorers = [get_scorer(obj) for obj in val_objectives]
    val_scorer = ScorerWrapper(val_scorers, learning_mode, scoring_method="fixed_logsum" if args.training_mode=="fixed" else "logsum", max_batch_size=12)

    # Define generation parameters for DialoGPT
    gen_params = {
        "max_new_tokens": 64,
        "early_stopping": True,
        "do_sample": True,
        "num_return_sequences": args.num_runs,
        "temperature": 1.0,
        "pad_token_id": tokenizer.eos_token_id  # DialoGPT uses eos_token as pad_token
    }
    
    return {
        'device': device,
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'tokenizer': tokenizer,
        'model': model,
        'optimizer': optimizer,
        'scaler': scaler,
        'rl_crit': rl_crit,
        'scorer': scorer,
        'val_scorer': val_scorer,
        'gen_params': gen_params,
        'objective': objective,
        'seed': seed
    }

def train(args, device, train_dataloader, val_dataloader, tokenizer, model, optimizer, scaler, \
             rl_crit, scorer, val_scorer, gen_params, objective, seed):
    # Initialize wandb
    if args.do_wandb:
        wandb.init(project="DMORL-1", group="TRAINING", name=f"training_{args.training_mode}_{objective}_{seed}")
        wandb.define_metric("mean_reward", step_metric="data_consuming")
    else:
        wandb.init(project="DMORL", mode="disabled")

    # Initialize bandit
    if args.training_mode == "centralized":
        bandit = Exp3(len(objective)+1)
        # bandit_history = []
        # bandit_weight_history = []
        # bandit_arm_weight_history = []
        chosen = bandit.draw()
        last_chosen = chosen
        # print("Bandit arm pulled:", chosen)
        rl_scorer_history = { k["name"]+"_scores":[] for k in val_scorer.scorers }
        bandit_pulls = { i:0 for i in range(len(scorer.scorers)+1) } 
        bandit_pulls[last_chosen] += 1
        # bandit_history.append(last_chosen)
        # bandit_arm_weight_history.append(bandit.weights.copy())
        # print("Bandit Pull:", bandit_pulls)
        # print("Bandit Weights:", bandit.weights)

    # Training loop
    step_count = 0
    rewards_history = []
    data_consuming = 0
    T_train_start = time.time()
    training = True
    while step_count < args.num_steps and training:
        print(f"Step count:{step_count}")
        
        # Training on train dataset, batches 15
        for batch in train_dataloader:
            # Generate outputs with given prompts
            prompts = batch["prompts"]
            responses = batch["responses"]
            # DialoGPT uses a different tokenization approach
            gen_input = tokenizer(prompts, max_length=128, \
                return_tensors="pt", padding=True, truncation=True)
            gen_input = {k: v.to(device) for k, v in gen_input.items()}
            gens_out = model.generate(input_ids=gen_input["input_ids"],\
                attention_mask=gen_input["attention_mask"], **gen_params)
            # Process outputs for DialoGPT
            generateds = tokenizer.batch_decode(gens_out, skip_special_tokens=True)
            # Clean up the generated text
            generateds = [g.replace("<pad>", "").strip() for g in generateds]
            
            # For DialoGPT, we don't need to handle [CLS] tokens the same way as T5
            # Re-encode for the loss calculation
            gens_out = tokenizer(generateds, max_length=128, \
                return_tensors="pt", padding="longest", truncation=True)["input_ids"]
            prompts = [p for p in prompts for _ in range(args.num_runs)]
            responses = [r for r in responses for _ in range(args.num_runs)]

            # Calculate scorers - kSCST
            scorer_returns = scorer.score(prompts, generateds, responses=responses, step_count=step_count, \
                                          bandit=bandit if args.training_mode=="centralized" else None, \
                                          chosen=chosen if args.training_mode=="centralized" else None, \
                                          extras=args.weights_dict if args.training_mode=="fixed" else {})
            total_scores = torch.FloatTensor(scorer_returns["total_scores"]).cuda()
            batch_scores = total_scores.reshape(args.train_batch_size, args.num_runs)
            mean_scores = batch_scores.mean(dim=1)
            unlooped_mean_scores = torch.repeat_interleave(mean_scores, args.num_runs)
            normalized_rewards = (unlooped_mean_scores - total_scores)

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
            data_consuming += args.train_batch_size

            # Validation model with val dataset
            current_scores = { k["name"]+"_scores":[] for k in val_scorer.scorers }
            if step_count !=0 and  step_count % args.val_interval_size == 0:
                for batch in val_dataloader:
                    responses = batch["responses"]
                    prompts = batch["prompts"]
                    gen_input = tokenizer(prompts, max_length=128, \
                        return_tensors="pt", padding="longest", truncation=True)
                    gen_input = {k: v.to(device) for k, v in gen_input.items()}
                    gens_out = model.generate(input_ids=gen_input["input_ids"],\
                        attention_mask=gen_input["attention_mask"], **gen_params)
                    # Process validation outputs for DialoGPT
                    generateds = tokenizer.batch_decode(gens_out, skip_special_tokens=True)
                    # Clean up the generated text
                    generateds = [g.replace("<pad>", "").strip() for g in generateds]
                    prompts = [p for p in prompts for _ in range(args.num_runs)]
                    responses = [r for r in responses for _ in range(args.num_runs)]
                    scorer_returns = val_scorer.rl_score(prompts, generateds, responses=responses, step_count=step_count, \
                                                         extras=args.weights_dict if args.training_mode=="fixed" else {})
                    for k,v in scorer_returns.items():
                        if k in current_scores:
                            current_scores[k].extend(v)
                    # data_consuming += len(batch) # validation doesn't count

                # Record example generateds
                print(f"Generateds:{generateds[0]}")
                # Record mean reward and check if converged
                mean_reward = [ np.mean(v) for k,v in current_scores.items() ]
                print(f"Mean reward:{mean_reward}")

                # Update bandit
                if args.training_mode == "centralized":
                    scaled = []
                    for k,v in rl_scorer_history.items():
                        HISTORY_SIZE = args.val_interval_size
                        history = v[-HISTORY_SIZE:]
                        if history == []:
                            scaled.append(0.0)
                        else:
                            scaled.append(np.mean(current_scores[k])-np.mean(history))

                    bandit(np.mean(scaled), last_chosen) 
                    # bandit_arm_weight_history.append(bandit.weights.copy())
                    weights = scorer.weight_bandit.weights
                    weights = weights / np.sum(weights) 
                    # bandit_weight_history.append(weights.tolist())
                    chosen = bandit.draw()
                    last_chosen = chosen
                    bandit_pulls[last_chosen] += 1
                    # bandit_history.append(last_chosen)
                    # print(f"Step {step_count} / Chosen arm: {chosen}")
                    # print("Bandit Pull:", bandit_pulls)
                    # print("Bandit weights:", bandit.weights)
                    for k,v in current_scores.items():
                        rl_scorer_history[k].extend(v)
                        rl_scorer_history[k] = rl_scorer_history[k][-HISTORY_SIZE:]

                # Record using wandb and check convergence
                all_objectives = ["reflection", "empathy", "fluency", "coherence", "specificity"]
                if isinstance(objective, str):
                    indices = [i for i, x in enumerate(all_objectives) if x in objective]
                else:
                    indices = [all_objectives.index(x) for x in objective if x in all_objectives]
                mean_mr = np.mean([mean_reward[i] for i in indices])
                rewards_history.append(mean_mr)
                wandb.log({"mean_reward": mean_mr, "data_consuming": data_consuming})
                if check_convergence(rewards_history):
                    print("Training converged!")
                    print(f"Entire data consuming: {data_consuming}")
                    print(f"Entire time consuming: {time.time() - T_train_start}")
                    training = False

    # Finish wandb
    wandb.finish()
    if args.training_mode == "distributed":
        lora_params = acquire_lora_params(model)
        return lora_params
    else:
        all_params = model.state_dict()
        return all_params
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--training_mode', type=str, default="distributed", help="options: distributed/centralized/fixed")
    parser.add_argument('--model_path', type=str, default="microsoft/DialoGPT-medium")
    parser.add_argument('--data_path', type=str, default="data/PAIR/pair_data.csv")
    parser.add_argument('--output_path', type=str, default="lora_results/")
    parser.add_argument('--objectives', nargs='+', default=["reflection", "empathy", "fluency"])
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--val_interval_size', type=int, default=8)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--num_steps', type=int, default=16)
    parser.add_argument('--weights_dict', type=dict, default={"reflection": 1/3, "empathy": 1/3, "fluency": 1/3})
    parser.add_argument('--do_wandb', type=int, default=0)
    
    args = parser.parse_args()
    # save_args(args, "DL_TRAINING", "logs/")

    timestamp = time.strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(args.output_path, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    components = {
        'device': None,
        'train_dataloader': None,
        'val_dataloader': None,
        'tokenizer': None,
        'model': None,
        'optimizer': None,
        'scaler': None,
        'rl_crit': None,
        'scorer': None,
        'val_scorer': None,
        'gen_params': None,
        'objective': None,
        'seed': None
    }

    seeds = [random.randint(1, 100000) for _ in range(args.num_seeds)]
    for seed in seeds:
        set_seed(seed)

        if args.training_mode == "distributed":
            # Train on multiple objective seperately
            for objective in args.objectives:
                print(f"Start training for objective: {objective}")
                components.update(config_training(args, objective, seed))
                lora_params = train(args, **components)            
                # Save LORA parameters
                filename = f"lora_{objective}_{seed}.npz"
                npz_path = os.path.join(save_dir, filename)
                save_lora(lora_params, npz_path=npz_path)
                print(f"Saved LORA parameters to {npz_path}")
        elif args.training_mode == "centralized":
            components.update(config_training(args, args.objectives, seed))
            all_params = train(args, **components)
            # Save centralized model
            filename = f"centralized_{seed}.pt"
            save_path = os.path.join(save_dir, filename)
            torch.save(all_params, save_path)
            print(f"Saved centralized model to {save_path}")
        elif args.training_mode == "fixed":
            components.update(config_training(args, args.objectives, seed))
            all_params = train(args, **components)
            # Save fixed model
            filename = f"fixed_{seed}.pt"
            save_path = os.path.join(save_dir, filename)
            torch.save(all_params, save_path)
            print(f"Saved fixed model to {save_path}")
        else:
            raise("Training Mode Wrong!")

if __name__ == "__main__":
    main()
