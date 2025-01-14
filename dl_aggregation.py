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

def aggregation(args):
    """
    Aggregates multiple LoRA parameters trained on different objectives into a single model.
    This function loads the base T5 model and multiple LoRA checkpoints, then combines them
    using weighted aggregation.

    Args:
        - seed (int): Random seed for reproducibility
        - model_path (str): Path to the base T5 model
        - data_path (str): Path to the evaluation data CSV file
        - lora_path (str): Directory containing LoRA checkpoint files
        - output_path (str): Directory to save aggregation results
        - objectives (list): List of objectives corresponding to LoRA checkpoints
        - test_batch_size (int): Batch size for evaluation
        - test_history_size (int): Size of history to maintain for evaluation
        - num_runs (int): Number of evaluation runs
        - num_steps (int): Number of steps per evaluation run
        - do_wandb (bool)

    Returns:
        torch.nn.Module: The model with aggregated parameters from multiple LoRA checkpoints

    Example:
        args = parser.parse_args()
        model = aggregation(args)
    """

    # Load seed, device and wandb
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if do_wandb:
        wandb.init(project="DynaDRL", name=f"DL_TRAINING_{timestamp}")
    else:
        wandb.init(project="DynaDRL", mode="disabled")

    # Load data
    train_data, val_data, test_data = get_data(args.data_path)
    test_dataloader = DataLoader(dataset=test_data, batch_size=args.test_batch_size,\
        sampler=RandomSampler(test_data), drop_last=True, collate_fn=collate_fn)

    # Load original model 
    model, tokenizer = get_model(
        args.model_path,  
        max_seq_length=128,
        max_output_length=64,
        lora=True
    )
    original_params = model.state_dict()
    model.to(device)
    model.eval()

    # Load lora params for all objectives
    all_lora_params = []
    all_lora_keys = []
    for objective in args.objectives:
        lora_params = load_lora(args.lora_path+"lora_"+objective+".npz")
        for key in lora_params.keys():
            if key not in all_lora_keys:
                assert key.startswith('base_model.model.'), f"Key {key} does not start with 'base_model.model.'"
                start_idx = len('base_model.model.')
                suffix = key[start_idx:]
                all_lora_keys.append(suffix)
        all_lora_params.append(lora_params)

    # Define generation parameters
    gen_params = {
        "max_new_tokens": 64,
        "early_stopping": True,
        "do_sample": True,
        "num_return_sequences": args.num_runs,
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

    # Initialize bandit
    bandit = Exp3(len(scorers)+1)
    bandit_history = []
    # bandit_weight_history = [] # reward is always 1, scorer["weight"] = self.weight_bandit.weights[i]
    bandit_arm_weight_history = [] # reward from scorer return
    chosen = bandit.draw() # select by bandit according to distr
    last_chosen = chosen
    print("Bandit arm pulled:", chosen)
    rl_scorer_history = { k["name"]+"_scores":[] for k in scorer.scorers }
    bandit_pulls = { i:0 for i in range(len(scorer.scorers)+1) } 
    bandit_pulls[last_chosen] += 1
    bandit_history.append(last_chosen)
    bandit_arm_weight_history.append(bandit.weights.copy())
    print("Bandit Pull:", bandit_pulls)
    print("Bandit Weights:", bandit.weights)

    # Dynamic weighting loop
    step_count = 0
    while step_count < args.num_steps:

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
            prompts = [p for p in prompts for _ in range(args.num_runs)]
            responses = [r for r in responses for _ in range(args.num_runs)]

            # Calculate scores
            scorer_returns = scorer.rl_score(prompts, generateds, responses=responses, step_count=step_count, bandit=bandit)
            print(scorer_returns.keys())
            for k,v in scorer_returns.items():
                if k in current_scores:
                    current_scores[k].extend(v)
        print("Mean Rewards", [ np.mean(v) for k,v in current_scores.items() ])
        scaled = []
        for k,v in rl_scorer_history.items():
            HISTORY_SIZE = args.test_history_size
            history = v[-HISTORY_SIZE:]
            if history == []:
                scaled.append(0.0)
            else:
                scaled.append(np.mean(current_scores[k])-np.mean(history))
        print("Mean Scaled Rewards", scaled) # all 3 objectives

        # Update step_count when one dataloader is finished
        step_count += 1

        # Return scores back to update bandit weights
        bandit(np.mean(scaled), last_chosen) # the object is set to np.mean(scaled)
        bandit_arm_weight_history.append(bandit.weights.copy())
        weights = bandit.weights[:3] / np.sum(bandit.weights[:3])
        chosen = bandit.draw()
        last_chosen = chosen
        bandit_pulls[last_chosen] += 1
        bandit_history.append(last_chosen)
        print(f"Step {step_count} / Chosen arm: {chosen}")
        print("Bandit Pull:", bandit_pulls)
        print("Bandit weights:", bandit.weights)
        for k,v in current_scores.items():
            rl_scorer_history[k].extend(v)
        print(f"LORA Weights:{weights}")
        assert abs(weights.sum() - 1) < 1e-5

        # Update models using lora and bandit weights
        for k in all_lora_keys:
            adapted_weights = adjust_weights(weights, k, all_lora_params)
            for i, (w, lora) in enumerate(zip(adapted_weights, all_lora_params)):
                if w > 0:
                    original_params[k] = original_params[k] + (lora[k][1] @ lora[k][0]) * lora[k][2] * w
        updated_params = original_params
        model.load_state_dict(updated_params)
    
    # Save aggregated model
    os.makedirs(args.output_path, exist_ok=True)
    objectives_str = '_'.join(args.objectives)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"aggregated_model_{objectives_str}_{timestamp}.pt"
    save_path = os.path.join(args.output_path, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Saved aggregated model to {save_path}")

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=429023)
    parser.add_argument('--model_path', type=str, default="models/google-t5-t5-base")
    parser.add_argument('--data_path', type=str, default="data/pair_data/pair_data.csv")
    parser.add_argument('--lora_path', type=str, default="lora_results/")
    parser.add_argument('--output_path', type=str, default="aggregation_results/")
    parser.add_argument('--objectives', nargs='+', default=["reflection", "coherence", "fluency"])
    parser.add_argument('--test_batch_size', type=int, default=10)    
    parser.add_argument('--test_history_size', type=int, default=10)
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--do_wandb', type=bool, default=False)
    
    args = parser.parse_args()
    save_args(args, "DL_AGGREGATION", "logs/")
    
    # Run aggregation
    model = aggregation(args)
    
    return model

if __name__ == "__main__":
    main()
            