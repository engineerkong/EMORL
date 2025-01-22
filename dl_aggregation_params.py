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

def grid_search_aggregation_weights(args, original_model, original_params, tokenizer, dataloader, scorer, gen_params, max_output_length=64, grid_size=11, test_batch_size=1, num_runs=1, device="cuda"):
    """
    Perform grid search to find the best weights for combining three models' hidden layers.
    
    Args:
        model_list (list): List of three T5 models.
        tokenizer (Tokenizer): Tokenizer for the models.
        prompts (list): List of input prompts for evaluation.
        rewards_fn (callable): A function to calculate rewards given outputs and references.
        max_output_length (int): Maximum length of the generated output.
        grid_size (int): Number of grid search steps per weight dimension.
        device (str): Device to run the models on ("cuda" or "cpu").
    
    Returns:
        dict: Dictionary with the best weights and corresponding mean reward.
    """
    # Define weight ranges for grid search
    weight_range = np.linspace(0, 1, grid_size)
    best_weights = None
    best_mean_reward = -float("inf")

    # Loop through all weight combinations
    for w1 in weight_range:
        for w2 in weight_range:
            for w3 in weight_range:
            # w3 = 1.0 - (w1 + w2)
            # if w3 < 0 or w3 > 1.0:
            #     continue  # Skip invalid weight combinations

                weights = [w1, w2, w3]
                rewards = []

                # Load lora params for all objectives
                updated_params = copy.deepcopy(original_params)
                new_model = copy.deepcopy(original_model)
                new_model.to(device)
                for i in range(len(args.objectives)):
                    lora_params = load_lora(args.lora_path+"lora_"+args.objectives[i]+".npz")
                    for key in lora_params.keys():
                        start_idx = len('base_model.model.')
                        k = key[start_idx:] + '.weight'
                        updated_params[k] = updated_params[k] + (lora_params[key][1] @ lora_params[key][0]) * lora_params[key][2] * weights[i]
                new_model.load_state_dict(updated_params)

                current_scores = { k["name"]+"_scores":[] for k in scorer.scorers }
                for batch in dataloader:
                    
                    # Generate outputs with given prompts
                    responses = batch["responses"]
                    prompts = batch["prompts"]
                    # Encode prompts for input
                    gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, return_tensors="pt", padding="longest", truncation=True)
                    gen_input = {k: v.to(device) for k, v in gen_input.items()}

                    # Generate outputs with given prompts
                    responses = batch["responses"]
                    prompts = batch["prompts"]
                    gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, \
                        return_tensors="pt", padding="longest", truncation=True)
                    gen_input = {k: v.to(device) for k, v in gen_input.items()}
                    gens_out = new_model.generate(input_ids=gen_input["input_ids"], \
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
                    scorer_returns = scorer.rl_score(prompts, generateds, responses=responses, step_count=None, bandit=None)
                    for k,v in scorer_returns.items():
                        if k in current_scores:
                            current_scores[k].extend(v)

                # Calculate mean reward for this weight combination    
                rewards = [ np.mean(v) for k,v in current_scores.items() ]
                mean_reward = np.mean(rewards)
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    best_weights = weights
                print(f"Weights:{weights}")
                print(f"Rewards:{rewards}")

    return {
        "best_weights": best_weights,
        "best_mean_reward": best_mean_reward,
    }

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
        - do_wandb (bool): Whether to use Weights & Biases for logging.

    Returns:
        torch.nn.Module: The model with aggregated parameters from multiple LoRA checkpoints

    Example:
        args = parser.parse_args()
        model = aggregation(args)
    """

    # Load seed, device and wandb
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Initialize wandb
    if args.do_wandb:
        wandb.init(project="DynaDRL", group="DL_AGGREGATION", name=f"aggregation_{timestamp}")
        wandb.define_metric("mean_reward", step_metric="data_consuming")
    else:
        wandb.init(project="DynaDRL", mode="disabled")

    # Load data
    train_data, val_data, test_data = get_data(args.data_path)
    test_dataloader = DataLoader(dataset=test_data[:4], batch_size=args.test_batch_size,\
        sampler=RandomSampler(test_data[:4]), drop_last=True, collate_fn=collate_fn)

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


    
    # Perform grid search
    results = grid_search_aggregation_weights(
        args,
        original_model=model,
        original_params=original_params,
        tokenizer=tokenizer,
        dataloader=test_dataloader,
        scorer=scorer,
        gen_params=gen_params,
        grid_size=11,  # e.g., weights range from 0.0 to 1.0 in 0.1 steps
        test_batch_size=args.test_batch_size,
        num_runs=args.num_runs,
        device=device
    )

    # Print results
    print("Best Weights:", results["best_weights"])
    print("Best Mean Reward:", results["best_mean_reward"])

    # Save aggregated model
    os.makedirs(args.output_path, exist_ok=True)
    objectives_str = '_'.join(args.objectives)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"aggregated_model_{objectives_str}_{timestamp}.pt"
    save_path = os.path.join(args.output_path, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Saved aggregated model to {save_path}")

    return model


    # weights = [0.9, 0.1, 0.0]
    # step_count = 0
    # rewards_history = []
    # while step_count < args.num_steps:
    # # Load lora params for all objectives
    #     updated_params = copy.deepcopy(original_params)
    #     for i in range(len(args.objectives)):
    #         lora_params = load_lora(args.lora_path+"lora_"+args.objectives[i]+".npz")
    #         for key in lora_params.keys():
    #             start_idx = len('base_model.model.')
    #             k = key[start_idx:] + '.weight'
    #             updated_params[k] = updated_params[k] + (lora_params[key][1] @ lora_params[key][0]) * lora_params[key][2] * weights[i]
    #     model.load_state_dict(updated_params)

    #     # Dynamic weighting on test dataset
    #     current_scores = { k["name"]+"_scores":[] for k in scorer.scorers }
    #     for batch in test_dataloader:

    #         # Generate outputs with given prompts
    #         responses = batch["responses"]
    #         prompts = batch["prompts"]
    #         gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, \
    #             return_tensors="pt", padding="longest", truncation=True)
    #         gen_input = {k: v.to(device) for k, v in gen_input.items()}
    #         gens_out = model.generate(input_ids=gen_input["input_ids"], \
    #             attention_mask=gen_input["attention_mask"], **gen_params)
    #         generateds = tokenizer.batch_decode(gens_out, skip_special_tokens=True)
    #         generateds = [ g.split("[CLS]") for g in generateds]
    #         new_generateds = []
    #         for g_list in generateds:
    #             if len(g_list) <= 1:
    #                 new_generateds.append([g_list[0].strip()])
    #             else:
    #                 new_generateds.append([x.strip() for x in g_list[:-1]])
    #         generateds = new_generateds
    #         cls_generateds = [ [ x.strip() + " [CLS]" for x in g] for g in generateds ]
    #         cls_generateds = [ " ".join(g) for g in cls_generateds]
    #         generateds = [ " ".join(g) for g in generateds]
    #         generateds = [ g.replace("<pad>", "").strip() for g in generateds]
    #         generateds = [g.replace("[CLS]", "").strip() for g in generateds]
    #         prompts = [p for p in prompts for _ in range(args.num_runs)]
    #         responses = [r for r in responses for _ in range(args.num_runs)]

    #         # Calculate scores
    #         scorer_returns = scorer.rl_score(prompts, generateds, responses=responses, step_count=step_count, bandit=None)
    #         for k,v in scorer_returns.items():
    #             if k in current_scores:
    #                 current_scores[k].extend(v)
    #     print("Mean Rewards", [ np.mean(v) for k,v in current_scores.items() ])
        # scaled = []
        # for k,v in rl_scorer_history.items():
        #     HISTORY_SIZE = args.test_history_size
        #     history = v[-HISTORY_SIZE:]
        #     if history == []:
        #         scaled.append(0.0)
        #     else:
        #         scaled.append(np.mean(current_scores[k])-np.mean(history))
        # print("Mean Scaled Rewards", scaled) # all 3 objectives

        # # Update step_count when one dataloader is finished
        # step_count += 1

        # # Return scores back to update bandit weights
        # mean_reward = [ np.mean(v) for k,v in current_scores.items() ]
        # bandit(np.mean(scaled)*100, last_chosen) # the object is set to np.mean(scaled)
        # bandit_arm_weight_history.append(bandit.weights.copy())
        # weights = bandit.weights[:len(args.objectives)] / np.sum(bandit.weights[:len(args.objectives)])
        # chosen = bandit.draw()
        # last_chosen = chosen
        # bandit_pulls[last_chosen] += 1
        # bandit_history.append(last_chosen)
        # print(f"Step {step_count} / Chosen arm: {chosen}")
        # print("Bandit Pull:", bandit_pulls)
        # print("Bandit weights:", bandit.weights)
        # for k,v in current_scores.items():
        #     rl_scorer_history[k].extend(v)
        # print(f"Scaled Bandit Weights:{weights}")
        # assert abs(weights.sum() - 1) < 1e-5

        # # Record mean reward and check if converged
        # rewards_history.append(mean_reward)
        # wandb.log({"mean_reward": mean_reward, "data_consuming": step_count*args.test_batch_size})
        # if slope_convergence(rewards_history):
        #     print("Training converged!")
        #     print(f"Data consuming:{step_count*args.test_batch_size}")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=429023)
    parser.add_argument('--model_path', type=str, default="models/t5-base")
    parser.add_argument('--data_path', type=str, default="data/PAIR/pair_data.csv")
    parser.add_argument('--lora_path', type=str, default="lora_results/20250115/")
    parser.add_argument('--output_path', type=str, default="aggregation_results/")
    parser.add_argument('--objectives', nargs='+', default=["reflection", "fluency", "coherence"])
    parser.add_argument('--test_batch_size', type=int, default=8)    
    parser.add_argument('--test_history_size', type=int, default=10)
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--do_wandb', type=int, default=0)
    
    args = parser.parse_args()
    # save_args(args, "DL_AGGREGATION", "logs/")
    
    # Run aggregation
    model = aggregation(args)
    
    return model

if __name__ == "__main__":
    main()
            