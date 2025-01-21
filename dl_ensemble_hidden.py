import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
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

def ensemble(args):
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
        wandb.init(project="DynaDRL", group="DL_ENSEMBLE", name=f"ensemble_{timestamp}")
        wandb.define_metric("mean_reward", step_metric="data_consuming")
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
                {"name": "fluency", "model": Multi(type="fluency"), "sign": 1, "weight": 1.0, "train": True}]
                # {"name": "coherence", "model": Multi(type="coherence"), "sign": 1, "weight": 1.0, "train": True}]
    scorer = ScorerWrapper(scorers, learning_mode = "bandit_weighted", \
            scoring_method="logsum", max_batch_size=12)


    # # Initialize bandit
    # print("Initilize Bandit")
    # bandit = Exp3(len(scorers)+1, gamma=0.07)
    # bandit_history = []
    # # bandit_weight_history = [] # reward is always 1, scorer["weight"] = self.weight_bandit.weights[i]
    # bandit_arm_weight_history = [] # reward from scaled scorer return
    # weights = bandit.weights[:len(args.objectives)] / np.sum(bandit.weights[:len(args.objectives)])
    # chosen = bandit.draw() # select by bandit according to distr
    # last_chosen = chosen
    # print("Bandit arm pulled:", chosen)
    # rl_scorer_history = { k["name"]+"_scores":[] for k in scorer.scorers }
    # bandit_pulls = { i:0 for i in range(len(scorer.scorers)+1) } 
    # bandit_pulls[last_chosen] += 1
    # bandit_history.append(last_chosen)
    # bandit_arm_weight_history.append(bandit.weights.copy())
    # print("Bandit Pull:", bandit_pulls)
    # print("Bandit Weights:", bandit.weights)
    # print(f"Scaled Bandit Weights:{weights}")
    # assert abs(weights.sum() - 1) < 1e-5
    weights = [0.5, 0.5]

    step_count = 0
    rewards_history = []
    decoder_input_ids = torch.tensor([[model1.config.decoder_start_token_id]])  # <pad> 或 <s> 起始标记
    while step_count < args.num_steps:
    # Load lora params for all objectives
        
        # Dynamic weighting on test dataset
        current_scores = { k["name"]+"_scores":[] for k in scorer.scorers }
        # Dynamic weighting on test dataset
        for batch in test_dataloader:

            all_hidden_states = 0
            model_list = []
            for i in range(len(args.objectives)):
                updated_params = copy.deepcopy(original_params)
                lora_params = load_lora(args.lora_path+"lora_"+args.objectives[i]+".npz")
                for key in lora_params.keys():
                    start_idx = len('base_model.model.')
                    k = key[start_idx:] + '.weight'
                    updated_params[k] = updated_params[k] + (lora_params[key][1] @ lora_params[key][0]) * lora_params[key][2]
                model_list.append(model.load_state_dict(updated_params))

            # Generate outputs with given prompts
            responses = batch["responses"]
            prompts = batch["prompts"]
            gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, \
                return_tensors="pt", padding="longest", truncation=True)
            gen_input = {k: v.to(device) for k, v in gen_input.items()}
            gens_out_0 = model_list[0].generate(input_ids=gen_input["input_ids"], \
                attention_mask=gen_input["attention_mask"], \
                decoder_input_ids=decoder_input_ids, \
                output_hidden_states=True, \
                return_dict=True, \
                **gen_params)
            all_hidden_states += gens_out_0.decoder_hidden_states[-1] * 0.5
            gens_out_1 = model_list[1].generate(input_ids=gen_input["input_ids"], \
                attention_mask=gen_input["attention_mask"], \
                decoder_input_ids=decoder_input_ids, \
                output_hidden_states=True, \
                return_dict=True, \
                **gen_params)
            all_hidden_states += gens_out_1.decoder_hidden_states[-1] * 0.5
            logits = model_list[0].lm_head(all_hidden_states)
      
            # 融合logits并生成文本
            merged_scores = []
            for step in range(len(all_outputs[0].scores)):
                step_scores = []
                for model_idx, outputs in enumerate(all_outputs):
                    logits = outputs.scores[step]
                    probs = F.softmax(logits, dim=-1)
                    weighted_probs = probs * weights[model_idx]
                    step_scores.append(weighted_probs)
                
                merged_prob = sum(step_scores)
                merged_logit = torch.log(merged_prob + 1e-10)
                merged_scores.append(merged_logit)

            # 使用merged_scores生成文本
            generated_tokens = [[] for i in range(len(merged_scores[0]))]
            current_tokens = gen_input["input_ids"]

            for i in range(len(merged_scores[0])):
                for step_logits in merged_scores:
                    # 获取最可能的token
                    next_token = torch.argmax(step_logits, dim=-1)
                    g = next_token[i].item()    
                    generated_tokens[i].append(g)

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
            prompts = [p for p in prompts for _ in range(args.num_runs)]
            responses = [r for r in responses for _ in range(args.num_runs)]

            # Calculate scores
            scorer_returns = scorer.rl_score(prompts, generateds, responses=responses, step_count=step_count, bandit=None)
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
        mean_reward = [ np.mean(v) for k,v in current_scores.items() ]
        bandit(np.mean(scaled)*100, last_chosen) # the object is set to np.mean(scaled)
        bandit_arm_weight_history.append(bandit.weights.copy())
        weights = bandit.weights[:len(args.objectives)] / np.sum(bandit.weights[:len(args.objectives)])
        chosen = bandit.draw()
        last_chosen = chosen
        bandit_pulls[last_chosen] += 1
        bandit_history.append(last_chosen)
        print(f"Step {step_count} / Chosen arm: {chosen}")
        print("Bandit Pull:", bandit_pulls)
        print("Bandit weights:", bandit.weights)
        for k,v in current_scores.items():
            rl_scorer_history[k].extend(v)
        print(f"Scaled Bandit Weights:{weights}")
        assert abs(weights.sum() - 1) < 1e-5

        # Record mean reward and check if converged
        rewards_history.append(mean_reward)
        wandb.log({"mean_reward": mean_reward, "data_consuming": step_count*args.test_batch_size})
        if slope_convergence(rewards_history):
            print("Training converged!")
            print(f"Data consuming:{step_count*args.test_batch_size}")
    
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
    parser.add_argument('--model_path', type=str, default="models/t5-base")
    parser.add_argument('--data_path', type=str, default="data/PAIR/pair_data.csv")
    parser.add_argument('--lora_path', type=str, default="lora_results/20250115/")
    parser.add_argument('--output_path', type=str, default="aggregation_results/")
    parser.add_argument('--objectives', nargs='+', default=["reflection", "fluency"])
    parser.add_argument('--test_batch_size', type=int, default=8)    
    parser.add_argument('--test_history_size', type=int, default=10)
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--do_wandb', type=int, default=0)
    
    args = parser.parse_args()
    # save_args(args, "DL_AGGREGATION", "logs/")
    
    # Run merge
    model = merge(args)
    
    return model

if __name__ == "__main__":
    main()
            