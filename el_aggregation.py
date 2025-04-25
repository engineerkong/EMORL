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
import glob

from model_empathy import *
from dynaopt_lib import *
from utils_lora import *
from utils_additional import *

import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def config_aggregation(args):
    # Load device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data, use validation data for training
    train_data, val_data, test_data = get_data(args.data_path)
    val_dataloader = DataLoader(dataset=val_data, batch_size=args.val_batch_size,\
        sampler=RandomSampler(val_data), drop_last=True, collate_fn=collate_fn)
    # Load original model, tokenizer and original_params
    model, tokenizer = get_model(
        args.model_path,  
        max_seq_length=128,
        max_output_length=64,
        lora=True
    )
    model.to(device)
    model.eval()
    original_params = model.state_dict()

    # Gather models individual objectives
    model_list = []
    if args.aggregation_mode == "states" or args.aggregation_mode == "logits":
        for i in range(len(args.objectives)):
            new_model = copy.deepcopy(model)
            new_model.to(device)
            updated_params = copy.deepcopy(original_params)
            pattern = os.path.join(args.lora_path, f"lora_{args.objectives[i]}*.npz")
            matching_files = glob.glob(pattern)
            lora_params = load_lora(matching_files[0])
            # lora_params = load_lora(args.lora_path+"lora_"+args.objectives[i]+".npz")
            for key in lora_params.keys():
                start_idx = len('base_model.model.')
                k = key[start_idx:] + '.weight'
                updated_params[k] = updated_params[k] + (lora_params[key][1] @ lora_params[key][0]) * lora_params[key][2]
            new_model.load_state_dict(updated_params)
            model_list.append(new_model)

    # Define validation scorer
    def get_scorer(obj):
        config = scorer_configs.get(obj, lambda: scorer_configs["default"](obj))()
        scorer_model = config["model"]
        if "type" in config:
            scorer_model.type = config["type"]
        return {"name": obj, "model": scorer_model, "sign": 1, "weight": 1.0, "train": True}
    
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
    val_objectives = ["reflection", "empathy", "fluency", "coherence", "specificity"]
    val_scorers = [get_scorer(obj) for obj in val_objectives]
    val_scorer = ScorerWrapper(val_scorers, learning_mode="bandit_weighted", scoring_method="logsum", max_batch_size=12)

    # Define generation parameters
    gen_params = {
        "max_new_tokens": 64,
        "early_stopping": True,
        "do_sample": True,
        "num_return_sequences": args.num_runs,
        "temperature": 1.0
    }

    return {
        'device': device,
        'val_dataloader': val_dataloader,
        'tokenizer': tokenizer,
        'model': model,
        'original_params': original_params,
        'model_list': model_list,
        'val_scorer': val_scorer,
        'gen_params': gen_params
    }

def logits_func(weights, args, df, device, val_dataloader, tokenizer, model, original_params, model_list, val_scorer, gen_params):

    current_scores = { k["name"]+"_scores":[] for k in val_scorer.scorers }
    for batch in val_dataloader:

        all_outputs = []
        for model_idx, model in enumerate(model_list):
            # Generate outputs with given prompts
            responses = batch["responses"]
            prompts = batch["prompts"]
            gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, \
                return_tensors="pt", padding="longest", truncation=True)
            gen_input = {k: v.to(device) for k, v in gen_input.items()}
            gens_out = model.generate(input_ids=gen_input["input_ids"], \
                attention_mask=gen_input["attention_mask"], output_scores=True, \
                return_dict_in_generate=True, **gen_params)
            all_outputs.append(gens_out)
                        
        # Merge logits
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

        # Use merged_scores to generate text
        generated_tokens = [[] for i in range(len(merged_scores[0]))]
        current_tokens = gen_input["input_ids"]
        for i in range(len(merged_scores[0])):
            for step_logits in merged_scores:
                # Acquire the most possible token
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

        scorer_returns = val_scorer.rl_score(inputs=prompts, generateds=generateds, responses=responses)
        for k,v in scorer_returns.items():
            if k in current_scores:
                current_scores[k].extend(v)

    # Calculate mean reward for this weight combination    
    rewards = [ np.mean(v) for k,v in current_scores.items() ]
    print(f"Rewards: {rewards}")
    mean_reward = np.mean(rewards[:3])
    d = np.append(weights, mean_reward)
    df.append(d.tolist())
    return mean_reward, df

def states_func(weights, args, df, device, val_dataloader, tokenizer, model, original_params, model_list, val_scorer, gen_params):

    current_scores = { k["name"]+"_scores":[] for k in val_scorer.scorers }
    for batch in val_dataloader:

        # Generate outputs with given prompts
        responses = batch["responses"]
        prompts = batch["prompts"]
        # Encode prompts for input
        gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, return_tensors="pt", padding="longest", truncation=True)
        gen_input = {k: v.to(device) for k, v in gen_input.items()}

        # Expand input ids and attention masks according to num_runs
        expanded_input_ids = gen_input["input_ids"].repeat_interleave(args.num_runs, dim=0)  # (test_batch_size * num_runs, seq_len)
        expanded_attention_mask = gen_input["attention_mask"].repeat_interleave(args.num_runs, dim=0)  # (test_batch_size * num_runs, seq_len)
        decoder_input_ids = torch.full(
            (args.val_batch_size * args.num_runs, 1),
            model_list[0].config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )  # (test_batch_size * num_runs, 1)

        generated_tokens = torch.zeros(args.val_batch_size*args.num_runs, 64, dtype=torch.long, device=device)
        # Generation by step
        for step in range(64):
            hidden_states_combined = torch.zeros(
                decoder_input_ids.size(0),  # test_batch_size * num_runs
                decoder_input_ids.size(1),  # current decoded ids length
                model_list[0].config.d_model,  # dimension of model hidden layers
                dtype=torch.float,
                device=device,
            )

            # Weighted combination of each model's hidden states
            for model_idx, model in enumerate(model_list):
                outputs = model(
                    input_ids=expanded_input_ids,
                    attention_mask=expanded_attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states_combined += weights[model_idx] * outputs.decoder_hidden_states[-1]

            # Compute logits and determine the next token
            logits = model_list[0].lm_head(hidden_states_combined) # lm_head of each model is same
            next_token = torch.argmax(logits[:, -1, :], dim=-1)  # shape: (test_batch_size * num_runs,)

            # Update generated_tokens
            generated_tokens[:, step] = next_token  # Store the generated token in the corresponding column

            # Update the decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=1)

            # Check if all sequences have generated an end token
            if (next_token == model_list[0].config.eos_token_id).all():
                break

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

        scorer_returns = val_scorer.rl_score(inputs=prompts, generateds=generateds, responses=responses)
        for k,v in scorer_returns.items():
            if k in current_scores:
                current_scores[k].extend(v)

    # Calculate mean reward for this weight combination    
    rewards = [ np.mean(v) for k,v in current_scores.items() ]
    print(f"Rewards: {rewards}")
    mean_reward = np.mean(rewards[:3])
    d = np.append(weights, mean_reward)
    df.append(d.tolist())
    return mean_reward, df

def params_func(weights, args, df, device, val_dataloader, tokenizer, model, original_params, model_list, val_scorer, gen_params):
    # Load lora params for all objectives
    updated_params = copy.deepcopy(original_params)
    new_model = copy.deepcopy(model)
    new_model.to(device)
    for i in range(len(args.objectives)):
        pattern = os.path.join(args.lora_path, f"lora_{args.objectives[i]}*.npz")
        matching_files = glob.glob(pattern)
        lora_params = load_lora(matching_files[0])
        for key in lora_params.keys():
            start_idx = len('base_model.model.')
            k = key[start_idx:] + '.weight'
            updated_params[k] = updated_params[k] + (lora_params[key][1] @ lora_params[key][0]) * lora_params[key][2] * weights[i]
    new_model.load_state_dict(updated_params)

    current_scores = { k["name"]+"_scores":[] for k in val_scorer.scorers }
    for batch in val_dataloader:
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
        scorer_returns = val_scorer.rl_score(prompts, generateds, responses=responses, step_count=None, bandit=None)
        for k,v in scorer_returns.items():
            if k in current_scores:
                current_scores[k].extend(v)

    # Calculate mean reward for this weight combination    
    rewards = [ np.mean(v) for k,v in current_scores.items() ]
    print(f"Rewards: {rewards}")
    mean_reward = np.mean(rewards[:3])
    d = np.append(weights, mean_reward)
    df.append(d.tolist())
    return mean_reward, df

def hierarchical_search(objective_func, args, num_components=3, iterations=5, **components):
    # Initialize wandb
    if args.do_wandb:
        wandb.init(project="DMORL-1", group="AGGREGATION", name=f"aggregation_{args.aggregation_mode}_{seed}")
        wandb.define_metric("mean_reward", step_metric="data_consuming")
    else:
        wandb.init(project="DMORL-1", mode="disabled")
    
    def generate_grid_points(bounds):
        """Generate 3x3x3 grid points in given bounds."""
        x_points = np.linspace(bounds[0][0], bounds[0][1], 3)
        y_points = np.linspace(bounds[1][0], bounds[1][1], 3)
        z_points = np.linspace(bounds[2][0], bounds[2][1], 3)
        grid_points = []
        for x in x_points:
            for y in y_points:
                for z in z_points:
                    grid_points.append([x, y, z])
        return np.array(grid_points)
    
    def evaluate_points(points, df):
        """Evaluate the metrics of all points and return."""
        results = {}  # Use dict to store results
        for point in points:
            point_tuple = tuple(point)  # Transfer tuple to dict keys
            print(point)
            score, df = objective_func(point, args, df, **components)
            results[point_tuple] = score
        return results, df
    
    def find_best_region(results, grid_points):
        """Find the region with the highest sum of eight neighbor points."""
        best_sum = float('-inf')
        best_region = None
        
        # Among 27 points, find all possible 8-point combinations (2x2x2 cube)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Get the 8 vertices of a 2x2x2 cube
                    region_points = []
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                idx = (i+di)*9 + (j+dj)*3 + (k+dk)
                                if idx < len(grid_points):
                                    region_points.append(tuple(grid_points[idx]))
                    
                    # If 8 complete points are found, calculate the region score
                    if len(region_points) == 8:
                        region_sum = sum(results[p] for p in region_points)
                        if region_sum > best_sum:
                            best_sum = region_sum
                            best_region = region_points
        
        return best_region
    
    def get_new_bounds(region_points):
        """Determine new search range based on the 8 points of the best region."""
        points = np.array([list(p) for p in region_points])
        bounds = []
        for dim in range(3):
            dim_min = points[:, dim].min()
            dim_max = points[:, dim].max()
            bounds.append((dim_min, dim_max))
        return bounds

    # Initial search range
    current_bounds = [(0, 1) for _ in range(num_components)]
    best_point = None
    best_score = float('-inf')
    
    df = []
    T_aggregation_start = time.time()
    for iter_num in range(iterations):
        print(f"\nIteration {iter_num + 1}:")
        
        # Generate grid points within current range
        grid_points = generate_grid_points(current_bounds)
        
        # Evaluate all points
        results, df = evaluate_points(grid_points, df)
        
        # Find the point with highest score
        best_point_iter = max(results.items(), key=lambda x: x[1])
        if best_point_iter[1] > best_score:
            best_score = best_point_iter[1]
            best_point = list(best_point_iter[0])
        
        print(f"Current best score: {best_score}")
        print(f"Current best point: {best_point}")
        
        # Find the region with the largest sum of 8 points
        best_region = find_best_region(results, grid_points)
        
        # Update search range
        current_bounds = get_new_bounds(best_region)
        print(f"New bounds: {current_bounds}")
    
    aggregation_time = time.time() - T_aggregation_start
    print(f"Entire aggregation time: {aggregation_time}")
    wandb.finish()

    txt_path = os.path.join(args.lora_path, "output3.txt")  # Combine folder path and filename
    # Write to file
    with open(txt_path, 'w') as f:
        f.write(f"Best score: {best_score}\n")
        f.write(f"Best weights: {best_point}\n")

    json_path = os.path.join(args.lora_path, "output3.json")  # Combine folder path and filename
    with open(json_path, 'w') as f:
        json.dump(df, f)
    return best_point, best_score

# def verify_lora(args, seed=5326, device="cuda", val_dataloader=None, tokenizer=None, model=None, \
#                         original_params=None, model_list=[], val_scorer=None, gen_params=None, aggregation_func=states_func):
#     wandb.init(project="DMORL", mode="disabled")
#     # Select only 0 and 1 for aggregating
#     weight_ranges= [np.array([0,2,1]), np.array([0,2,1]), np.array([0,2,1])]
#     best_weights, best_reward = aggregation_func(args, device, val_dataloader, tokenizer, model, original_params, model_list, \
#                                                 val_scorer, gen_params, weight_ranges=weight_ranges)
#     wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregation_mode', type=str, default="states", help="states/params/logits")
    parser.add_argument('--model_path', type=str, default="google-t5/t5-base")
    parser.add_argument('--data_path', type=str, default="data/PAIR/pair_data.csv")
    parser.add_argument('--lora_path', type=str, default="agg_results/")
    parser.add_argument('--objectives', nargs='+', default=["reflection", "empathy", "fluency"])
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--do_wandb', type=int, default=0, help="it doesn't make sense to record wandb")
    
    args = parser.parse_args()
    # save_args(args, "DL_AGGREGATION", "logs/")

    components = config_aggregation(args)
    best_point, best_score = hierarchical_search(objective_func=logits_func, args=args, **components)
if __name__ == "__main__":
    main()
            