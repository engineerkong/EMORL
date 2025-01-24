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
    original_params = model.state_dict() if args.aggregation_mode == "states" else None
    model.to(device)
    model.eval()

    # Gather models individual objectives
    model_list = []
    if args.aggregation_mode == "params":
        for i in range(len(args.objectives)):
            new_model = copy.deepcopy(model)
            new_model.to(device)
            updated_params = copy.deepcopy(original_params)
            lora_params = load_lora(args.lora_path+"lora_"+args.objectives[i]+".npz")
            for key in lora_params.keys():
                start_idx = len('base_model.model.')
                k = key[start_idx:] + '.weight'
                updated_params[k] = updated_params[k] + (lora_params[key][1] @ lora_params[key][0]) * lora_params[key][2]
            new_model.load_state_dict(updated_params)
            model_list.append(new_model)

    def get_scorer(obj):
        config = scorer_configs.get(obj, lambda: scorer_configs["default"](obj))()
        scorer_model = config["model"]
        if "type" in config:
            scorer_model.type = config["type"]
        return {"name": obj, "model": scorer_model, "sign": 1, "weight": 1.0, "train": True}
    
    # Define validation scorer
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


def states_func(args, device, val_dataloader, tokenizer, model, original_params, model_list, val_scorer, gen_params, weight_ranges):
    # Grid sample weights TODO
    for w1 in weight_ranges[0]:
        for w2 in weight_ranges[1]:
            w3 = 1.0 - (w1 + w2)
            if w3 < 0 or w3 > 1.0:
                continue  # Skip invalid weight combinations

            weights = torch.tensor([w1, w2, w3], device=device)
            rewards = []

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

                generated_tokens = torch.zeros(args.val_batch_size*args.num_runs, max_output_length=64, dtype=torch.long, device=device)
                # Generation by step
                for step in range(max_output_length=64):
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
            mean_reward = np.mean(rewards[:2])
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_weights = weights.cpu().numpy()
            print(f"Weights:{weights}")
            print(f"Rewards:{rewards}")

    return {
        "best_weights": best_weights,
        "best_mean_reward": best_mean_reward,
    }

def params_func(args, device, val_dataloader, tokenizer, model, original_params, model_list, val_scorer, gen_params, weight_ranges):
    # Grid sample weights TODO
    for w1 in weight_ranges[0]:
        for w2 in weight_ranges[1]:
            for w3 in weight_ranges[2]:
                weights = torch.tensor([w1, w2, w3], device=device)
                rewards = []

                # Load lora params for all objectives
                updated_params = copy.deepcopy(original_params)
                new_model = copy.deepcopy(model)
                new_model.to(device)
                for i in range(len(args.objectives)):
                    lora_params = load_lora(args.lora_path+"lora_"+args.objectives[i]+".npz")
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
                mean_reward = np.mean(rewards[:3])
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    best_weights = weights
                print(f"Weights:{weights}")
                print(f"Rewards:{rewards}")

    return {
        "best_weights": best_weights,
        "best_mean_reward": best_mean_reward,
    }

def hierarchical_search(args, seed=5326, precision_levels=[0.1, 0.01], device="cuda", val_dataloader=None, tokenizer=None, model=None, \
                        original_params=None, model_list=[], val_scorer=None, gen_params=None, aggregation_func=states_func):
    # Initialize wandb
    if args.do_wandb:
        wandb.init(project="DMORL", group="AGGREGATION", name=f"aggregation_{args.aggregation_mode}_{seed}")
        wandb.define_metric("mean_reward", step_metric="data_consuming")
    else:
        wandb.init(project="DMORL", mode="disabled")
    # Define a hierarchical search
    best_values = {}
    for precision in precision_levels:
        if precision == precision_levels[0]:
            weight_ranges = [np.arange(0, 1+precision, precision) for _ in range(len(args.objectives))]
        else:
            weight_ranges = [
                np.arange(
                    max(0, best_weights[i]-precision_levels[0]),
                    min(1, best_weights[i]+precision_levels[0])+precision,
                    precision
                ) for i in range(len(args.objectives))
            ]
        # Aggregation for every levels
        best_weights, best_reward = aggregation_func(args, device, val_dataloader, tokenizer, model, original_params, model_list, \
                                                    val_scorer, gen_params, weight_ranges=weight_ranges)
        best_values[precision] = {
            'weights': best_weights,
            'reward': best_reward
        }
    wandb.finish()

    return best_values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--aggregation_mode', type=str, default="states", help="states/params")
    parser.add_argument('--model_path', type=str, default="models/t5-base")
    parser.add_argument('--data_path', type=str, default="data/PAIR/pair_data.csv")
    parser.add_argument('--lora_path', type=str, default="lora_results/20250115/")
    parser.add_argument('--output_path', type=str, default="aggregation_results/")
    parser.add_argument('--objectives', nargs='+', default=["reflection", "empathy", "fluency"])
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--do_wandb', type=int, default=0, help="it doesn't make sense to record wandb")
    
    args = parser.parse_args()
    # save_args(args, "DL_AGGREGATION", "logs/")

    timestamp = time.strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(args.output_path, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    components = config_aggregation(args)

    seeds = [random.randint(1, 100000) for _ in range(args.num_seeds)]
    for seed in seeds:
        set_seed(seed)
        
        if args.aggregation_mode == "states":
            print(f"Start aggregating last_hidden_states")
            best_values = hierarchical_search(args, seed, **components, agg_func=states_func)
            print(f"Best for aggregating states of {args.objectives}: {best_values[-1]}")
        elif args.aggregation_mode == "params":
            print(f"Start aggregating parameters of encoder and decoder")
            best_values = hierarchical_search(args, seed, **components, agg_func=params_func)
            print(f"Best for aggregating params of {args.objectives}: {best_values[-1]}")
        else:
            raise("Training Mode Wrong!")

if __name__ == "__main__":
    main()
            