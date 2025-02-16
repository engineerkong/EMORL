import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import copy
import random
import numpy as np
import wandb
import argparse
import time
import os
import glob
import pandas as pd
from model_empathy import *
from dynaopt_lib import *
from utils_lora import *
from utils_additional import *
    
def config_testing(args):

    # Load device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    if args.test_datasets == "PAIR":
        train_data, val_data, test_data = get_data(args.data_path)
    elif args.test_datasets == "P8K":
        train_data, val_data, test_data = get_p8k(args.data_path)
    test_dataloader = DataLoader(dataset=test_data, batch_size=args.test_batch_size,\
        sampler=RandomSampler(test_data), drop_last=True, collate_fn=collate_fn)

    # Load original model
    model, tokenizer = get_model(
        args.model_path,  
        max_seq_length=128,
        max_output_length=64,
        lora=False
    )
    model.to(device)
    model.eval()

    # Define testing scorer
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
            "model": Multi(type=o, score_change=False)}
    }
    test_objectives = ["reflection", "empathy", "fluency", "coherence", "specificity", "diveristy-2", "edit_rate"]
    test_scorers = [get_scorer(obj) for obj in test_objectives]
    test_scorer = ScorerWrapper(test_scorers, learning_mode="bandit_weighted", scoring_method="logsum", max_batch_size=12)

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
        'test_dataloader': test_dataloader,
        'tokenizer': tokenizer,
        'model': model,
        'test_scorer': test_scorer,
        'gen_params': gen_params,
    }

def test(args, device, test_dataloader, tokenizer, model, test_scorer, gen_params):
    # Enumerate models parameters
    if os.path.isfile(args.model_params):
        file_path = os.path.join(args.model_params)
        # Load parameters to original model
        c_ckpt = torch.load(file_path, weights_only=True) 
        model.load_state_dict(c_ckpt, strict=False) 
    elif os.path.isdir(args.model_params):
        original_params = model.state_dict()
        model_list = []
        for i in range(len(args.objectives)):
            new_model = copy.deepcopy(model)
            new_model.to(device)
            updated_params = copy.deepcopy(original_params)
            pattern = os.path.join(args.model_params, f"lora_{args.objectives[i]}*.npz")
            matching_files = glob.glob(pattern)
            lora_params = load_lora(matching_files[0])
            for key in lora_params.keys():
                start_idx = len('base_model.model.')
                k = key[start_idx:] + '.weight'
                updated_params[k] = updated_params[k] + (lora_params[key][1] @ lora_params[key][0]) * lora_params[key][2]
            new_model.load_state_dict(updated_params)
            model_list.append(new_model)
    else:
        pass
    print(f"Load model: {args.model_params}")

    weights = args.weight_combination # [np.float64(0.78125), np.float64(0.5), np.float64(0.0625)]
    # Testing loop
    prompts_list, generateds_list = [], []
    current_scores = { k["name"]+"_scores":[] for k in test_scorer.scorers }
    for n in range(4):
        for i, batch in enumerate(test_dataloader):
            if i >= 4:
                break
            # Generate outputs with given prompts
            responses = batch["responses"]
            prompts = batch["prompts"]
            # Encode prompts for input
            gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, return_tensors="pt", padding="longest", truncation=True)
            gen_input = {k: v.to(device) for k, v in gen_input.items()}

            if os.path.isfile(args.model_params) or args.model_params == "":
                # Generate outputs with given prompts
                responses = batch["responses"]
                prompts = batch["prompts"]
                gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, \
                    return_tensors="pt", padding="longest", truncation=True)
                gen_input = {k: v.to(device) for k, v in gen_input.items()}
                gens_out = model.generate(input_ids=gen_input["input_ids"], \
                    attention_mask=gen_input["attention_mask"], **gen_params)
            elif os.path.isdir(args.model_params):
                # Expand input ids and attention masks according to num_runs
                expanded_input_ids = gen_input["input_ids"].repeat_interleave(args.num_runs, dim=0)  # (test_batch_size * num_runs, seq_len)
                expanded_attention_mask = gen_input["attention_mask"].repeat_interleave(args.num_runs, dim=0)  # (test_batch_size * num_runs, seq_len)
                decoder_input_ids = torch.full(
                    (args.test_batch_size * args.num_runs, 1),
                    model_list[0].config.decoder_start_token_id,
                    dtype=torch.long,
                    device=device,
                )  # (test_batch_size * num_runs, 1)

                gens_out = torch.zeros(args.test_batch_size*args.num_runs, 64, dtype=torch.long, device=device)
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
                    gens_out[:, step] = next_token  # Store the generated token in the corresponding column

                    # Update the decoder input
                    decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=1)

                    # Check if all sequences have generated an end token
                    if (next_token == model_list[0].config.eos_token_id).all():
                        break
            
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

            # Record prompts and generateds
            prompts_list.extend(prompts)
            generateds_list.extend(generateds)

            # Calculate scores
            scorer_returns = test_scorer.rl_score(prompts, generateds, responses=responses, step_count=None, bandit=None)
            for k,v in scorer_returns.items():
                if k in current_scores:
                    current_scores[k].extend(v)

    # Calculate mean reward for this weight combination    
    rewards = [ np.mean(v) for k,v in current_scores.items() ]
    mean_reward = np.mean(rewards[:3])
    print(f"Reward: {rewards}")
    print(f"Mean Reward of 3: {mean_reward}")

    if args.output_path:
        # Record prompts and generateds
        data = {
            "prompts": prompts_list,
            "generateds": generateds_list
            }
        df = pd.DataFrame(data)
        try:
            with pd.ExcelWriter(args.output_path, mode='a') as writer:
                df.to_excel(writer, sheet_name=args.model_params, index=False)
        except:
            with pd.ExcelWriter(args.output_path, mode='w') as writer:
                df.to_excel(writer, sheet_name=args.model_params, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="google-t5/t5-base")
    parser.add_argument('--test_datasets', type=str, default="P8K")
    parser.add_argument('--data_path', type=str, default="data/P8K/Psycho8k.json")
    parser.add_argument('--model_params', type=str, default="")
    parser.add_argument('--weight_combination', nargs='+', default=[0.78125, 0.5, 0.0625])
    parser.add_argument('--objectives', nargs='+', default=["reflection", "empathy", "fluency"])
    parser.add_argument('--output_path', type=str, default="output_psycho8k.xlsx")
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--num_runs', type=int, default=1)
    
    args = parser.parse_args()
    # save_args(args, "DL_TESTING", "logs/")

    set_seed(567654) # Use one random seed for the testing
    components = config_testing(args)

    test(args, **components)

if __name__ == "__main__":
    main()