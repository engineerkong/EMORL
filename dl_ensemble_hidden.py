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

def grid_search_hidden_weights(model_list, tokenizer, dataloader, scorer, gen_params, max_output_length=64, grid_size=101, test_batch_size=1, num_runs=1, device="cuda"):
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
        w2 = 1.0 - w1
        # for w2 in weight_range:
        #     w3 = 1.0 - (w1 + w2)
        #     if w3 < 0 or w3 > 1.0:
        #         continue  # Skip invalid weight combinations

        weights = torch.tensor([w1, w2], device=device)
        rewards = []

        current_scores = { k["name"]+"_scores":[] for k in scorer.scorers }
        for batch in dataloader:
            
            # Generate outputs with given prompts
            responses = batch["responses"]
            prompts = batch["prompts"]
            # Encode prompts for input
            gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, return_tensors="pt", padding="longest", truncation=True)
            gen_input = {k: v.to(device) for k, v in gen_input.items()}

            # 扩展输入和解码器输入
            expanded_input_ids = gen_input["input_ids"].repeat_interleave(num_runs, dim=0)  # (test_batch_size * num_runs, seq_len)
            expanded_attention_mask = gen_input["attention_mask"].repeat_interleave(num_runs, dim=0)  # (test_batch_size * num_runs, seq_len)
            decoder_input_ids = torch.full(
                (test_batch_size * num_runs, 1),
                model_list[0].config.decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )  # (test_batch_size * num_runs, 1)

            generated_tokens = torch.zeros(test_batch_size*num_runs, max_output_length, dtype=torch.long, device=device)
            # 逐步生成
            for step in range(max_output_length):
                hidden_states_combined = torch.zeros(
                    decoder_input_ids.size(0),  # test_batch_size * num_runs
                    decoder_input_ids.size(1),  # 当前解码长度
                    model_list[0].config.d_model,  # 模型隐藏层维度
                    dtype=torch.float,
                    device=device,
                )

                # 加权组合每个模型的隐藏状态
                for model_idx, model in enumerate(model_list):
                    outputs = model(
                        input_ids=expanded_input_ids,
                        attention_mask=expanded_attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states_combined += weights[model_idx] * outputs.decoder_hidden_states[-1]

                # 计算 logits 并确定下一个 token
                logits = model_list[0].lm_head(hidden_states_combined) # 模型的lm_head均一致
                next_token = torch.argmax(logits[:, -1, :], dim=-1)  # shape: (test_batch_size * num_runs,)

                # 更新 generated_tokens
                generated_tokens[:, step] = next_token  # 将生成的 token 存储在对应列

                # 更新解码器输入
                decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=1)

                # 检查是否所有序列都生成结束标记
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
            prompts = [p for p in prompts for _ in range(num_runs)]
            responses = [r for r in responses for _ in range(num_runs)]

            scorer_returns = scorer.rl_score(inputs=prompts, generateds=generateds, responses=responses)
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
    
    # Create validation criterion and scorer
    reflection_scorer_model = ReflectionScoreDeployedCL(score_change=False, model_file= "./weights/reflection_scorer_weight.pt")
    reflection_scorer_model.type = "CLM"
    empathy_scorer_model = EmpathyScoreDeployedCL(score_change=False)
    empathy_scorer_model.type = "CLM"
    scorers = [{"name": "reflection", "model": reflection_scorer_model, "sign": 1, "weight": 1.0, "train": True},
                {"name": "empathy", "model": empathy_scorer_model, "sign": 1, "weight": 1.0, "train": True},
                {"name": "fluency", "model": Multi(type="fluency"), "sign": 1, "weight": 1.0, "train": True},
                {"name": "coherence", "model": Multi(type="coherence"), "sign": 1, "weight": 1.0, "train": True},
                {"name": "specificity", "model": Multi(type="specificity"), "sign": 1, "weight": 1.0, "train": True}]
    scorer = ScorerWrapper(scorers, learning_mode = "bandit_weighted", scoring_method="logsum", max_batch_size=12)
    
    # Gather models individual objectives
    model_list = []
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
    
    # Perform grid search
    results = grid_search_hidden_weights(
        model_list=model_list,
        tokenizer=tokenizer,
        dataloader=test_dataloader,
        scorer=scorer,
        gen_params=gen_params,
        grid_size=101,  # e.g., weights range from 0.0 to 1.0 in 0.1 steps
        test_batch_size=args.test_batch_size,
        num_runs=args.num_runs,
        device=device
    )

    # Print results
    print("Best Weights:", results["best_weights"])
    print("Best Mean Reward:", results["best_mean_reward"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=429023)
    parser.add_argument('--model_path', type=str, default="models/t5-base")
    parser.add_argument('--data_path', type=str, default="data/PAIR/pair_data.csv")
    parser.add_argument('--lora_path', type=str, default="lora_results/20250115/")
    parser.add_argument('--output_path', type=str, default="ensemble_results/")
    parser.add_argument('--objectives', nargs='+', default=["reflection", "fluency", "coherence"])
    parser.add_argument('--test_batch_size', type=int, default=4)    
    parser.add_argument('--test_history_size', type=int, default=10)
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--do_wandb', type=int, default=0)
    
    args = parser.parse_args()
    # save_args(args, "DL_ENSEMBLE", "logs/")
    
    # Run ensemble
    model = ensemble(args)
    
    return model

if __name__ == "__main__":
    main()
            