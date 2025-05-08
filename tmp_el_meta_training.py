import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import logging
from tqdm import tqdm
import copy
import glob
import wandb

from mha import MOMultiHeadAttentionLayer
from model_empathy import *
from dynaopt_lib import *
from utils_lora import *
from utils_additional import *

class MetaLearner(nn.Module):
    def __init__(self, base_model, n_heads, embed_dim, feed_forward_hidden, normalization, device):
        super().__init__()
        self.base_model = base_model
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.feed_forward_hidden = feed_forward_hidden
        self.normalization = normalization
        self.device = device

        self.mha = MOMultiHeadAttentionLayer(self.n_heads, self.embed_dim, self.feed_forward_hidden, self.normalization)
        self.lm_head = self.base_model.lm_head

    def __call__(self, input_ids=None, attention_mask=None, labels=None, output_hidden_states=False, return_dict=True, **kwargs):
        """
        Makes MetaLearner compatible with the ReinforceCriterion by implementing 
        the expected forward interface of the language model.
        """
        
        # Process through mha and get combined output
        hidden_states_combined = torch.zeros(
            len(self.model_list),
            input_ids.size(0),
            input_ids.size(1),
            self.model_list[0].config.d_model,
            dtype=torch.float,
            device=self.device,
        )
        
        # Get logits from each model
        for model_idx, model in enumerate(self.model_list):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )
            # Store the last hidden state for MHA
            hidden_states_combined[model_idx] = outputs.decoder_hidden_states[-1]
        
        # Process through multi-head attention
        mha_states = self.mha(hidden_states_combined)
        # Get logits from lm_head
        logits = self.lm_head(mha_states)
        
        # Create a structure similar to what the base model would return
        from dataclasses import dataclass
        
        @dataclass
        class ModelOutput:
            logits: torch.Tensor
            loss: torch.Tensor = None
        
        return ModelOutput(logits=logits)

    def config_models(self, lora_updates):
        original_params = self.base_model.state_dict()
        assert len(lora_updates) == self.n_heads, "lora_updates should have the same number of heads as the model"
        self.model_list = []
        for lora_params in lora_updates:
            new_model = copy.deepcopy(self.base_model)
            new_model.to(self.device)
            updated_params = copy.deepcopy(original_params)
            for key in lora_params.keys():
                start_idx = len('base_model.model.')
                k = key[start_idx:] + '.weight'
                updated_params[k] = updated_params[k] + (lora_params[key][1] @ lora_params[key][0]) * lora_params[key][2]
            new_model.load_state_dict(updated_params)
            self.model_list.append(new_model)
        
    def forward(self, gen_input, num_runs, train_batch_size, max_output_length):
        assert len(self.model_list) == self.n_heads, "features should have the same number of heads as the model"
        # Expand input ids and attention masks according to num_runs
        expanded_input_ids = gen_input["input_ids"].repeat_interleave(num_runs, dim=0)  # (train_batch_size * num_runs, seq_len)
        expanded_attention_mask = gen_input["attention_mask"].repeat_interleave(num_runs, dim=0)  # (train_batch_size * num_runs, seq_len)
        decoder_input_ids = torch.full(
            (train_batch_size * num_runs, 1),
            self.model_list[0].config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )  # (train_batch_size * num_runs, 1)
        hidden_states_combined = torch.zeros(
            len(self.model_list),  # number of models
            decoder_input_ids.size(0),  # train_batch_size * num_runs
            decoder_input_ids.size(1),  # current decoded ids length
            self.model_list[0].config.d_model,  # dimension of model hidden layers
            dtype=torch.float,
            device=self.device,
        )
        generated_tokens = torch.zeros(train_batch_size*num_runs, max_output_length, dtype=torch.long, device=self.device)
        hidden_states_list = []
        for model_idx, model in enumerate(self.model_list):
            generated_outputs = model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_length=max_output_length,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            model_hidden_states = torch.stack([step[-1] for step in generated_outputs.decoder_hidden_states])
            hidden_states_list.append(model_hidden_states)
        all_models_hidden_states = torch.stack(hidden_states_list)
        hidden_states_combined = all_models_hidden_states.permute(1, 0, 2, 3, 4)
        print(f"hidden_states_combined shape: {hidden_states_combined.shape}")

        # mha_states = self.mha(hidden_states_combined)
        # logits = self.lm_head(mha_states)  # shape: (num_heads, train_batch_size * num_runs, vocab_size)
        # next_token = torch.argmax(logits[:, -1, :], dim=-1)  # shape: (train_batch_size * num_runs,)
        # generated_tokens[:, i] = next_token

        # # Generation by step
        # for step in range(max_output_length):
        #     hidden_states_combined = torch.zeros(
        #         len(self.model_list),  # number of models
        #         decoder_input_ids.size(0),  # train_batch_size * num_runs
        #         decoder_input_ids.size(1),  # current decoded ids length
        #         self.model_list[0].config.d_model,  # dimension of model hidden layers
        #         dtype=torch.float,
        #         device=self.device,
        #     )

        #     # Weighted combination of each model's hidden states
        #     for model_idx, model in enumerate(self.model_list):
        #         outputs = model(
        #             input_ids=expanded_input_ids,
        #             attention_mask=expanded_attention_mask,
        #             decoder_input_ids=decoder_input_ids,
        #             output_hidden_states=True,
        #             return_dict=True,
        #         )
        #         hidden_states_combined[model_idx] = outputs.decoder_hidden_states[-1]

        # mha_states = self.mha(hidden_states_combined)
        # logits = self.lm_head(mha_states)  # shape: (num_heads, test_batch_size * num_runs, vocab_size)
        # next_token = torch.argmax(logits[:, -1, :], dim=-1)  # shape: (test_batch_size * num_runs,)
        # generated_tokens[:, step] = next_token
        # # Update the decoder input
        # decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=1)

        # # Check if all sequences have generated an end token
        # if (next_token == self.model_list[0].config.eos_token_id).all():
        #     break

        return generated_tokens

def config_scorer(meta_model, tokenizer, ref_model, device):
    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(meta_model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device=device)

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
    objectives = ["reflection", "empathy", "fluency"]
    scorers = [get_scorer(obj) for obj in objectives]
    rl_crit = ReinforceCriterion(model=meta_model, tokenizer=tokenizer, optimizer=optimizer, scaler=scaler, ref_model=ref_model, kl_coeff=0.05)
    scorer = ScorerWrapper(scorers, learning_mode="weighted", scoring_method="logsum", max_batch_size=12)

    return optimizer, scaler, scorer, rl_crit

def meta_train(meta_model, tokenizer, train_dataloader, val_dataloader, optimizer, scaler, scorer, rl_crit, gens_params, 
                train_batch_size, val_batch_size, val_interval_size, num_runs, num_steps, device):
    step_count = 0
    training = True
    while step_count < num_steps and training:
        print(f"Step count:{step_count}")
        
        # Training on train dataset, batches 15
        for batch in train_dataloader:
            # Generate inputs
            prompts = batch["prompts"]
            responses = batch["responses"]
            gen_input = tokenizer.batch_encode_plus(prompts, max_length=128, \
                return_tensors="pt", padding="longest", truncation=True)
            gen_input = {k: v.to(device) for k, v in gen_input.items()}

            # meta_model forward pass
            meta_model.train()
            generated_tokens = meta_model.forward(gen_input, num_runs, train_batch_size, max_output_length=32)

            # Generate outputs
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
            gens_out = tokenizer(generateds, max_length=128, \
                return_tensors="pt", padding="longest", truncation=True)["input_ids"]
            prompts = [p for p in prompts for _ in range(num_runs)]
            responses = [r for r in responses for _ in range(num_runs)]

            # Calculate scorers - kSCST
            scorer_returns = scorer.score(prompts, generateds, responses=responses, step_count=step_count, \
                                          bandit=None, \
                                          chosen=None)
            total_scores = torch.FloatTensor(scorer_returns["total_scores"]).cuda()
            print(f"Total scores: {total_scores}")
            batch_scores = total_scores.reshape(train_batch_size, num_runs)
            mean_scores = batch_scores.mean(dim=1)
            unlooped_mean_scores = torch.repeat_interleave(mean_scores, num_runs)
            normalized_rewards = (unlooped_mean_scores - total_scores)

            # Calculate loss with KL penalty
            loss = rl_crit(prompts, gens_out, normalized_rewards)

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=2.0, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update step_count when one batch is finished
            step_count += 1

def main(model_name, data_path, lora_path, device, train_batch_size, val_batch_size, val_interval_size, num_runs, num_steps):
    # Load data
    train_data, val_data, test_data = get_data(data_path)
    train_dataloader = DataLoader(dataset=train_data, batch_size=train_batch_size,\
        sampler=RandomSampler(train_data), drop_last=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_data, batch_size=val_batch_size,\
        sampler=RandomSampler(val_data), drop_last=True, collate_fn=collate_fn)

    # Load model and tokenizer
    model, tokenizer = get_model(
        model_name,  
        max_seq_length=128,
        max_output_length=64,
        lora=True
    )
    model.to(device)

    # Load reference model for KL divergence
    ref_model = copy.deepcopy(model)
    ref_model.to(device)
    ref_model.eval()
    ref_model.requires_grad_(False)

    # Initialize meta-learner
    meta_learner = MetaLearner(
        base_model=model,
        n_heads=3,
        embed_dim=768,
        feed_forward_hidden=512,
        normalization='batch',
        device='cuda'
    )
    # Load LoRA updates
    pattern = os.path.join(lora_path, f"lora_*.npz")
    matching_files = glob.glob(pattern)
    lora_updates = []
    for matching_file in matching_files:
        print(f"Loading LoRA parameters from {matching_file}")
        lora_params = load_lora(matching_file)
        lora_updates.append(lora_params)
    meta_learner.config_models(lora_updates)

    # Define generation parameters
    gen_params = {
        "max_new_tokens": 64,
        "early_stopping": True,
        "do_sample": True,
        "num_return_sequences": num_runs,
        "temperature": 1.0,
        "pad_token_id": tokenizer.eos_token_id  # DialoGPT uses eos_token as pad_token
    }

    # Train the meta-learner
    optimizer, scaler, scorer, rl_crit = config_scorer(meta_learner, tokenizer, ref_model, device)
    meta_train(meta_learner, tokenizer, train_dataloader, val_dataloader, optimizer, scaler, scorer, rl_crit, gen_params, 
            train_batch_size, val_batch_size, val_interval_size, num_runs, num_steps, device)

if __name__ == "__main__":
    wandb.init(project="Improved_EMORL", mode="disabled")

    # Default configurations
    model_name = "google-t5/t5-base"
    data_path = "data/PAIR/pair_data.csv"
    lora_path = "lora_results/results_example"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_batch_size = 8
    val_batch_size = 8
    val_interval_size = 8
    num_runs = 3
    num_steps = 1000

    main(model_name, data_path, lora_path, device, train_batch_size, val_batch_size, val_interval_size, num_runs, num_steps)