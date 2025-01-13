import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import copy
import random
import numpy as np
import wandb

from dynaopt_lib import *
from lora_utils import *
from additional_utils import *
from options import args_parser, save_options

def rl_train(model, tokenizer, ref_model, optimizer, scaler, rl_crit, scorer, train_dataloader, val_dataloader, \
            gen_params, device, num_runs, train_batch_size, num_steps=1000, val_step=100, max_seq_length=128, max_output_length=64):
    # Training loop
    step_count = 0
    model.train()

    while step_count < num_steps:
        print(f"step_count:{step_count}")
        for batch in train_dataloader:
            prompts = batch["prompts"]
            responses = batch["responses"]

            # # automatic mixed precision (unused)
            # with torch.amp.autocast('cuda'):

            # Forward pass
            gen_input = tokenizer.batch_encode_plus(prompts, max_length=max_seq_length, \
                return_tensors="pt", padding="longest", truncation=True)
            gen_input = {k: v.to(device) for k, v in gen_input.items()}
            gens_out = model.generate(input_ids=gen_input["input_ids"],\
                # decoder_start_token_id=tokenizer.bos_token_id,\
                attention_mask=gen_input["attention_mask"], **gen_params)
            # Decode generations
            generateds = tokenizer.batch_decode(gens_out, skip_special_tokens=True)
            # generateds = [ [ x.strip() for x in g.split("[CLS]")[:-1]] for g in generateds] # original will lead to []
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
            gens_out = tokenizer.batch_encode_plus(cls_generateds, max_length=max_output_length, \
                return_tensors="pt", padding="longest", truncation=True)["input_ids"]  
            prompts = [p for p in prompts for _ in range(num_runs)]
            responses = [r for r in responses for _ in range(num_runs)]

            # Calculate rewards - RL - kSCST
            scorer_returns = scorer.score(prompts, generateds, responses=responses, step_count=step_count, bandit=None, chosen=None)
            total_scores = torch.FloatTensor(scorer_returns["total_scores"]).cuda()
            batch_scores = total_scores.reshape(train_batch_size, num_runs)
            mean_scores = batch_scores.mean(dim=1)
            max_scores = torch.max(batch_scores, dim=1).values 
            unlooped_mean_scores = torch.repeat_interleave(mean_scores, num_runs)
            normalized_rewards = (unlooped_mean_scores - total_scores)
            n_diff_pos, n_diff_neg = (normalized_rewards<-0.02).long().sum().item(), (normalized_rewards>0.02).long().sum().item()

            # Calculate loss with KL penalty
            loss = rl_crit(prompts, gens_out, normalized_rewards)
            # loss = loss.requires_grad_(True)

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step_count += 1

            # Validation
            current_scores = { k["name"]+"_scores":[] for k in scorer.scorers }
            if step_count !=0 and  step_count % val_step == 0:
                for batch in val_dataloader:
                    responses = batch["responses"]
                    prompts = batch["prompts"]

                    with torch.amp.autocast('cuda'):
                        gen_input = tokenizer.batch_encode_plus(prompts, max_length=max_seq_length, \
                            return_tensors="pt", padding="longest", truncation=True)
                        gen_input = {k: v.to(device) for k, v in gen_input.items()}
                        gens_out = model.generate(input_ids=gen_input["input_ids"],\
                            # decoder_start_token_id=tokenizer.bos_token_id,\
                            attention_mask=gen_input["attention_mask"], **gen_params)
                        generateds = tokenizer.batch_decode(gens_out, skip_special_tokens=True)
                        # cut_generateds = [ [ x.strip() for x in g.split("[CLS]")[:-1]] for g in generateds]
                        # print(cut_generateds)
                        # new = []
                        # for c, g in zip(cut_generateds, generateds):
                        #     if c == []:
                        #         new.append([g])
                        #     else:
                        #         new.append(c)
                        # generateds = [ " ".join(g) for g in new]
                        # generateds = [ g.replace("<pad>", "").strip() for g in generateds]
                        # generateds = [g.replace("[CLS]", "").strip() for g in generateds]
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
                    scorer_returns = scorer.rl_score(prompts, generateds, responses=responses, step_count=step_count, bandit=None)
                    for k,v in scorer_returns.items():
                        if k in current_scores:
                            current_scores[k].extend(v)
                    print("Mean Rewards", [ np.mean(v) for k,v in current_scores.items() ])

    lora_params = acquire_lora_params(model)

    return model, tokenizer, lora_params

def collate_fn(batch):
    prompts = [item["prompt"] + " [SEP] " for item in batch]
    responses = [item["response"] for item in batch]
    return {"prompts": prompts, "responses": responses}

# 2. 确保LoRA相关参数可训练
def check_lora_trainable(model):
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params.append(f"{name}: requires_grad = {param.requires_grad}")
    return lora_params
    
def main():
    args = args_parser()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.init(project="DynaDRL", mode="disabled")

    # Load data
    train_data, val_data, test_data = get_data(args.data_path)
    train_dataloader = DataLoader(dataset=train_data, batch_size=args.train_batch_size,\
        sampler=RandomSampler(train_data), drop_last=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_data, batch_size=args.val_batch_size,\
        sampler=SequentialSampler(val_data), drop_last=True, collate_fn=collate_fn)

    # Train on multiple objective
    for objective in args.objectives:
        # Load models
        model, tokenizer = get_model(
            args.model_path,  
            max_seq_length=128,
            max_output_length=64,
            lora=True
        )
        model = setup_lora_config(model)
        model.to(device)

        # Initialize reference model
        ref_model = copy.deepcopy(model)
        ref_model.to(device)
        ref_model.eval()
        ref_model.requires_grad_(False)
        
        # Setup optimizer and apex scaler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')

        # Create criterion and scorer 
        rl_crit = ReinforceCriterion(model, tokenizer, optimizer, scaler, ref_model=ref_model, kl_coeff=0.05)
        scorers = [{"name": objective, "model": Multi(type=objective, experiment='train'), "sign": 1, "weight": 1.0, "train": True}]
        scorer = ScorerWrapper(scorers, learning_mode = "single", \
                scoring_method="logsum", max_batch_size=12)

        gen_params = {
            "max_new_tokens": 64,
            "early_stopping": True,
            "do_sample": True,
            "num_return_sequences": args.num_runs,
            "temperature": 1.0
        }
        # Train model
        model, tokenizer, lora_params = rl_train(model=model, tokenizer=tokenizer, ref_model=ref_model, optimizer=optimizer, \
                                                scaler=scaler, rl_crit=rl_crit, scorer=scorer, train_dataloader=train_dataloader, \
                                                val_dataloader=val_dataloader, gen_params=gen_params, device=device, \
                                                num_runs=args.num_runs, train_batch_size=args.train_batch_size, num_steps=args.num_steps)


        save_lora(lora_params, npz_path=args.save_lora_path)
if __name__ == "__main__":
    main()