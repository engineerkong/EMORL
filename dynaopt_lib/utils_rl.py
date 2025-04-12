from . import utils_edits

import time, numpy as np, torch
from datetime import datetime
import torch.nn.functional as F
import wandb
def reduce_sum(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask)
    return torch.sum(value * mask, axis)
def reduce_mean(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask) / torch.sum(mask)
    return reduce_sum(value, mask, axis) / torch.sum(mask, axis)
def select_logprobs(logits, decoded_tokens, eos_id, reward_shaping="default", attentions=None):
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected_logprobs = []
    for i, generated_tokenized in enumerate(decoded_tokens):
        generated_tokenized.append(eos_id)
        generated_tokenized = generated_tokenized[:generated_tokenized.index(eos_id)] 
        selected_logprob = logprobs[i, torch.arange(len(generated_tokenized)), generated_tokenized]
        summed_logprob = torch.sum(selected_logprob)
        selected_logprobs.append(summed_logprob)
    selected_logprobs = torch.stack(selected_logprobs, dim=0)
    return selected_logprobs
class ReinforceCriterion:
    def __init__(self, model, tokenizer, optimizer, scaler, use_apex=False, \
                 reward_shaping="default", ref_model=None, ref_tokenizer=None, ref_optimizer=None, kl_coeff=0.05):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.eos_id = self.tokenizer.eos_token_id
        self.use_apex = use_apex
        self.scaler = scaler
        self.reward_shaping = reward_shaping
        self.ref_model = ref_model
        self.kl_coeff = kl_coeff
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")
    def __call__(self, prompt_inputs, decoded_tokens, rewards, train_model=True, sampled_actions=None, freeze_responder=False):
        if not train_model:
            return 0.0
        assert len(prompt_inputs)==len(decoded_tokens), "There's a shape mismatch between inputs and outputs %d != %d" % (len(prompt_inputs), len(decoded_tokens))
        
        # Check if we're using DialoGPT (decoder-only) or T5 (encoder-decoder)
        is_dialogpt = hasattr(self.model, 'config') and hasattr(self.model.config, '_name_or_path') and 'DialoGPT' in self.model.config._name_or_path
        
        encoded_prompt = self.tokenizer(prompt_inputs, return_tensors="pt", padding="longest", truncation=True, max_length=512)
        encoded_prompt = {k: v.to(self.model.device) for k, v in encoded_prompt.items()}
        decoded_tokens_tensor = decoded_tokens.to(self.model.device)
        
        if is_dialogpt:
            # For DialoGPT, we need to handle the loss calculation differently
            # We'll use the model's forward pass without labels and calculate the loss manually
            output = self.model(input_ids=encoded_prompt['input_ids'],
                                attention_mask=encoded_prompt['attention_mask'])
            logits = output.logits
            
            # Get the logits for the generated tokens
            decoded_tokens_list = decoded_tokens_tensor.tolist()
            selected_logprobs = []
            
            for i, tokens in enumerate(decoded_tokens_list):
                # Filter out padding tokens
                tokens = [t for t in tokens if t != self.tokenizer.pad_token_id]
                if tokens:
                    try:
                        # Get logits for this sequence, ensuring we don't go out of bounds
                        seq_length = min(len(tokens), logits.size(1))
                        tokens = tokens[:seq_length]  # Truncate tokens if needed
                        
                        # Get logits for valid tokens
                        seq_logits = logits[i, :seq_length, :]
                        
                        # Get log probabilities
                        log_probs = torch.nn.functional.log_softmax(seq_logits, dim=-1)
                        
                        # Create a valid range for indexing
                        valid_range = torch.arange(seq_length, device=self.model.device)
                        
                        # Ensure tokens are within vocabulary size
                        vocab_size = log_probs.size(-1)
                        valid_tokens = [min(t, vocab_size-1) for t in tokens]
                        valid_tokens = torch.tensor(valid_tokens, device=self.model.device)
                        
                        # Select log probs for the actual tokens
                        token_log_probs = log_probs[valid_range, valid_tokens]
                        
                        # Sum log probs
                        seq_log_prob = token_log_probs.sum()
                        selected_logprobs.append(seq_log_prob)
                    except Exception as e:
                        print(f"Error processing sequence {i}: {e}")
                        # Fallback to a zero log prob
                        selected_logprobs.append(torch.tensor(0.0, device=self.model.device))
                else:
                    # If all tokens were padding, use a zero log prob
                    selected_logprobs.append(torch.tensor(0.0, device=self.model.device))
            
            selected_logprobs = torch.stack(selected_logprobs)
        else:
            # Original T5 approach
            output = self.model(input_ids=encoded_prompt['input_ids'], 
                    attention_mask=encoded_prompt['attention_mask'], labels=decoded_tokens_tensor)
            logits = output.logits
            decoded_tokens = decoded_tokens_tensor.tolist()
            selected_logprobs = select_logprobs(logits, decoded_tokens, self.eos_id)
        if self.ref_model is not None:
            with torch.no_grad():
                # Check if we're using DialoGPT
                if is_dialogpt:
                    # For DialoGPT, we need to handle the KL calculation differently
                    ref_output = self.ref_model(input_ids=encoded_prompt['input_ids'],
                                            attention_mask=encoded_prompt['attention_mask'])
                    ref_logits = ref_output.logits
                    
                    # Calculate KL divergence manually for each sequence
                    kl_values = []
                    for i in range(logits.size(0)):
                        # Get sequence length (excluding padding)
                        seq_length = torch.sum(encoded_prompt['attention_mask'][i]).item()
                        
                        # Get logits for this sequence
                        seq_logits = logits[i, :seq_length, :]
                        ref_seq_logits = ref_logits[i, :seq_length, :]
                        
                        # Calculate KL divergence
                        log_probs = F.log_softmax(seq_logits, dim=-1)
                        ref_probs = F.softmax(ref_seq_logits, dim=-1)
                        seq_kl = F.kl_div(log_probs, ref_probs, reduction='batchmean')
                        kl_values.append(seq_kl)
                    
                    # Average KL divergence across sequences
                    kl = torch.stack(kl_values).mean()
                else:
                    # Original T5 approach
                    ref_logits = self.ref_model(input_ids=encoded_prompt['input_ids'], \
                        attention_mask=encoded_prompt['attention_mask'], labels=decoded_tokens_tensor).logits
                    ref_selected_logprobs = select_logprobs(ref_logits, decoded_tokens, self.eos_id)
                    kl = self.kl_loss(F.log_softmax(ref_logits, dim=-1), F.softmax(logits, dim=-1))
                    kl = torch.sum(kl, dim=-1)
                    kl_mask = decoded_tokens_tensor != self.tokenizer.pad_token_id
                    kl = reduce_mean(kl, kl_mask)
        if self.ref_model is not None:
            loss = torch.mean(rewards * (selected_logprobs + self.kl_coeff * kl))
            wandb.log({"KL term": torch.mean(rewards * self.kl_coeff * kl).item()})
            wandb.log({"KL": torch.mean(kl).item()})
            wandb.log({"Reward": torch.mean(rewards*selected_logprobs).item()})
        else:
            loss = torch.mean(rewards * selected_logprobs)
        return loss  
class RLThermostat:
    def __init__(self):
        self.temperature = 1.0
        self.threshold_enough = 0.7
        self.step = 0.1
    def log_diversity(self, diversity):
        if diversity <= self.threshold_enough:
            self.temperature += self.step
        elif self.temperature > 1.0:
            self.temperature -= self.step
        return self.temperature
class RLModelCheckpoint:
    def __init__(self, model, ckpt_every, ckpt_lookback, ckpt_file):
        self.model = model
        self.ckpt_every = ckpt_every
        self.ckpt_lookback = ckpt_lookback
        self.best_ckpt_score = None
        self.score_history = []
        self.ckpt_file = ckpt_file
        self.time_start = time.time()
        self.time_ckpt = time.time()
    def tick(self, latest_score):
        self.score_history.append(latest_score)
        is_this_best = False
        if time.time() - self.time_start > 30*60 and len(self.score_history) > self.ckpt_lookback:
            current_score = np.mean(self.score_history[-self.ckpt_lookback:])
            if time.time()-self.time_ckpt > self.ckpt_every:
                revert_ckpt = self.best_ckpt_score is not None and current_score < min(1.15*self.best_ckpt_score, 0.85*self.best_ckpt_score) 
                print("================================== CKPT "+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" =================================")
                print("[CKPT] Previous best: %.4f vs. current: %.4f" % ((0.0 if self.best_ckpt_score is None else self.best_ckpt_score), current_score))
                print("[CKPT] Am I reverting? %s" % ("yes" if revert_ckpt else "no! BEST CKPT"))
                if revert_ckpt:
                    self.model = self.model.from_pretrained(self.ckpt_file)
                self.time_ckpt = time.time()
                print("============================== END OF CKPT TIME ==============================")
            is_this_best = self.best_ckpt_score is None or current_score > self.best_ckpt_score
            if is_this_best:
                print("[CKPT] Saved new best at: %.4f" % (current_score))
                self.best_ckpt_score = current_score
                self.model.save_pretrained(self.ckpt_file)
        return is_this_best
class RLExamplePrinter:
    def __init__(self, print_every, N_samples, print_source=False, print_edit=False):
        self.print_every = print_every
        self.N_samples = N_samples
        self.print_source = print_source
        self.print_edit = print_edit
        self.time_print = time.time()
    def tick(self, paragraphs, generateds, scorer_returns, responses, steps=None):
        if time.time()-self.time_print > self.print_every:
            IDX = int(np.argmax(scorer_returns['total_scores']) / self.N_samples)
            if steps is not None:
                print("-"*30)
                print("Step:", steps)
            if self.print_source:
                print("----------- ORIGINAL -------------")
                print(paragraphs[IDX])
            print("----------- GENERATED OPTIONS ---------")
            gen_is = sorted(range(self.N_samples*IDX, self.N_samples*(IDX+1)),\
                 key=lambda gen_i: -scorer_returns["total_scores"][gen_i]) 
            for idx, gen_i in enumerate(gen_is):
                para_split = paragraphs[IDX].split("[SEP]")
                assert(len(para_split) == 2 )
                client_prompt = para_split[0]
                annotated_response = responses[IDX]
                if self.print_edit:
                    if type(annotated_response) == list:
                        annotated_response = annotated_response[0]
                    print(utils_edits.show_diff_word(annotated_response, generateds[gen_i]))
                else:
                    print(generateds[gen_i])
                print("Prompt:", client_prompt)
                print("GT Response:", annotated_response)
                print("Model Response:", generateds[gen_i])
                print("["+"; ".join(["%s: %.4f"% (k.replace("_scores", ""), scorer_returns[k][gen_i]) for k in scorer_returns if ("_score" in k or "pred_level" in k)])+"]")
                print("---")
            self.time_print = time.time()
            print("==========================================")
