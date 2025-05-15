import torch
import torch.nn as nn
import copy

class MetaLearner(nn.Module):
    def __init__(self, base_model, n_heads, embed_dim, num_models, feed_forward_hidden, normalization, 
                batch_size, num_runs, max_output_length, device, max_steps=100):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.num_models = num_models
        self.feed_forward_hidden = feed_forward_hidden
        self.normalization = normalization
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.max_output_length = max_output_length
        self.device = device
        self.model_list = []
        
        head_weight = base_model.lm_head.weight
        zero_weight = torch.zeros_like(head_weight, requires_grad=False)
        combined_weight = torch.cat([head_weight, head_weight, head_weight], dim=1)
        self.lm_head = nn.Linear(3*768, 32128, bias=False)
        self.lm_head.weight = nn.Parameter(combined_weight)
        self.lm_head.weight.requires_grad = True

    def __call__(self, input_ids=None, attention_mask=None, labels=None, output_hidden_states=False, return_dict=True, **kwargs):
        """
        Makes MetaLearner compatible with the ReinforceCriterion by implementing 
        the expected forward interface of the language model.
        """
        
        hidden_states_list = []
        for model_idx, model in enumerate(self.model_list):
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_output_length,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

            model_hidden_states = torch.stack([step[-1] for step in generated_outputs.decoder_hidden_states])
            hidden_states_list.append(model_hidden_states)

        # 堆叠所有模型的隐藏状态 [num_models, num_steps, batch_size, 1, hidden_dim]
        all_models_hidden_states = torch.stack(hidden_states_list)
        all_models_hidden_states = all_models_hidden_states.permute(2,1,0,3,4).squeeze(3)
        all_models_hidden_states = all_models_hidden_states.contiguous().view(
            all_models_hidden_states.size(0), 
            all_models_hidden_states.size(1), 
            -1)
        logits = self.lm_head(all_models_hidden_states)
        # Create a structure similar to what the base model would return
        from dataclasses import dataclass
        
        @dataclass
        class ModelOutput:
            logits: torch.Tensor
            loss: torch.Tensor = None
        
        return ModelOutput(logits=logits)

    def config_models(self, base_model, lora_updates):
        original_params = base_model.state_dict()
        self.model_list = []
        for lora_params in lora_updates:
            new_model = copy.deepcopy(base_model)
            new_model.to(self.device)
            updated_params = copy.deepcopy(original_params)
            for key in lora_params.keys():
                start_idx = len('base_model.model.')
                k = key[start_idx:] + '.weight'
                updated_params[k] = updated_params[k] + (lora_params[key][1] @ lora_params[key][0]) * lora_params[key][2]
            new_model.load_state_dict(updated_params)
            self.model_list.append(new_model)
        
    def forward(self, gen_input, num_runs, train_batch_size, max_output_length):
        # Expand input ids and attention masks according to num_runs
        expanded_input_ids = gen_input["input_ids"].repeat_interleave(num_runs, dim=0)  # (train_batch_size * num_runs, seq_len)
        expanded_attention_mask = gen_input["attention_mask"].repeat_interleave(num_runs, dim=0)  # (train_batch_size * num_runs, seq_len)
        generated_tokens = torch.zeros(train_batch_size*num_runs, max_output_length, dtype=torch.long, device=self.device)
        # 收集所有模型的隐藏状态
        hidden_states_list = []
        for model_idx, model in enumerate(self.model_list):
            generated_outputs = model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_new_tokens=max_output_length,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            # 提取每个步骤的最后一层隐藏状态
            model_hidden_states = torch.stack([step[-1] for step in generated_outputs.decoder_hidden_states])
            hidden_states_list.append(model_hidden_states)
        
        # 堆叠所有模型的隐藏状态 [num_models, num_steps, batch_size, 1, hidden_dim]
        all_models_hidden_states = torch.stack(hidden_states_list)
        all_models_hidden_states = all_models_hidden_states.permute(2,1,0,3,4).squeeze(3)
        all_models_hidden_states = all_models_hidden_states.contiguous().view(
            all_models_hidden_states.size(0), 
            all_models_hidden_states.size(1), 
            -1)
        logits = self.lm_head(all_models_hidden_states)
        # 生成输出
        generated_tokens = torch.argmax(logits, dim=-1).squeeze(-1)
    
        return generated_tokens