import torch
import torch.nn as nn
import copy
from mha import MOMultiHeadAttentionLayer

class MetaLearner(nn.Module):
    def __init__(self, base_model, n_heads, embed_dim, num_models, feed_forward_hidden, normalization,
                batch_size, num_runs, max_output_length, device, max_steps=100):
        super().__init__()
        self.base_model = base_model
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

        # 使用分层模型融合替代简单的MHA
        self.hierarchical_fusion = self.HierarchicalModelFusion(
            hidden_dim=self.embed_dim, 
            num_models=self.num_models,  # 假设模型数量等于注意力头数
            num_heads=self.n_heads, 
            num_steps=max_steps,
            device=self.device
        )
        
        self.lm_head = self.base_model.lm_head

    class HierarchicalModelFusion(nn.Module):
        def __init__(self, hidden_dim, num_models, num_heads=8, num_steps=64, device='cuda'):
            super().__init__()
            # 第一层: 每个步骤内的模型融合
            self.model_fusion = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_models,
                batch_first=True
            ).to(device)
            
            # 第二层: 跨步骤的信息融合
            self.step_fusion = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True
            ).to(device)
            
            # 位置编码(用于第二层)
            self.position_encoding = nn.Parameter(
                torch.zeros(1, num_steps, hidden_dim)  # 假设最大步骤数为100
            ).to(device)
            nn.init.normal_(self.position_encoding, mean=0, std=0.02)
            
        def forward(self, hidden_states_combined):
            """
            输入: [num_steps, num_models, batch_size, 1, hidden_dim]
            输出: [num_steps, batch_size, 1, hidden_dim]
            """
            num_steps, num_models, batch_size, _, hidden_dim = hidden_states_combined.size()
            # 第一阶段: 按步骤融合模型
            per_step_fused = []
            for step_idx in range(num_steps):
                # [num_models, batch_size, hidden_dim]
                step_models = hidden_states_combined[step_idx, :, :, 0, :]
                # 转置为[batch_size, num_models, hidden_dim]
                step_models = step_models.permute(1, 0, 2)
                # 模型内融合
                fused, _ = self.model_fusion(
                    query=step_models,
                    key=step_models,
                    value=step_models
                )
                # [batch_size, 1, hidden_dim]
                fused = fused.mean(dim=1, keepdim=True)
                per_step_fused.append(fused)

            # [batch_size, num_steps, hidden_dim]
            sequence_repr = torch.cat(per_step_fused, dim=1)
            
            # 添加位置编码
            sequence_repr = sequence_repr + self.position_encoding[:, :num_steps, :]
            # 第二阶段: 跨步骤融合
            output, _ = self.step_fusion(
                query=sequence_repr,
                key=sequence_repr,
                value=sequence_repr
            )

            return output

    def __call__(self, input_ids=None, attention_mask=None, labels=None, output_hidden_states=False, return_dict=True, **kwargs):
        """
        Makes MetaLearner compatible with the ReinforceCriterion by implementing 
        the expected forward interface of the language model.
        """
        # 收集所有模型的隐藏状态
        hidden_states_list = []
        for model_idx, model in enumerate(self.model_list):
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_output_length,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            # 提取每个步骤的最后一层隐藏状态

            model_hidden_states = torch.stack([step[-1] for step in generated_outputs.decoder_hidden_states])
            hidden_states_list.append(model_hidden_states)

        # 堆叠所有模型的隐藏状态 [num_models, num_steps, batch_size, 1, hidden_dim]
        all_models_hidden_states = torch.stack(hidden_states_list)
        # 转置为层次融合所需的形状 [num_steps, num_models, batch_size, 1, hidden_dim]
        hidden_states_combined = all_models_hidden_states.permute(1, 0, 2, 3, 4)
        # 应用层次融合
        fused_states = self.hierarchical_fusion(hidden_states_combined)
        logits = self.lm_head(fused_states)
        # Create a structure similar to what the base model would return
        from dataclasses import dataclass
        
        @dataclass
        class ModelOutput:
            logits: torch.Tensor
            loss: torch.Tensor = None
        
        return ModelOutput(logits=logits)

    def config_models(self, lora_updates):
        original_params = self.base_model.state_dict()
        assert len(lora_updates) == self.num_models, "lora_updates should have the same number of heads as the model"
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
        # 转置为层次融合所需的形状 [num_steps, num_models, batch_size, 1, hidden_dim]
        hidden_states_combined = all_models_hidden_states.permute(1, 0, 2, 3, 4)
        # 应用层次融合
        fused_states = self.hierarchical_fusion(hidden_states_combined)
        logits = self.lm_head(fused_states)
        # 生成输出
        generated_tokens = torch.argmax(logits, dim=-1).squeeze(-1)
    
        return generated_tokens