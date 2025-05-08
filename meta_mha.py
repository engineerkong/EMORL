import torch
import torch.nn as nn
import copy
from mha import MOMultiHeadAttentionLayer


class MetaLearner(nn.Module):
    def __init__(self, base_model, n_heads, embed_dim, num_models, feed_forward_hidden, normalization, device, max_steps=100):
        super().__init__()
        self.base_model = base_model
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.num_models = num_models
        self.feed_forward_hidden = feed_forward_hidden
        self.normalization = normalization
        self.device = device
        self.model_list = []

        # 使用分层模型融合替代简单的MHA
        self.hierarchical_fusion = self.HierarchicalModelFusion(
            num_heads=self.n_heads, 
            hidden_dim=self.embed_dim, 
            num_models=self.num_models,  # 假设模型数量等于注意力头数
            feed_forward_hidden=self.feed_forward_hidden, 
            normalization=self.normalization,
            max_steps=max_steps
        )
        
        self.lm_head = self.base_model.lm_head

    # 嵌入HierarchicalModelFusion类作为内部类
    class HierarchicalModelFusion(nn.Module):
        def __init__(self, num_heads, hidden_dim, num_models, 
                     feed_forward_hidden=512, normalization='batch', max_steps=100):
            super().__init__()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 第一层: 每个步骤内的模型融合
            self.model_fusion = MOMultiHeadAttentionLayer(
                n_heads=num_models,
                embed_dim=hidden_dim,
                feed_forward_hidden=feed_forward_hidden,
                normalization=normalization
            )
            
            # 第二层: 跨步骤的信息融合 - 使用单头注意力
            self.step_fusion = MOMultiHeadAttentionLayer(
                n_heads=1,
                embed_dim=hidden_dim,
                feed_forward_hidden=feed_forward_hidden,
                normalization=normalization
            )
            
            # 位置编码
            self.position_encoding = nn.Parameter(torch.zeros(1, max_steps, hidden_dim)).to(device)
            nn.init.normal_(self.position_encoding, mean=0, std=0.02)
            
            # 输出投影层
            self.output_projection = nn.Linear(hidden_dim, hidden_dim).to(device)
            
        def forward(self, hidden_states_combined):
            """
            输入: [num_steps, num_models, batch_size, 1, hidden_dim]
            输出: [num_steps, batch_size, 1, hidden_dim]
            """
            num_steps, num_models, batch_size, seq_len, hidden_dim = hidden_states_combined.size()
            
            # 第一阶段: 按步骤融合模型
            per_step_fused = []
            for step_idx in range(num_steps):
                # 提取当前步骤的所有模型隐藏状态: [num_models, batch_size, seq_len, hidden_dim]
                step_models = hidden_states_combined[step_idx]
                
                # 准备输入形状 - 确保维度正确
                # MOMultiHeadAttention期望输入维度为[n_heads, batch_size, graph_size, input_dim]
                # 注意: MOMultiHeadAttention期望输入维度为[n_heads, batch_size, graph_size, input_dim]
                step_models_prepared = step_models.contiguous()
                print(f"step_models_prepared: {step_models_prepared}")
                # 应用MHA融合
                fused = self.model_fusion(step_models_prepared)
                print(f"fused: {fused}")
                # 处理输出，得到 [batch_size, 1, hidden_dim]
                fused = fused.squeeze(0).mean(dim=1, keepdim=True)
                per_step_fused.append(fused)
            
            # 合并所有步骤的融合结果: [batch_size, num_steps, hidden_dim]
            sequence_repr = torch.cat(per_step_fused, dim=1)
            
            # 添加位置编码
            position_enc = self.position_encoding[:, :num_steps, :]
            sequence_repr = sequence_repr + position_enc
            
            # 准备第二层输入: [1, batch_size, num_steps, hidden_dim]
            sequence_repr_prepared = sequence_repr.unsqueeze(0)
            
            # 第二阶段: 跨步骤融合
            enhanced_sequence = self.step_fusion(sequence_repr_prepared)
            print(f"enhanced_sequence: {enhanced_sequence}")
            # 处理输出: [batch_size, num_steps, hidden_dim]
            enhanced_sequence = enhanced_sequence.squeeze(0)
            
            # 应用输出投影
            enhanced_sequence = self.output_projection(enhanced_sequence)
            
            # 重塑为 [num_steps, batch_size, seq_len, hidden_dim]
            output = enhanced_sequence.transpose(0, 1).unsqueeze(2)
            
            return output

    def __call__(self, input_ids=None, attention_mask=None, labels=None, output_hidden_states=False, return_dict=True, **kwargs):
        """
        Makes MetaLearner compatible with the ReinforceCriterion by implementing 
        the expected forward interface of the language model.
        """
        
        hidden_states_list = []
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
            # hidden_states_combined[model_idx] = outputs.decoder_hidden_states[-1]
            hidden_states_list.append(outputs.decoder_hidden_states[-1])
        hidden_states_combined = torch.stack(hidden_states_list)
        # 转置为层次融合所需的形状: [1, num_models, batch_size, 1, hidden_dim]
        hidden_states_combined = hidden_states_combined.unsqueeze(0)
        # 应用层次融合
        mha_states = self.hierarchical_fusion(hidden_states_combined)
        # 压缩num_steps维度(此处为1)，得到[batch_size, 1, hidden_dim]
        mha_states = mha_states.squeeze(0)
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
        
        first_model = self.model_list[0]
        first_outputs = first_model.generate(
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            max_length=max_output_length,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        
        # 正确提取生成的tokens
        generated_tokens = first_outputs.sequences

        # 收集所有模型的隐藏状态
        hidden_states_list = []
        for model_idx, model in enumerate(self.model_list):
            generated_outputs = model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_length=max_output_length,
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
        generated_tokens = torch.argmax(logits, dim=-1)
        generated_tokens = generated_tokens.permute(1, 0, 2).squeeze(-1)
    

        return generated_tokens