import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalModelFusion(nn.Module):
    def __init__(self, hidden_dim, num_models, num_heads=8, num_steps=64):
        super().__init__()
        # 第一层: 每个步骤内的模型融合
        self.model_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 第二层: 跨步骤的信息融合
        self.step_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 位置编码(用于第二层)
        self.position_encoding = nn.Parameter(
            torch.zeros(1, num_steps, hidden_dim)  # 假设最大步骤数为100
        )
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
        enhanced_sequence, _ = self.step_fusion(
            query=sequence_repr,
            key=sequence_repr,
            value=sequence_repr
        )
        
        # 重塑为[num_steps, batch_size, 1, hidden_dim]
        output = enhanced_sequence.transpose(0, 1).unsqueeze(2)
        
        return output