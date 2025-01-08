import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import numpy as np
from peft.tuners.lora import LoraLayer
from torch.utils.data import TensorDataset, DataLoader

def setup_model(device="cuda:0"):
    """设置T5模型"""
    # 加载T5模型和分词器
    tokenizer = T5Tokenizer.from_pretrained("/home/ubuntu/test/google-t5-t5-base")
    model = T5ForConditionalGeneration.from_pretrained(
        "/home/ubuntu/test/google-t5-t5-base",
        device_map=device
    )
    return model, tokenizer

def find_all_linear_names(model):
    """查找所有线性层的名称"""
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def setup_lora_config(model):
    """配置LORA参数"""
    target_modules = find_all_linear_names(model)
    print(f"target_modules: {target_modules}")
    
    config = LoraConfig(
        r=8,  # LoRA矩阵的秩
        lora_alpha=32,  # LoRA的缩放因子
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    
    # 将模型转换为PEFT模型
    model = get_peft_model(model, config)
    return model

def train_model(model, train_dataloader, device="cuda", num_epochs=3):
    """训练模型"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"Current batch loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

def prepare_t5_data_and_dataloader(tokenizer, texts, targets, batch_size=2, max_length=512):
    """准备T5的训练数据和DataLoader"""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).input_ids
    
    dataset = TensorDataset(
        inputs.input_ids,
        inputs.attention_mask,
        labels
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def verify_lora_update(model, original_weights=None):
    """验证LoRA更新"""
    verification_results = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if not hasattr(module, 'weight'):
                continue
                
            # 获取当前权重
            current_weight = module.weight.detach().cpu().numpy()

            # 获取LoRA权重
            lora_A = module.lora_A.default.weight.detach().cpu().numpy()
            lora_B = module.lora_B.default.weight.detach().cpu().numpy()
            
            # 获取缩放因子
            scaling = module.scaling["default"]
            
            # 计算LoRA更新
            lora_update = np.matmul(lora_B, lora_A) * scaling
            
            # 使用原始权重或从当前权重反推
            if original_weights is not None and name in original_weights:

                original_weight = original_weights[name]
            else:
                original_weight = current_weight - lora_update
            
            # 计算预期的更新后权重
            expected_weight = original_weight + lora_update
            
            # 计算差异
            diff = np.abs(current_weight - expected_weight)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            # 计算权重大小统计
            current_weight_abs = np.abs(current_weight)
            current_weight_mean = np.mean(current_weight_abs)
            current_weight_max = np.max(current_weight_abs)
            
            original_weight_abs = np.abs(original_weight)
            original_weight_mean = np.mean(original_weight_abs)
            original_weight_max = np.max(original_weight_abs)
            
            lora_update_abs = np.abs(lora_update)
            lora_update_mean = np.mean(lora_update_abs)
            lora_update_max = np.max(lora_update_abs)

            verification_results[name] = {
                'max_difference': max_diff,
                'mean_difference': mean_diff,
                'original_shape': original_weight.shape,
                'lora_A_shape': lora_A.shape,
                'lora_B_shape': lora_B.shape,
                'scaling_factor': scaling,
                'is_close': np.allclose(current_weight, expected_weight, rtol=1e-4, atol=1e-4),
                # 添加权重大小统计
                'current_weight_mean': current_weight_mean,
                'current_weight_max': current_weight_max,
                'original_weight_mean': original_weight_mean,
                'original_weight_max': original_weight_max,
                'lora_update_mean': lora_update_mean,
                'lora_update_max': lora_update_max
            }
            
            print(f"\nVerifying module: {name}")
            print(f"Original weight shape: {original_weight.shape}")
            print(f"LoRA A shape: {lora_A.shape}")
            print(f"LoRA B shape: {lora_B.shape}")
            print(f"Current weight shape: {current_weight.shape}")
            print(f"Maximum difference: {max_diff:.2e}")
            print(f"Mean difference: {mean_diff:.2e}")
            print("\nWeight statistics:")
            print(f"Current weight - Mean: {current_weight_mean:.2e}, Max: {current_weight_max:.2e}")
            print(f"Original weight - Mean: {original_weight_mean:.2e}, Max: {original_weight_max:.2e}")
            print(f"LoRA update - Mean: {lora_update_mean:.2e}, Max: {lora_update_max:.2e}")
    
    return verification_results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 设置模型
    model, tokenizer = setup_model(device)
    model = setup_lora_config(model)
    
    # 准备示例数据
    texts = [
        "translate English to German: The house is wonderful",
        "translate English to German: I love this book",
        "translate English to German: The weather is nice today",
        "translate English to German: She plays the piano"
    ]
    targets = [
        "Das Haus ist wunderbar",
        "Ich liebe dieses Buch",
        "Das Wetter ist heute schön",
        "Sie spielt Klavier"
    ]
    
    # 创建dataloader
    train_dataloader = prepare_t5_data_and_dataloader(
        tokenizer,
        texts,
        targets,
        batch_size=2
    )
    
    # 存储原始权重
    original_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            original_weights[name] = module.weight.detach().cpu().numpy()
    
    # 训练模型
    train_model(model, train_dataloader, device=device)
    
    # 验证LoRA更新
    verification_results = verify_lora_update(model, original_weights)
    
    # 打印验证结果
    all_verified = all(result['is_close'] for result in verification_results.values())
    print(f"\nOverall verification {'passed' if all_verified else 'failed'}")

if __name__ == "__main__":
    main()