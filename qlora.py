import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch.nn as nn
import numpy as np
from peft.tuners.lora import LoraLayer
from torch.utils.data import TensorDataset, DataLoader

def setup_model_for_qlora(device="cuda:0"):
    """设置量化配置和T5模型"""
    # 配置4位量化参数
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载T5模型和分词器
    tokenizer = T5Tokenizer.from_pretrained("/home/ubuntu/test/google-t5-t5-base")
    model = T5ForConditionalGeneration.from_pretrained(
        "/home/ubuntu/test/google-t5-t5-base",
        quantization_config=bnb_config,
        device_map=device
    )

    # if tokenizer._pad_token is None:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
    #         tokenizer=tokenizer,
    #         model=model,
    #     )

    # 准备模型进行QLORA训练
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def setup_lora_config(model):
    """配置LORA参数，特别针对T5模型架构"""
    # T5特定的目标模块
    target_modules = find_all_linear_names(model)
    print(f"target_modules:{target_modules}")
    
    config = LoraConfig(
        r=8,  # LORA矩阵的秩
        lora_alpha=32,  # LORA的缩放因子
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"  # 设置为序列到序列任务
    )
    
    # 将模型转换为PEFT模型
    model = get_peft_model(model, config)

    # for name, module in model.named_modules():
    #     if isinstance(module, LoraLayer):
    #         module = module.to(torch.bfloat16)
    #     if 'norm' in name:
    #         module = module.to(torch.float32)
    #     if 'lm_head' in name or 'embed_tokens' in name:
    #         if hasattr(module, 'weight'):
    #             if module.weight.dtype == torch.float32:
    #                 module = module.to(torch.bfloat16)
    return model

def train_model(model, train_dataloader, device="cuda", num_epochs=3):
    """训练T5模型"""
    model.config.use_cache = False
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            # 解包batch数据
            input_ids, attention_mask, labels = batch
            
            # 将数据移至设备
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
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

def extract_lora_matrices(model):
    """提取LORA的A和B矩阵"""
    lora_matrices = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # 获取A矩阵和B矩阵
            lora_A = module.lora_A.default.weight.detach().cpu().numpy()  # [r, in_dim]
            lora_B = module.lora_B.default.weight.detach().cpu().numpy()  # [out_dim, r]
            
            # 计算完整的LORA矩阵
            lora_matrix = np.matmul(lora_B, lora_A)  # [out_dim, in_dim]
            
            lora_matrices[name] = {
                'A': lora_A,
                'B': lora_B,
                'combined': lora_matrix
            }
    
    return lora_matrices

def save_lora_matrices(lora_matrices, save_path):
    """保存LORA矩阵"""
    np.savez(save_path, **{
        f"{name}_{matrix_type}": matrix 
        for name, matrices in lora_matrices.items()
        for matrix_type, matrix in matrices.items()
    })

def prepare_t5_data_and_dataloader(tokenizer, texts, targets, batch_size=2, max_length=512):
    """准备T5的训练数据和DataLoader"""
    # 对输入文本进行编码
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 对目标文本进行编码
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).input_ids
    
    # 创建Dataset
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(
        inputs.input_ids,
        inputs.attention_mask,
        labels
    )
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return dataloader

def dequantize_weight(module):
    """
    将4bit量化的权重反量化为原始大小
    Args:
        module: 包含量化权重的模块
    Returns:
        numpy.ndarray: 反量化后的权重矩阵
    """
    if isinstance(module, bnb.nn.Linear4bit):
        # 获取量化状态
        quant_state = module.weight.quant_state
        
        # 获取原始形状信息
        out_features = module.out_features
        in_features = module.in_features
        
        # 使用bnb的内部方法进行反量化
        w_tensor = bnb.functional.dequantize_4bit(
            module.weight.data,
            quant_state[0],  # absmax
            quant_state[1],  # shape
            quant_state[2]   # quant_type
        )
        
        # 转换为numpy并重塑为正确的形状
        w_array = w_tensor.detach().cpu().numpy()
        w_array = w_array.reshape(out_features, in_features)
        
        return w_array
    else:
        raise ValueError("Module is not a 4-bit quantized layer")
        
def verify_lora_update(model, original_weights=None):
    """验证LoRA更新是否正确"""
    verification_results = {}
    
    # 遍历所有模块
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # 直接从当前模块获取原始权重
            if not hasattr(module, 'weight'):
                print(f"Module {name} has LoRA but no weight attribute")
                continue
                
            # 获取当前权重
            current_weight = module.weight.detach().cpu().numpy()
            current_weight = dequantize_weight(module)

            print(f"weight_shape:{current_weight.shape}")
            # 获取LoRA权重
            lora_A = module.lora_A.default.weight.detach().cpu().numpy()  # [r, in_dim]
            lora_B = module.lora_B.default.weight.detach().cpu().numpy()  # [out_dim, r]
            
            # 获取缩放因子
            scaling = module.scaling["default"]

            # 计算LoRA更新
            lora_update = np.matmul(lora_B, lora_A) * scaling
            
            # 如果提供了原始权重，使用它；否则使用当前权重减去LoRA更新
            if original_weights is not None and name in original_weights:
                original_weight = original_weights[name]
            else:
                # 从当前权重反推原始权重
                original_weight = current_weight - lora_update
            
            # 计算预期的更新后权重
            expected_weight = original_weight + lora_update
            
            # 计算差异
            diff = np.abs(current_weight - expected_weight)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            verification_results[name] = {
                'max_difference': max_diff,
                'mean_difference': mean_diff,
                'original_shape': original_weight.shape,
                'lora_A_shape': lora_A.shape,
                'lora_B_shape': lora_B.shape,
                'scaling_factor': scaling,
                'is_close': np.allclose(current_weight, expected_weight, rtol=1e-5, atol=1e-5)
            }
            
            print(f"\nVerifying module: {name}")
            print(f"Original weight shape: {original_weight.shape}")
            print(f"LoRA A shape: {lora_A.shape}")
            print(f"LoRA B shape: {lora_B.shape}")
            print(f"Current weight shape: {current_weight.shape}")
            print(f"Maximum difference: {max_diff:.2e}")
            print(f"Mean difference: {mean_difference:.2e}")
    
    return verification_results

def inspect_layer_attributes(model):
    """检查具有weight的层和具有LoRA的层的所有属性"""
    
    # 用于存储两种类型层的属性集合
    weight_layer_attrs = set()
    lora_layer_attrs = set()
    
    print("\n=== Layers with Weight ===")
    i = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and not isinstance(module, LoraLayer):
            i += 1
            # 收集该模块的所有属性
            attrs = set(dir(module))
            weight_layer_attrs.update(attrs)
            # if i <= 3:  # 只打印前3个层的详细信息作为样本
            #     print(f"\nLayer {i}: {name}")
            #     print("Attributes:")
            #     for attr in sorted(attrs):
            #         # 跳过内置属性
            #         if not attr.startswith('__'):
            #             attr_value = getattr(module, attr)
            #             # 如果是张量，打印其形状
            #             if isinstance(attr_value, torch.Tensor):
            #                 print(f"  {attr}: Tensor of shape {attr_value.shape}")
            #             # 如果是简单类型，直接打印值
            #             elif isinstance(attr_value, (int, float, str, bool)):
            #                 print(f"  {attr}: {attr_value}")
            #             # 对于其他类型，只打印类型名
            #             else:
            #                 print(f"  {attr}: {type(attr_value).__name__}")
    print(f"\nTotal layers with weight: {i}")
    
    print("\n=== LoRA Layers ===")
    j = 0
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            j += 1
            # 收集该模块的所有属性
            attrs = set(dir(module))
            lora_layer_attrs.update(attrs)
            # if j <= 3:  # 只打印前3个层的详细信息作为样本
            #     print(f"\nLayer {j}: {name}")
            #     print("Attributes:")
            #     for attr in sorted(attrs):
            #         # 跳过内置属性
            #         if not attr.startswith('__'):
            #             attr_value = getattr(module, attr)
            #             # 如果是张量，打印其形状
            #             if isinstance(attr_value, torch.Tensor):
            #                 print(f"  {attr}: Tensor of shape {attr_value.shape}")
            #             # 如果是简单类型，直接打印值
            #             elif isinstance(attr_value, (int, float, str, bool)):
            #                 print(f"  {attr}: {attr_value}")
            #             # 对于其他类型，只打印类型名
            #             else:
            #                 print(f"  {attr}: {type(attr_value).__name__}")
    print(f"\nTotal LoRA layers: {j}")
    
    # # 比较两种层的属性差异
    # print("\n=== Attribute Comparison ===")
    # print("\nAttributes unique to weight layers:")
    # weight_only = weight_layer_attrs - lora_layer_attrs
    # for attr in sorted(weight_only):
    #     if not attr.startswith('__'):
    #         print(f"  {attr}")
            
    # print("\nAttributes unique to LoRA layers:")
    # lora_only = lora_layer_attrs - weight_layer_attrs
    # for attr in sorted(lora_only):
    #     if not attr.startswith('__'):
    #         print(f"  {attr}")
            
    # print("\nCommon attributes:")
    # common = weight_layer_attrs & lora_layer_attrs
    # for attr in sorted(common):
    #     if not attr.startswith('__'):
    #         print(f"  {attr}")

def inspect_model_modules(model):
    """检查模型中的所有模块类型和结构
    
    Args:
        model: PyTorch模型
    """
    # 用于存储已发现的模块类型
    module_types = set()
    
    # 用于统计每种类型的数量
    type_count = {}
    
    print("\n=== Model Module Inspection ===")
    
    for name, module in model.named_modules():
        # 获取模块类型
        module_type = type(module).__name__
        
        # 添加到集合中
        module_types.add(module_type)
        
        # 统计数量
        type_count[module_type] = type_count.get(module_type, 0) + 1
        
        # 打印详细信息
        # print(f"\nModule: {name}")
        # print(f"Type: {module_type}")
        
        # 打印模块的主要属性
        attributes = []
        if hasattr(module, 'weight'):
            weight_shape = tuple(module.weight.shape) if hasattr(module.weight, 'shape') else 'N/A'
            attributes.append(f"weight shape: {weight_shape}")
        if hasattr(module, 'bias') and module.bias is not None:
            bias_shape = tuple(module.bias.shape) if hasattr(module.bias, 'shape') else 'N/A'
            attributes.append(f"bias shape: {bias_shape}")
        if hasattr(module, 'in_features'):
            attributes.append(f"in_features: {module.in_features}")
        if hasattr(module, 'out_features'):
            attributes.append(f"out_features: {module.out_features}")
        if hasattr(module, 'lora_A'):
            attributes.append("has lora_A")
        if hasattr(module, 'lora_B'):
            attributes.append("has lora_B")
            
        if attributes:
            # print("Attributes:", ", ".join(attributes))
            pass
    
    # 打印总结信息
    print("\n=== Summary ===")
    print(f"Total unique module types: {len(module_types)}")
    print("\nModule types count:")
    for module_type, count in sorted(type_count.items()):
        print(f"{module_type}: {count}")
    
    return module_types, type_count

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 设置模型
    model, tokenizer = setup_model_for_qlora(device)
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
    
    # 使用新的函数创建dataloader
    train_dataloader = prepare_t5_data_and_dataloader(
        tokenizer,
        texts,
        targets,
        batch_size=2
    )
    
    # for batch in train_dataloader:
    #     input_ids, attention_mask, labels = batch
    #     print(f"Input shape: {input_ids.shape}")
    #     print(f"Attention mask shape: {attention_mask.shape}")
    #     print(f"Labels shape: {labels.shape}")
    #     break  # 只打印第一个batch

    # 存储原始权重
    original_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            original_weights[name] = module.weight.detach().cpu().numpy()

    # 训练模型
    # inspect_model_modules(model)
    inspect_layer_attributes(model)
    train_model(model, train_dataloader, device=device)
    # inspect_model_modules(model)
    inspect_layer_attributes(model)

    # 提取和检查LoRA矩阵
    # lora_matrices = extract_lora_matrices(model)
    # for name, matrices in lora_matrices.items():
    #     print(f"\nModule: {name}")
    #     print(f"A matrix shape: {matrices['A'].shape}")
    #     print(f"A matrix mean: {np.mean(matrices['A']):.6f}")
    #     print(f"B matrix shape: {matrices['B'].shape}")
    #     print(f"B matrix mean: {np.mean(matrices['B']):.6f}")
    
    # 验证LoRA更新
    verification_results = verify_lora_update(model, original_weights)
    print(verification_results)
    
    # 可以添加具体的阈值检查
    all_verified = all(result['is_close'] for result in verification_results.values())
    print(f"\nOverall verification {'passed' if all_verified else 'failed'}")

if __name__ == "__main__":
    main()