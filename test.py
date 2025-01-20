from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn.functional as F

# 加载模型和tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# 输入文本
input_text = "translate English to German: The house is wonderful."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 模拟多个模型的输出（这里我们使用同一个模型三次）
num_models = 3
weights = [0.4, 0.3, 0.3]  # 模型权重
all_outputs = []

for _ in range(num_models):
    outputs = model.generate(
        input_ids,
        max_length=50,
        return_dict_in_generate=True,
        output_scores=True,
    )
    all_outputs.append(outputs)

# 融合logits并生成文本
merged_scores = []
for step in range(len(all_outputs[0].scores)):
    step_scores = []
    for model_idx, outputs in enumerate(all_outputs):
        logits = outputs.scores[step]
        probs = F.softmax(logits, dim=-1)
        weighted_probs = probs * weights[model_idx]
        step_scores.append(weighted_probs)
    
    merged_prob = sum(step_scores)
    merged_logit = torch.log(merged_prob + 1e-10)
    merged_scores.append(merged_logit)

# 使用merged_scores生成文本
generated_tokens = []
current_tokens = input_ids

for step_logits in merged_scores:
    # 获取最可能的token
    next_token = torch.argmax(step_logits, dim=-1)
    generated_tokens.append(next_token.item())

# 解码生成的tokens
merged_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print("\n使用融合logits生成的文本:")
print(merged_output)

# 对比：直接使用单个模型生成的文本
original_output = tokenizer.decode(all_outputs[0].sequences[0], skip_special_tokens=True)
print("\n原始单个模型生成的文本:")
print(original_output)

# 打印每一步的top tokens（用于分析）
print("\n每一步的top 3 tokens:")
for step, logits in enumerate(merged_scores[:5]):  # 只显示前5步
    top_scores, top_tokens = torch.topk(logits[0], k=3)
    top_tokens = tokenizer.convert_ids_to_tokens(top_tokens)
    
    print(f"\nStep {step + 1}:")
    for token, score in zip(top_tokens, top_scores):
        print(f"  {token}: {score:.2f}")