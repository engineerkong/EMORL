import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载两个模型和分词器
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model1 = T5ForConditionalGeneration.from_pretrained("t5-small")
model2 = T5ForConditionalGeneration.from_pretrained("t5-small")

# 输入文本
input_text = "translate English to French: Hello world."
inputs = tokenizer(input_text, return_tensors="pt")

# 初始化解码器输入
decoder_input_ids = torch.tensor([[model1.config.decoder_start_token_id]])  # <pad> 或 <s> 起始标记

# 最大生成长度
max_length = 20
generated_tokens = []

# 模拟自回归生成
for _ in range(max_length):
    # 前向传播，获取两个模型的输出和隐藏状态
    outputs1 = model1(
        input_ids=inputs["input_ids"],
        decoder_input_ids=decoder_input_ids,
        output_hidden_states=True,  # 返回隐藏状态
        return_dict=True,
    )
    outputs2 = model2(
        input_ids=inputs["input_ids"],
        decoder_input_ids=decoder_input_ids,
        output_hidden_states=True,  # 返回隐藏状态
        return_dict=True,
    )

    # 获取两个模型解码器的最后一层隐藏状态
    decoder_hidden_states1 = outputs1.decoder_hidden_states[-1]  # shape: (batch_size, seq_len, hidden_dim)
    decoder_hidden_states2 = outputs2.decoder_hidden_states[-1]  # shape: (batch_size, seq_len, hidden_dim)

    # 自定义组合隐藏状态（示例：加权平均）
    combined_hidden_states = (decoder_hidden_states1 + decoder_hidden_states2) / 2

    # 使用组合后的隐藏状态，重新预测 logits
    logits = model1.lm_head(combined_hidden_states)  # shape: (batch_size, seq_len, vocab_size)

    # 获取下一个 token
    next_token = torch.argmax(logits[:, -1, :], dim=-1)  # shape: (batch_size,)
    generated_tokens.append(next_token.item())

    # 更新解码器输入
    decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(-1)], dim=-1)

    # 检查是否生成了结束标记
    if next_token == model1.config.eos_token_id:
        break

# 解码生成的序列
output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print("Generated Text:", output_text)
