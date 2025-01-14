from additional_utils import *
from lora_utils import *
import torch

# load original model and lora params
model, tokenizer = get_model(
    "models/google-t5-t5-base",  
    max_seq_length=128,
    max_output_length=64,
    lora=True
)
original_params = model.state_dict()
lora_params = load_lora('/home/ubuntu/DynaDRL/results/lora_fluency.npz')

for key, value in lora_params.items():
    layer_name = '.'.join(key.split('.')[2:])
    original_params[layer_name+'.weight'] = original_params[layer_name+'.weight'] + (value[1] @ value[0]) * value[2]
updated_params = original_params
model.load_state_dict(updated_params)

def check_lora_trainable(model):
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params.append(f"{name}: requires_grad = {param.requires_grad}")
    return lora_params