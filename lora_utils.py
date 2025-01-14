from peft import LoraConfig, get_peft_model
import os
import numpy as np
from torch import nn

def find_all_linear_names(model):
    """find names for all linear layers"""
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
        
    return list(lora_module_names)

def setup_lora_config(model):
    """configure LORA and transfer to PEFT model"""
    target_modules = find_all_linear_names(model)
    print(f"target_modules: {target_modules}")
    
    config = LoraConfig(
        r=8,  # LORA rank
        lora_alpha=32,  # LORA scaling factor
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    
    # transfer model to peft model
    model = get_peft_model(model, config)
    return model

def acquire_lora_params(model):
    """acquire LORA parameters (lora_A, lora_B and scaling) from layers"""
    lora_params = {}

    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if not hasattr(module, 'weight'):
                continue
                
            # Get LoRA weights
            lora_A = module.lora_A.default.weight.detach().cpu().numpy()
            lora_B = module.lora_B.default.weight.detach().cpu().numpy()
            # Get scaling factor
            scaling = module.scaling["default"]
            # Store in dictionary
            lora_params[name] = [lora_A, lora_B, scaling]

    return lora_params

def save_lora(lora_params, npz_path):
    # TODO: test this function
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, 
        **{f"{name}_{param_name}": param 
        for name in lora_params 
        for param_name, param in zip(['loraA', 'loraB', 'scaling'], lora_params[name])})


def load_lora(npz_path):
    loaded = np.load(npz_path)
    lora_params = {}

    for key in loaded.keys():
        name, param_type = key.rsplit('_', 1)
        
        if name not in lora_params:
            lora_params[name] = [None, None, None]
            
        if param_type == 'loraA':
            lora_params[name][0] = loaded[key]
        elif param_type == 'loraB':
            lora_params[name][1] = loaded[key]
        elif param_type == 'scaling':
            lora_params[name][2] = loaded[key]
    
    return lora_params

def adjust_weights(weights, key, all_lora_params):
    key_exists = [key in lora.keys() for lora in all_lora_params]
    if all(key_exists):
        return weights
    valid_indices = [i for i, exists in enumerate(key_exists) if exists]
    sum_valid_weights = sum(weights[i] for i in valid_indices)
    
    return [
        (weights[i] / sum_valid_weights if exists else 0.0)
        for i, exists in enumerate(key_exists)
    ]