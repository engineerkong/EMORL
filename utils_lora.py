from peft import LoraConfig, get_peft_model
import os
import numpy as np
from torch import nn
import torch

def find_all_linear_names(model):
    """
    Function for finding names for all linear layers, not update lm_head for keeping the 
    stability of vocab and avoiding to influence the basic generation ability
    """
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split('.')
            # For DialoGPT, we need to target specific modules
            if 'microsoft/DialoGPT' in model.config._name_or_path:
                # DialoGPT uses GPT-2 architecture
                if any(key in name for key in ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']):
                    lora_module_names.add(name)
            else:
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    # If no modules were found, use default GPT-2 target modules
    if len(lora_module_names) == 0 and 'microsoft/DialoGPT' in model.config._name_or_path:
        lora_module_names = ['c_attn', 'c_proj', 'c_fc']
    
    return list(lora_module_names)

def setup_lora_config(model):
    """
    Function for configuring LORA and transfering to PEFT model
    """
    target_modules = find_all_linear_names(model)
    print(f"target_modules: {target_modules}")
    config = LoraConfig(
        r=8,  # LORA rank
        lora_alpha=32,  # LORA scaling factor
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"  # Changed from SEQ_2_SEQ_LM to CAUSAL_LM for DialoGPT
    )
    model = get_peft_model(model, config)
    return model

def acquire_lora_params(model):
    """
    Function for acquiring LORA parameters (lora_A, lora_B and scaling) from layers
    """
    lora_params = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if not hasattr(module, 'weight'):
                continue
            lora_A = module.lora_A.default.weight.detach().cpu().numpy()
            lora_B = module.lora_B.default.weight.detach().cpu().numpy()
            scaling = module.scaling["default"]
            lora_params[name] = [lora_A, lora_B, scaling]
    return lora_params

def save_lora(lora_params, npz_path):
    """
    Function for saving LORA parameters in .npz file
    """
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, 
        **{f"{name}_{param_name}": param 
        for name in lora_params 
        for param_name, param in zip(['loraA', 'loraB', 'scaling'], lora_params[name])})

def load_lora(npz_path):
    """
    Function for loading LORA parameters from .npz file and converting to CUDA tensors
    """
    loaded = np.load(npz_path)
    lora_params = {}
    for key in loaded.keys():
        name, param_type = key.rsplit('_', 1)
        if name not in lora_params:
            lora_params[name] = [None, None, None]
        tensor_value = torch.from_numpy(loaded[key]).cuda()
        if param_type == 'loraA':
            lora_params[name][0] = tensor_value
        elif param_type == 'loraB':
            lora_params[name][1] = tensor_value
        elif param_type == 'scaling':
            lora_params[name][2] = tensor_value
    lora_params = {k: tuple(v) for k, v in lora_params.items()}
    return lora_params

def adjust_weights(weights, key, all_lora_params):
    """
    Function for adjusting weights when some updates not exist for specific layers (Unused)
    """
    key_exists = [key in lora.keys() for lora in all_lora_params]
    if all(key_exists):
        return weights
    valid_indices = [i for i, exists in enumerate(key_exists) if exists]
    sum_valid_weights = sum(weights[i] for i in valid_indices)
    return [
        (weights[i] / sum_valid_weights if exists else 0.0)
        for i, exists in enumerate(key_exists)
    ]
