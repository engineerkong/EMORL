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
        for param_name, param in zip(['lora_A', 'lora_B', 'scaling'], lora_params[name])})


def load_lora(npz_path):
    loaded = np.load(npz_path)
    lora_params = {}
    for name in set(k.split('_lora')[0] for k in loaded.keys()):
        lora_params[name] = [
            loaded[f"{name}_lora_A"],
            loaded[f"{name}_lora_B"],
            loaded[f"{name}_scaling"]
        ]
    return lora_params