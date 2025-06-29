"""Phoebe: A Flower Baseline."""

from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "conf/base.yaml"))
cfg = OmegaConf.load(base_path)


# myfl/models.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model,get_peft_model_state_dict, set_peft_model_state_dict

def get_weights(model):
    """Extract LoRA adapter weights only."""
    lora_state = get_peft_model_state_dict(model)
    return [v.cpu().numpy() for v in lora_state.values()]

def set_weights(model, parameters):
    """Set LoRA adapter weights only."""
    keys = list(get_peft_model_state_dict(model).keys())
    new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    set_peft_model_state_dict(model, new_state)

def get_llama_with_lora(
    base_model_name_or_path: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
) -> nn.Module:
    """
    使用 HuggingFace Transformers 加载基础 Llama 模型，并在指定模块上添加 LoRA。
    返回一个已封装好的带 LoRA adapter 的 PyTorch Model，待客户端直接调用 train() 即可。
    """
    # 1. 加载基础模型配置
    config = AutoConfig.from_pretrained(base_model_name_or_path)
    # 2. 加载预训练 LLaMA 模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # 3. 配置 LoRA 参数
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # 4. 将 LoRA 植入模型
    model = get_peft_model(model, lora_cfg)
    return model

def model_to_parameters(model: nn.Module) -> list:
    """
    将 PyTorch 模型（含 LoRA adapter）参数转为 Flower 可传输的 numpy.ndarray 列表。
    """
    param_list = []
    for _, param in model.named_parameters():
        param_list.append(param.detach().cpu().numpy())
    return param_list

def parameters_to_model(model: nn.Module, parameters: list) -> nn.Module:
    """
    将 Flower 下发的参数列表（numpy.ndarray）载入到模型里，
    假设参数顺序与 model.named_parameters() 顺序一致。
    """
    state_dict = {}
    for (name, _), array in zip(model.named_parameters(), parameters):
        state_dict[name] = torch.tensor(array)
    model.load_state_dict(state_dict, strict=False)
    return model



def load_model():
    """
    加载基础 Llama 模型并添加 LoRA adapter。
    返回一个已封装好的 PyTorch Model，待客户端直接调用 train() 即可。
    """
    base_model = cfg.base_model
    data_path = cfg.data_path
    token = cfg.token

    if base_model in ["meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"]:
    
        config = AutoConfig.from_pretrained(base_model, token=token)
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            
            if "type" not in config.rope_scaling or "factor" not in config.rope_scaling:
                config.rope_scaling = None  
            else:
                config.rope_scaling = {"type": "linear", "factor": 32.0}  

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
            config=config,
            # rope_scaling=rope_scaling_fix,  
            token=token,
        )
    
    elif base_model in ["roberta-large"]:
        if data_path.lower() == "mnli":
            num_labels = 3
        elif data_path.lower() == "sst2" or data_path.lower() == "qnli" or data_path.lower() == "qqp":
            num_labels = 2

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels, 
            trust_remote_code=True,
        )

    elif base_model in ["roberta-base"]:

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=20, 
            trust_remote_code=True,
        )
    
    else:

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
            token=token,
        )
    from omegaconf import ListConfig
    target_modules = cfg.target_modules
    if isinstance(target_modules, ListConfig):
        target_modules = list(target_modules)
    
    if base_model in ["meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"]:
        config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            # lora_use_scale=use_scalelora,  
            # lora_use_mask=use_masklora,   
            target_modules=target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # lora_mask_client=mask_init,
        )
    elif base_model in ["roberta-base", "roberta-large"]:
        config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,  
            # modules_to_save=["classifier"],
        )
        

    model = get_peft_model(model, config)  
    
    return model