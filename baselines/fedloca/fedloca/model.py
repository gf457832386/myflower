"""Phoebe: A Flower Baseline."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf
cfg = OmegaConf.load("fedloca/conf/base.yaml")
# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')."""

#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         """Do forward."""
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)


# def train(net, trainloader, epochs, device):
#     """Train the model on the training set."""
#     net.to(device)  # move model to GPU if available
#     criterion = torch.nn.CrossEntropyLoss()
#     criterion.to(device)
#     optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
#     net.train()
#     running_loss = 0.0
#     for _ in range(epochs):
#         for batch in trainloader:
#             images = batch["img"]
#             labels = batch["label"]
#             optimizer.zero_grad()
#             loss = criterion(net(images.to(device)), labels.to(device))
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#     avg_trainloss = running_loss / len(trainloader)
#     return avg_trainloss


# def test(net, testloader, device):
#     """Validate the model on the test set."""
#     net.to(device)
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in testloader:
#             images = batch["img"].to(device)
#             labels = batch["label"].to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     loss = loss / len(testloader)
#     return loss, accuracy


# def get_weights(net):
#     """Extract model parameters as numpy arrays from state_dict."""
#     return [val.cpu().numpy() for _, val in net.state_dict().items()]


# def set_weights(net, parameters):
#     """Apply parameters to an existing model."""
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)


# myfl/models.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model,get_peft_model_state_dict, set_peft_model_state_dict

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
