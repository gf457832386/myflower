# main.py
import pickle
from pathlib import Path
import os
import sys
from typing import List
import argparse
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import torch
import flwr as fl
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from fedloca.dataset_process import dataprocess
from fedloca.client_app import gen_client_fn
from fedloca.evaluate import get_evaluate_fn  # ✅ 新评估函数
from fedloca.strategy import FedLoCAStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel , AutoConfig,LlamaTokenizerFast,AutoModelForSequenceClassification # 
import copy
from fedloca.model import get_llama_with_lora  # or other get_lora_model
from peft import TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="conf", config_name="base")
def main(cfg: DictConfig) -> None:

    #创建文件夹
    # 取出模型保存路径和结果保存路径，如果目录不存在，就递归创建
    model_dir = Path(cfg.model_path)
    results_dir = Path(cfg.results_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 固定随机种子，保证可重现
    np.random.seed(cfg.seed)

    # 1. 打印完整配置，方便调试
    print(OmegaConf.to_yaml(cfg))

    # 2. 加载并划分处理数据集
    train_dataloader_list, eval_dataloader_list, n_sample_list = dataprocess()
    print(f">>> Data partition finished: {cfg.num_clients} clients.")

    # 3. 获得模型
    base_model = cfg.base_model
    data_path = cfg.data_path

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

    
    if base_model in ["meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"]:
        config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            # lora_use_scale=use_scalelora,  
            # lora_use_mask=use_masklora,   
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # lora_mask_client=mask_init,
        )
    elif base_model in ["roberta-base", "roberta-large"]:
        config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,  
            # modules_to_save=["classifier"],
            )
    model = get_peft_model(model, config)  

    if "llama2" in base_model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        # tokenizer = LlamaTokenizer.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        # tokenizer = LlamaTokenizerFast.from_pretrained(base_model)

    global_dict = {k: v.clone().detach() for k, v in get_peft_model_state_dict(model).items()}
    print(global_dict)

    # 4. 生成 Flower 客户端函数
    client_fn = gen_client_fn(model,global_dict,tokenizer,train_dataloader_list, eval_dataloader_list)

    # 5. 构造服务端评估函数：需要一个裸模型实例（带 LoRA）
    base_model=cfg.base_model
    output_dir=cfg.output_dir
    results_path=cfg.results_path
    test_dataset = os.path.join(os.path.dirname(__file__),"dataset", cfg.data_path, "test.json")
    data_path=cfg.data_path
    evaluate_fn = get_evaluate_fn(base_model,output_dir,results_path,test_dataset,data_path)


    # 7. 实例化自定义 FedLoCA 策略
    strategy = FedLoCAStrategy()

    # 8. 启动 Flower Simluation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.num_cpus, "num_gpus": cfg.num_gpus},
    )

    # 9. 保存结果
    final_acc = history.metrics_centralized["accuracy"][-1][1]
    save_dir = HydraConfig.get().runtime.output_dir
    strategy_name = strategy.__class__.__name__
    dataset_name = "20Newsgroup"
    filename = f"results_{strategy_name}_{dataset_name}_clients{cfg.num_clients}_rounds{cfg.num_rounds}_acc{final_acc:.4f}.pkl"
    results_path = Path(save_dir) / filename
    with open(results_path, "wb") as f:
        pickle.dump({"history": history}, f)
    print(f">>> Results saved to {results_path}")

if __name__ == "__main__":
    main()
