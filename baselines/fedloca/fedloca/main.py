# main.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


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
from fedloca.strategy.alg_fedloca import Fedloca
from fedloca.dataset_process import dataprocess
from fedloca.client_app import gen_client_fn
from fedloca.evaluate import get_evaluate_fn  # ✅ 新评估函数
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel , AutoConfig,LlamaTokenizerFast,AutoModelForSequenceClassification # 
import copy
from fedloca.model import get_llama_with_lora, load_model  # or other get_lora_model
from peft import TaskType



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="conf", config_name="base")
def main(cfg: DictConfig) -> None:

    #创建文件夹
    # 取出模型保存路径和结果保存路径，如果目录不存在，就递归创建
    os.chdir(hydra.utils.get_original_cwd())
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

    model=load_model() 
    model.save_pretrained(cfg.output_dir)  


    if "llama2" in base_model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
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
    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=evaluate_fn
    )


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
