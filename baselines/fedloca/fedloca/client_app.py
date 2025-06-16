"""Phoebe: A Flower Baseline."""

import torch
from transformers import Trainer, TrainingArguments, default_data_collator
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fedloca.model import get_weights, set_weights
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Tuple
from transformers import AutoTokenizer
from omegaconf import OmegaConf
cfg = OmegaConf.load("fedloca/conf/base.yaml")
from copy import deepcopy
import random
from typing import Dict, List
import torch
import copy
import subprocess
import transformers
import torch.nn.functional as F
#from trl import SFTTrainer
from transformers import TrainerCallback # type: ignore
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import get_peft_model_state_dict, set_peft_model_state_dict,LoraConfig,LoraModel,get_peft_model
#from peft import get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm
import os
import numpy as np
import math
from transformers import default_data_collator


class FlowerClient(NumPyClient):
    """A class defining the client."""

    def __init__(self, model,global_dict,tokenizer,train_dataloader, eval_dataloader, cid):
        self.model = model
        self.trainloader = train_dataloader
        self.evalloader = eval_dataloader
        self.local_epochs = cfg.local_epochs
        self.cid = cid
        self.output_dir = f"./client_{cid}_output"
        self.tokenizer = tokenizer
        self.global_dict = global_dict


        
        

    def fit(self, parameters, config):
        # 设置模型权重
        set_weights(self.model, parameters)

        # 构建 Trainer 对象
        fed_args =cfg
        if fed_args.data_path=='20newsgroup' or fed_args.base_model=="roberta-base" or fed_args.base_model=="roberta-large":
            trainer = transformers.Trainer(  
                model=self.model,
                train_dataset=self.trainloader.dataset,
                eval_dataset=self.evalloader.dataset,
                args=transformers.TrainingArguments(
                    per_device_train_batch_size=fed_args.micro_batch_size,
                    gradient_accumulation_steps=fed_args.gradient_accumulation_steps,
                    warmup_steps=100,
                    num_train_epochs=fed_args.num_epochs,
                    learning_rate= fed_args.learning_rate,
                    fp16=True,
                    logging_steps=10,
                    optim="adamw_torch",
                    evaluation_strategy="steps" if fed_args.val_set_size > 0 else "no",
                    save_strategy="steps",
                    eval_steps=200 if fed_args.val_set_size > 0 else None,
                    save_steps=200,
                    output_dir=fed_args.output_dir,
                    save_total_limit=3,
                    load_best_model_at_end=True if fed_args.val_set_size > 0 else False,
                    ddp_find_unused_parameters=False,
                    group_by_length=False,
                    # report_to="wandb" if use_wandb else None,
                    # # debug="underflow_overflow",
                    # # report_to=None,
                    # run_name=wandb_run_name if use_wandb else None,
                    # #remove_unused_columns=False  
                    
                ),
                data_collator=default_data_collator,
                
            )
        else:
            trainer = transformers.Trainer(  
                model=self.model,
                train_dataset=self.trainloader.dataset,
                eval_dataset=self.evalloader.dataset,
                args=transformers.TrainingArguments(
                    per_device_train_batch_size=fed_args.micro_batch_size,
                    gradient_accumulation_steps=fed_args.gradient_accumulation_steps,
                    warmup_steps=100,
                    num_train_epochs=fed_args.num_epochs,
                    learning_rate= fed_args.learning_rate,
                    fp16=True,
                    logging_steps=10,
                    optim="adamw_torch",
                    evaluation_strategy="steps" if fed_args.val_set_size > 0 else "no",
                    save_strategy="steps",
                    eval_steps=200 if fed_args.val_set_size > 0 else None,
                    save_steps=200,
                    output_dir=fed_args.output_dir,
                    save_total_limit=3,
                    load_best_model_at_end=True if fed_args.val_set_size > 0 else False,
                    ddp_find_unused_parameters=False,
                    group_by_length=False,
                    # report_to="wandb" if use_wandb else None,
                    # # debug="underflow_overflow",
                    # # report_to=None,
                    # run_name=wandb_run_name if use_wandb else None,
                    # #remove_unused_columns=False   
                ),
                data_collator=transformers.DataCollatorForSeq2Seq(
                    self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ),
            )

        # 训练模型
        result = trainer.train()

        # 获取训练后的lora参数
        updated_weights = get_weights(self.model)

        return updated_weights, len(self.trainloader.dataset), {}
        


    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        trainer = Trainer(
            model=self.model,
            eval_dataset=self.evalloader.dataset,
            args=TrainingArguments(
                output_dir=self.output_dir,
                per_device_eval_batch_size=config.get("batch_size", 8),
                dataloader_drop_last=False,
                report_to="none",
                do_train=False,
                do_eval=True,
            ),
            data_collator=default_data_collator,
            tokenizer=self.tokenizer,
        )
        eval_result = trainer.evaluate()
        return eval_result["eval_loss"], len(self.evalloader.dataset), {
            "accuracy": eval_result.get("eval_accuracy", -1),
        }



#定义一个生成客户端的函数，它返回一个可用来构造 Flower 客户端的函数（即 client_fn）
def gen_client_fn(model,global_dict,tokenizer,train_dataloader_list, eval_dataloader_list):
    """Generate the client function that creates the Flower Clients."""

    def client_fn(cid: str):
        train_dataloader = train_dataloader_list[int(cid)]
        eval_dataloader = eval_dataloader_list[int(cid)]
        local_model = deepcopy(model)
        
        return FlowerClient(
           local_model,global_dict,tokenizer,train_dataloader, eval_dataloader,cid
        ).to_client()

    return client_fn

# Flower ClientApp
app = ClientApp(
    client_fn=gen_client_fn,
)
