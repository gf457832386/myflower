# evaluate.py

import torch
from typing import Callable, Dict, Tuple
from peft import set_peft_model_state_dict, get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, default_data_collator
from omegaconf import OmegaConf
import copy
import json
import os
import re
import sys
import argparse

import fire
import time
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification

cfg = OmegaConf.load("fedloca/conf/base.yaml")
def get_evaluate_fn(base_model,lora_weights,save_dir,testdatapath,dataset):
    """
    返回一个 evaluation function，可供 FedAvg 或自定义策略在每轮聚合后评估模型效果。
    参数:
        model: 一个包含 LoRA 的 PEFT 模型
        eval_loader: 验证集 DataLoader
    返回:
        evaluate(weights): 函数，接收聚合后的 LoRA 参数权重，返回 (loss, metrics_dict)
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:
        pass

    def evaluate(base_model,lora_weights,save_dir,testdatapath,dataset,round_num) :


        tokenizer, model = load_model(base_model,lora_weights,save_dir,testdatapath,dataset,round_num)
        data_list = load_data(base_model,lora_weights,save_dir,testdatapath,dataset,round_num)
        print(f"Test dataset size = {len(data_list)}")
        

        batches = create_batch(data_list, batch_size=1)

        total = 0
        correct = 0
        results = []

        pbar = tqdm(total=len(batches), desc="Evaluating")

        for batch_data in batches:
            # batch_data = [{ "text": "...", "label": "label7" }, ...]
            # texts = [item["text"] for item in batch_data]
            texts = preprocess_batch(batch_data, dataset)

            gold_answers = [item["label"] for item in batch_data]

            # 1) tokenizer
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 2) forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # shape: [batch_size, num_labels]

            # 3) argmax
            preds = torch.argmax(logits, dim=-1).tolist()
            # print("Predictions:", preds)

            # 4) compare
            for i, pred_id in enumerate(preds):
                pred_label_str = pred_id  # e.g. label7
                gold_label_str = gold_answers[i]    # e.g. label7
                # print("pred_label_str:", pred_label_str)
                # print("gold_label_str:", gold_label_str)

                is_correct = (pred_label_str == gold_label_str)
                if is_correct:
                    correct += 1
                total += 1

                # record details
                r = copy.deepcopy(batch_data[i])
                r["pred"] = pred_label_str
                r["flag"] = is_correct
                results.append(r)

            pbar.update(1)

        pbar.close()
        acc = correct / total if total else 0
        print(f"Final accuracy = {acc:.4f}  ({correct}/{total})")

    
        round_dir = os.path.join(save_dir, f"round_{round_num}")
        if not os.path.exists(round_dir):
            os.mkdir(round_dir)
        
        save_file = os.path.join(round_dir, f'{base_model}-{dataset}.json')

        with open(save_file, "w+", encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"Saved details to {save_file}")


    def load_model():
        print("Loading classification model + LoRA weights...")

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        # roberta-base typically uses pad_token_id=1
        tokenizer.pad_token_id = tokenizer.pad_token_id or 1

        if dataset == "20newsgroup":
            num_lables = 20
        elif dataset == "mnli":
            num_lables = 3
        else:
            num_lables = 2
        

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_lables  # e.g. 20 newsgroup
        )
        model.to(device)

        if lora_weights.lower() != "none":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16 if device=="cuda" else torch.float32,
                device_map="auto" if device=="cuda" else {"":device}
            )

        model.eval()
        return tokenizer, model

    def load_data() -> list:
        # data = [ { "text": "...", "answer": "label7" }, ... ]
        file_path = testdatapath
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot find dataset file : {file_path}")
        data_list = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  
                obj = json.loads(line)  
                data_list.append(obj)
        return data_list

    def create_batch(dataset, batch_size):
        batches = []
        total = len(dataset)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batches.append(dataset[start:end])
        return batches

    def preprocess_batch(batch_data, task_name):
        
        inputs = []
        task_name = task_name.lower()

        for item in batch_data:
            if task_name == "20newsgroup":
                inputs.append(item["text"])

            elif task_name in {"mnli", "rte", "qnli", "qqp"}:
            
                first = item.get("premise") or item.get("question1") or item.get("question") or item.get("sentence1")
                second = item.get("hypothesis") or item.get("question2") or item.get("sentence2")

                if first is None or second is None:
                    raise ValueError(f"Missing fields in data for {task_name}")
                inputs.append(f"{first} [SEP] {second}")

            elif task_name == "sst2":
                inputs.append(item["sentence"])

            else:
                raise NotImplementedError(f"Unsupported task: {task_name}")

        return inputs
