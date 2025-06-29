"""Phoebe: A Flower Baseline."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from datasets import load_dataset
from omegaconf import DictConfig
import os
import json
from omegaconf import OmegaConf
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "conf/base.yaml"))
cfg = OmegaConf.load(base_path)
from fedloca.datapartition import partition



from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel , AutoConfig,LlamaTokenizerFast
FDS = None  # Cache FederatedDataset


def dataprocess() -> tuple[list, list, list]:
    """
    处理数据集，返回 train_dataloader_list, eval_dataloader_list, n_sample_list。
    这里假设 data_path 是一个字符串，指向数据集的路径。
    """
    data_path=cfg.data_path
    if data_path.endswith(".json"):  # todo: support jsonl
        from datasets import load_dataset
        data = load_dataset("json", data_files=data_path)

    elif data_path.lower() == "20newsgroup":  
        from sklearn.datasets import fetch_20newsgroups
        from datasets import Dataset
        import random

        if cfg.generate_data==1:
            raw = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
            texts = raw.data
            labels = raw.target
           

            data_list = [{"text": t, "label": l} for t, l in zip(texts, labels)]
            random.shuffle(data_list)  

            dataset_all = Dataset.from_list(data_list)
            print("Total samples in 20Newsgroup(all):", len(dataset_all))

            # data = data.train_test_split(test_size=0.2)  

            
            train_val_test_split = dataset_all.train_test_split(
                test_size=0.1,
                seed=42,
            )
            dataset_intermediate = train_val_test_split["train"]  # 90%
            dataset_test = train_val_test_split["test"]           # 10%

            # print("=> Intermediate set:", len(dataset_intermediate))
            # print("=> Test set:", len(dataset_test))


           
            train_val_split = dataset_intermediate.train_test_split(
                test_size=0.1111111,  
                seed=42,
            )
            dataset_train = train_val_split["train"]  # ~80%
            dataset_val   = train_val_split["test"]   # ~10%

            print("Loaded 20Newsgroup dataset:")
            print("Final Train set:", len(dataset_train))
            print("Final Val set:  ", len(dataset_val))
            print("Final Test set: ", len(dataset_test))

           
            save_path = os.path.join("dataset", data_path)
            os.makedirs(save_path, exist_ok=True)

            def save_jsonl(dataset, file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

            save_jsonl(dataset_train, os.path.join(save_path, "train.json"))
            save_jsonl(dataset_val,   os.path.join(save_path, "val.json"))
            save_jsonl(dataset_test,  os.path.join(save_path, "test.json"))

           
            train_data = dataset_train.map(generate_and_tokenize_prompt, batched=False)
            val_data   = dataset_val.map(generate_and_tokenize_prompt, batched=False)
            test_data  = dataset_test.map(generate_and_tokenize_prompt, batched=False)
            datasetjson = {
                "train": train_data,
                "val":   val_data,
                "test":  test_data
            }

        
        else:
            # load_path = os.path.join("./dataset", data_path)
            load_path = os.path.join(os.path.dirname(__file__), "dataset", data_path)

            assert os.path.exists(load_path), f"Load path does not exist: {load_path}"

            def load_jsonl(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    return [json.loads(line) for line in f]

            train_raw = load_jsonl(os.path.join(load_path, "train.json"))
            val_raw   = load_jsonl(os.path.join(load_path, "val.json"))
            test_raw  = load_jsonl(os.path.join(load_path, "test.json"))

            dataset_train = Dataset.from_list(train_raw)
            dataset_val   = Dataset.from_list(val_raw)
            dataset_test  = Dataset.from_list(test_raw)

            print("Loaded cached dataset from", load_path)
            print("Train set:", len(dataset_train))
            print("Val set:  ", len(dataset_val))
            print("Test set: ", len(dataset_test))

            train_data = dataset_train.map(generate_and_tokenize_prompt, batched=True,num_proc=8)
            val_data   = dataset_val.map(generate_and_tokenize_prompt, batched=True,num_proc=8)
            test_data  = dataset_test.map(generate_and_tokenize_prompt, batched=True,num_proc=8)
            datasetjson = {
                "train": train_data,
                "val":   val_data,
                "test":  test_data
            }
    elif data_path.lower() in ["mnli", "sst2", "qqp", "qnli", "rte"]:
        if cfg.generate_data==1:
        
            from datasets import load_dataset

            
            raw_datasets = load_dataset("glue", data_path.lower())
            

           
            if data_path.lower() == "mnli":
                val_split = "validation_matched"
                test_split = "test_matched"
                

            else:
                val_split = "validation"
                test_split = "test"

            train_dataset= raw_datasets["train"]
            val_dataset= raw_datasets[val_split]
            # test_dataset= raw_datasets[test_split]

           
            save_path = os.path.join("dataset", data_path)
            os.makedirs(save_path, exist_ok=True)

            def save_jsonl(dataset, file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

            train_subset = train_dataset.shuffle(seed=42).select(range(int(len(train_dataset) * 0.2)))
            save_jsonl(train_subset, os.path.join(save_path, "train.json"))

            
            val_dataset_shuffled = val_dataset.shuffle(seed=42)
            mid = len(val_dataset_shuffled) // 2
            val_eval = val_dataset_shuffled.select(range(mid))
            val_test = val_dataset_shuffled.select(range(mid, len(val_dataset_shuffled)))

            save_jsonl(val_eval,   os.path.join(save_path, "val.json"))
            save_jsonl(val_test,  os.path.join(save_path, "test.json"))
            train_data = train_subset.map(
                generate_and_tokenize_prompt,
                batched=False
            )
            val_data = val_eval.map(
                generate_and_tokenize_prompt,
                batched=False
            )
            test_data = val_test.map(
                generate_and_tokenize_prompt,
                batched=False
            )
            
            datasetjson = {
                "train": train_data,
                "val":   val_data,
                "test":  test_data
            }
            print(f"Loaded GLUE {data_path.lower()}: "
                    f"train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        else: 
            from datasets import Dataset
            load_path = os.path.join(os.path.dirname(__file__), "dataset", data_path)
            assert os.path.exists(load_path), f"Load path does not exist: {load_path}"

            def load_jsonl(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    return [json.loads(line) for line in f]

            train_raw = load_jsonl(os.path.join(load_path, "train.json"))
            val_raw   = load_jsonl(os.path.join(load_path, "val.json"))
            test_raw  = load_jsonl(os.path.join(load_path, "test.json"))

            dataset_train = Dataset.from_list(train_raw)
            dataset_val   = Dataset.from_list(val_raw)
            dataset_test  = Dataset.from_list(test_raw)

            print("Loaded cached dataset from", load_path)
            print("Train set:", len(dataset_train))
            print("Val set:  ", len(dataset_val))
            print("Test set: ", len(dataset_test))

            train_data = dataset_train.map(generate_and_tokenize_prompt, batched=False)
            val_data   = dataset_val.map(generate_and_tokenize_prompt, batched=False)
            test_data  = dataset_test.map(generate_and_tokenize_prompt, batched=False)
            datasetjson = {
                "train": train_data,
                "val":   val_data,
                "test":  test_data
            }
        
            
    else:
        data = load_dataset(data_path)
    


    if data_path.lower() == "20newsgroup":
        train_data = datasetjson["train"]
        val_data   = datasetjson["val"] 
        test_data  = datasetjson["test"]
       
    elif data_path.lower() in ["mnli", "sst2", "qqp", "qnli", "rte"]:
        train_data = datasetjson["train"]
        val_data   = datasetjson["val"] 
        test_data  = datasetjson["test"]
    
    else:
       
        if cfg.val_set_size > 0:
            
            train_val = data["train"].train_test_split(
                test_size=cfg.val_set_size, shuffle=True, seed=42
            )
            
            train_data = (
                train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else: 
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
            val_data = None

    #切分数据
    train_dataloader_list, eval_dataloader_list, n_sample_list = partition(train_data, val_data)

    return train_dataloader_list, eval_dataloader_list, n_sample_list



    
def tokenize(prompt,add_eos_token=True):

        base_model= cfg.base_model    
       
        if "llama2" in base_model:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
        else:
            # tokenizer = LlamaTokenizer.from_pretrained(base_model)
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            # tokenizer = LlamaTokenizerFast.from_pretrained(base_model)


        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cfg.dataset.cutoff_len,
            padding=False,  
            #padding = 'max_length',
            return_tensors=None  
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cfg.dataset.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501

def generate_and_tokenize_prompt(data_point): 
    
        data_path = cfg.data_path
        base_model = cfg.base_model
        cutoff_len = cfg.cutoff_len
        base_model= cfg.base_model    
       
        if "llama2" in base_model:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
        else:
            # tokenizer = LlamaTokenizer.from_pretrained(base_model)
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            # tokenizer = LlamaTokenizerFast.from_pretrained(base_model)


        if data_path.lower() == "20newsgroup":
            tokenized = tokenizer(
                data_point["text"],
                truncation=True,
                padding="max_length",
                max_length=cutoff_len
            )
            tokenized["labels"] = data_point["label"]
            return tokenized
        elif data_path.lower() in ["mnli", "sst2", "qnli", "qqp", "rte"]:
            
            if data_path.lower() == "sst2":
                text_a, text_b = data_point["sentence"], None
            elif data_path.lower() == "mnli":
                text_a, text_b = data_point["premise"], data_point["hypothesis"]
            elif data_path.lower() == "qqp":
                text_a, text_b = data_point["question1"], data_point["question2"]
            elif data_path.lower() == "qnli":
                text_a, text_b = data_point["question"], data_point["sentence"]
            elif data_path.lower() == "rte":
                text_a, text_b = data_point["sentence1"], data_point["sentence2"]

           
            if text_b is None:
                tokenized = tokenizer(
                    text_a,
                    truncation=True,
                    padding="max_length",
                    max_length=cutoff_len
                )
            else:
                tokenized = tokenizer(
                    text_a,
                    text_b,
                    truncation=True,
                    padding="max_length",
                    max_length=cutoff_len
                )

            
            tokenized["labels"] = data_point["label"]
            return tokenized
            
        elif base_model in ["meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"]:

            full_prompt = generate_prompt(data_point)

           
            tokenized_full_prompt = tokenize(full_prompt)
            if not cfg.model.train_on_inputs:
                user_prompt = generate_prompt({**data_point, "output": ""})
                tokenized_user_prompt = tokenize(
                    user_prompt, 
                    add_eos_token=False
                )
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                # user_prompt_len = tokenized_user_prompt["input_ids"].shape[-1]

                tokenized_full_prompt["labels"] = [
                                                    -100
                                                ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                        user_prompt_len:
                                                                        ]  # could be sped up, probably
                
                
                if not tokenized_full_prompt.get("input_ids"):
                    print("Error: No input_ids generated for data_point:", data_point)
                else:
                    print("Generated input_ids:", tokenized_full_prompt["input_ids"])
            return tokenized_full_prompt






# def twenty_newsgroup(
#     data_path: str,
#     val_set_size: float,
#     test_set_size: float,
#     seed: int,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple, int]:
#     """
#     通过 HuggingFace datasets 加载 20Newsgroup 数据集，
#     并根据 val_set_size/test_set_size 进行 train/val/test 划分，
#     最后返回 (X_train, y_train, X_test, y_test, input_shape, num_classes)。
#     """
#     # 1. 从原始代码中复用 data_processers.py 里的数据预处理函数
#     from baselines.fedloca.fedloca.data_processers import preprocess_20newsgroup

#     # 假设 preprocess_20newsgroup 返回一个字典 {'train':DatasetDict, 'test':DatasetDict}
#     dataset = load_dataset("newsgroup", "20news-bydate")
#     # 调用您自己 data_processers.py 里的预处理
#     processed = preprocess_20newsgroup(dataset)

#     # processed["train"] 已包含分词、截断、encode 等逻辑，类似您原来做的
#     # processed["test"] 是原始测试集
#     train_val = processed["train"].train_test_split(
#         test_size=val_set_size, seed=seed
#     )
#     dataset_train = train_val["train"]  # ~ (1 - val_set_size) 部分
#     dataset_val = train_val["test"]     # ~ val_set_size 部分
#     dataset_test = processed["test"]    # ~全部测试集

#     print("Loaded 20Newsgroup dataset:")
#     print("Train set size:", len(dataset_train))
#     print("Val set size:  ", len(dataset_val))
#     print("Test set size: ", len(dataset_test))

#     # 将 HF Dataset 转成 numpy 数组或 torch.Tensor（根据后续 client.py 接受的格式）
#     # 这里假定 tokenizer 已经在 preprocess_20newsgroup 中统一做过 encode，
#     # 且每个样本 item["input_ids"] 是长度为 cutoff_len 的 list[int]，
#     # item["label"] 是 0~19 之间的整数 label。

#     def dataset_to_numpy(ds):
#         X = np.array(ds["input_ids"])
#         y = np.array(ds["label"])
#         return X, y

#     x_train, y_train = dataset_to_numpy(dataset_train)
#     x_val, y_val = dataset_to_numpy(dataset_val)
#     x_test, y_test = dataset_to_numpy(dataset_test)

#     # 2. 合并 train/val，为了后续分区先合并到一起，再在 client_fn 中 90%:10% 划分
#     x_all = np.concatenate([x_train, x_val], axis=0)
#     y_all = np.concatenate([y_train, y_val], axis=0)

#     # 3. 返回给 main.py 用于全局划分
#     input_shape = list(x_all.shape[1:])  # e.g. [cutoff_len]
#     num_classes = len(set(y_all.tolist()))
#     return x_all, y_all, x_test, y_test, tuple(input_shape), num_classes


# def load_data(partition_id: int, num_partitions: int):
#     """Load partition CIFAR10 data."""
#     # Only initialize `FederatedDataset` once
#     global FDS  # pylint: disable=global-statement
#     if FDS is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         FDS = FederatedDataset(
#             dataset="uoft-cs/cifar10",
#             partitioners={"train": partitioner},
#         )
#     partition = FDS.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     pytorch_transforms = Compose(
#         [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     def apply_transforms(batch):
#         """Apply transforms to the partition from FederatedDataset."""
#         batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#         return batch

#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=32)
#     return trainloader, testloader
