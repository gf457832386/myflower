import numpy as np
from scipy.stats import dirichlet
import torch
from torch.utils.data import DataLoader, RandomSampler, random_split, SequentialSampler, Subset
import copy
import os
import logging  
from collections import defaultdict 
from omegaconf import OmegaConf
cfg = OmegaConf.load("fedloca/conf/base.yaml")
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def partition(train_dataset, test_dataset):
    set_seed(42)
    train_dataloader_list = [copy.deepcopy(1) for _ in range(cfg.num_clients)]
    test_dataloader_list = [copy.deepcopy(1) for _ in range(cfg.num_clients)]
    
    n_sample_list = [0 for _ in range(cfg.num_clients)]

    
    if cfg.data_partition_method == 'iid': 

        if cfg.data_path=='20newsgroup' or cfg.data_path in ["mnli", "sst2", "qqp", "qnli", "rte"]:


          
            label_to_indices = defaultdict(list)
            for i, example in enumerate(train_dataset):
                label = example["label"]
                label_to_indices[label].append(i)

           
            client_to_idxs = [[] for _ in range(cfg.num_clients)]

            for label, indices in label_to_indices.items():
                indices = np.array(indices)
                np.random.shuffle(indices) 
                samples_per_client = len(indices) // cfg.num_clients
                extra = len(indices) % cfg.num_clients

                start = 0
                for c in range(cfg.num_clients):
                    end = start + samples_per_client + (1 if c < extra else 0)
                    client_to_idxs[c].extend(indices[start:end].tolist())
                    start = end

           
            for c in range(cfg.num_clients):
                subset_idxs = client_to_idxs[c]
                subset = Subset(train_dataset, subset_idxs)
                train_sampler = RandomSampler(subset)
                train_dataloader_list[c] = DataLoader(subset, sampler=train_sampler, batch_size=cfg.micro_batch_size)
                n_sample_list[c] = len(subset)
                print(f"Client {c}: {len(subset)} samples")

            
            for i in range(cfg.num_clients):
                test_sampler = SequentialSampler(test_dataset)
                test_dataloader_list[i] = DataLoader(test_dataset, sampler=test_sampler, batch_size=cfg.micro_batch_size)
                print(f'Client {i}: {len(test_dataloader_list[i].dataset)} (test set)')
            

           

        else:
            subset_size = len(train_dataset) // cfg.num_clients
            remaining_size = len(train_dataset) - subset_size * cfg.num_clients
            subset_sizes = [subset_size] * cfg.num_clients
            for i in range(remaining_size):
                subset_sizes[i] += 1
            subsets = random_split(train_dataset, subset_sizes)
            print('number of samples for train')
            for i, subset in enumerate(subsets):
                train_sampler = RandomSampler(subset)
                train_dataloader_list[i] = DataLoader(subset, sampler=train_sampler, batch_size=cfg.micro_batch_size)
                print(f'Client {i}: {len(train_dataloader_list[i].dataset)}')

                n_sample_list[i] = len(train_dataloader_list[i].dataset)
           
            for i in range(cfg.num_clients):
                test_sampler = SequentialSampler(test_dataset)
                test_dataloader_list[i]=DataLoader(test_dataset, sampler=test_sampler, batch_size=cfg.micro_batch_size)
                print(f'Client {i}: {len(test_dataloader_list[i].dataset)}')

    elif cfg.data_partition_method == 'noniid':

        if cfg.data_path in ["mnli", "sst2", "qqp", "qnli", "rte"]:
            if cfg.data_path == "mnli":
                lable_nums=3
            else:
                lable_nums=2
            num_clients = cfg.num_clients
            num_samples = len(train_dataset)
            dirichlet_alpha = cfg.dirichlet_alpha  
            print(f"Using non-IID Dirichlet partition for {cfg.data_path}, alpha={dirichlet_alpha}, #clients={num_clients}")
            
            
            label_indices = {}
            for i in range(lable_nums): 
                label_indices[i] = []

            for idx in range(num_samples):
                sample = train_dataset[idx]
                y = sample["label"] 
                label_indices[y].append(idx)

           
            client_to_idxs = [ [] for _ in range(num_clients)] 

            for i in range(lable_nums):
                n_i = len(label_indices[i])
                if n_i == 0:
                    continue
               
                proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
                proportions = proportions ** 0.1    
                proportions /= proportions.sum()

                
                counts = (proportions * n_i).astype(int)

                
                diff = n_i - np.sum(counts)
                while diff > 0:
                    for c in range(num_clients):
                        counts[c] += 1
                        diff -= 1
                        if diff == 0:
                            break
                while diff < 0:
                    for c in range(num_clients):
                        if counts[c] > 0:
                            counts[c] -= 1
                            diff += 1
                            if diff == 0:
                                break
                
             
                idx_array = np.array(label_indices[i])
                np.random.shuffle(idx_array)
                start = 0
                for c in range(num_clients):
                    c_size = counts[c]
                    c_part = idx_array[start : start + c_size]
                    start += c_size
                    client_to_idxs[c].extend(c_part.tolist())

           
            for c in range(num_clients):
                subset_idxs = client_to_idxs[c]
                subset = Subset(train_dataset, subset_idxs)
                train_sampler = RandomSampler(subset)
                train_dataloader_list[c] = DataLoader(subset, sampler=train_sampler, batch_size=cfg.micro_batch_size)
                n_sample_list[c] = len(subset)
                print(f"Client {c}: {len(subset)} samples")

            
            for i in range(cfg.num_clients):
                test_sampler = SequentialSampler(test_dataset)
                test_dataloader_list[i] = DataLoader(test_dataset, sampler=test_sampler, batch_size=cfg.micro_batch_size)
                print(f'Client {i}: {len(test_dataloader_list[i].dataset)} (test set)')

        elif cfg.data_path=='20newsgroup':
           
            num_clients = cfg.num_clients
            num_samples = len(train_dataset)
            dirichlet_alpha = cfg.dirichlet_alpha  
            print(f"Using non-IID Dirichlet partition for 20newsgroup, alpha={dirichlet_alpha}, #clients={num_clients}")
            
           
            label_indices = {}
            for i in range(20): 
                label_indices[i] = []

            for idx in range(num_samples):
                sample = train_dataset[idx]
                y = sample["label"]  
                label_indices[y].append(idx)

           
            client_to_idxs = [ [] for _ in range(num_clients)]  

            for i in range(20):
                n_i = len(label_indices[i])
                if n_i == 0:
                    continue
               
                proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))

               
                counts = (proportions * n_i).astype(int)

               
                diff = n_i - np.sum(counts)
                while diff > 0:
                    for c in range(num_clients):
                        counts[c] += 1
                        diff -= 1
                        if diff == 0:
                            break
                while diff < 0:
                    for c in range(num_clients):
                        if counts[c] > 0:
                            counts[c] -= 1
                            diff += 1
                            if diff == 0:
                                break
                
               
                idx_array = np.array(label_indices[i])
                np.random.shuffle(idx_array)
                start = 0
                for c in range(num_clients):
                    c_size = counts[c]
                    c_part = idx_array[start : start + c_size]
                    start += c_size
                    client_to_idxs[c].extend(c_part.tolist())

           
            for c in range(num_clients):
                subset_idxs = client_to_idxs[c]
                subset = Subset(train_dataset, subset_idxs)
                train_sampler = RandomSampler(subset)
                train_dataloader_list[c] = DataLoader(subset, sampler=train_sampler, batch_size=cfg.micro_batch_size)
                n_sample_list[c] = len(subset)
                print(f"Client {c}: {len(subset)} samples")

           
            for i in range(cfg.num_clients):
                test_sampler = SequentialSampler(test_dataset)
                test_dataloader_list[i] = DataLoader(test_dataset, sampler=test_sampler, batch_size=cfg.micro_batch_size)
                print(f'Client {i}: {len(test_dataloader_list[i].dataset)} (test set)')
        

        else:

            num_clients = cfg.num_clients
            num_samples = len(train_dataset)
            idxs = np.arange(num_samples)  
            np.random.shuffle(idxs)

           
            dirichlet_alpha=cfg.dirichlet_alpha*10
            proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))  
            proportions = (proportions * num_samples).astype(int)  

            min_samples=int(cfg.dirichlet_alpha*num_samples/num_clients)

            
            proportions = np.maximum(proportions, min_samples)
            proportions = (proportions / proportions.sum() * num_samples).astype(int)

           
            diff = num_samples - np.sum(proportions)
            if diff > 0:  
                for i in range(diff):
                    proportions[i % num_clients] += 1
            elif diff < 0:  
                for i in range(-diff):
                    if proportions[i % num_clients] > min_samples:
                        proportions[i % num_clients] -= 1

          

          
            train_dataloader_list = {}
            n_sample_list = {}

            remaining_idxs = set(idxs)  

            for i, num_samples_user in enumerate(proportions):
                if len(remaining_idxs) < num_samples_user:  
                    num_samples_user = len(remaining_idxs)

                
                subset_idxs = set(np.random.choice(list(remaining_idxs), num_samples_user, replace=False))
                remaining_idxs -= subset_idxs  

                subset = Subset(train_dataset, list(subset_idxs)) 
                train_sampler = RandomSampler(subset)
                train_dataloader_list[i] = DataLoader(subset, sampler=train_sampler, batch_size=cfg.micro_batch_size)

                print(f'Client {i}: {len(train_dataloader_list[i].dataset)} samples')

                n_sample_list[i] = len(train_dataloader_list[i].dataset)

            if len(remaining_idxs) > 0:
                print(f"There are {len(remaining_idxs)} samples left unassigned. Handling them if needed...")
            
        

            print(len(test_dataset))

           
            for i in range(cfg.num_clients):
                test_sampler = SequentialSampler(test_dataset)
                test_dataloader_list[i]=DataLoader(test_dataset, sampler=test_sampler, batch_size=cfg.micro_batch_size)
                print(f'Client {i}: {len(test_dataloader_list[i].dataset)}')  

    else:
        raise NotImplementedError()

   
    return train_dataloader_list, test_dataloader_list, n_sample_list


