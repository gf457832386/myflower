import argparse
import logging
import dill
import torch
import math
import os
import random
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from torch.optim import Adam, AdamW, SGD
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaForMaskedLM
from transformers import RobertaModel
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from flwr.common import FitIns,Parameters
from scipy.optimize import minimize
import csv
import time
import copy
import os
import random
import time
import cma
import numpy as np
import torch
from ..data_process import construct_true_few_shot_data, split_data,data_processor,perturb_dataset
from cma.recombination_weights import RecombinationWeights
from transformers import RobertaTokenizer
from ..LMForwardAPI import LMForwardAPI
from flwr.common import FitRes,Status,Code,EvaluateRes
from flwr.client.client import Client
from ..utils import parameters2es,result2parameters
logger = logging.getLogger(__name__)
class ApiCallLimitError(Exception):
    pass

class GumbelBDPLClient(Client):
    def __init__(self, args,train_data,dev_data,test_data,user_dict_train,user_dict_dev, client_id,tokenizer,model_forward_api,local_cma_mu,frac = 1):

        self.tokenizer = tokenizer
        self.model_forward_api = model_forward_api
        self.cid = client_id
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.user_dict_dev = user_dict_dev
        self.user_dict_train = user_dict_train

        self.args = args

        self.config = args
        result=[]
        if args.file_name:
            with open("./pmi/" + args.file_name.lower() + ".txt",'r') as f:
                for line in f:
                    result = result + (list(line.strip('\n').split(',')))
        elif args.task_name:
            with open("./pmi/" + args.task_name.lower() + ".txt",'r') as f:
                for line in f:
                    result = result + (list(line.strip('\n').split(',')))

        unique = []
        [unique.append(i) for i in result if not i in unique]
        self.ngram_list = list(map(int, unique))
        self.eval_clients = args.eval_clients

        # initialize prompt. 
        prompt_search_space = args.prompt_search_space
        prompt_length = args.prompt_length
        #self.prompts_probs = torch.FloatTensor([[1 / prompt_search_space] * prompt_search_space] * prompt_length)
        #self.prompts_probs.requires_grad = True
        # self.train_dataloader = DataLoader(self.dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        # gumbel 
        self.prompts_alpha = torch.FloatTensor([[1 / prompt_search_space] * prompt_search_space] * prompt_length)
        self.prompts_alpha.requires_grad = True
        self.prompts_probs = F.gumbel_softmax(torch.log(self.prompts_alpha), tau=args.tau)

        # 
        self.prompt_optimizer = AdamW([{
            "params": [self.prompts_alpha],   # optimize alpha. 
            "weight_decay": args.weight_decay,  #0
        }], lr=args.prompt_learning_rate)

        # FL parameter. 
        self.num_local_step = args.local_iter

        self.completed_steps = 0

    def fit(self,ins):

        # Load the average prompt into the client's model
        #self.prompts_probs.data = average_theta.clone().detach()
        #self.prompts_probs.requires_grad = True
        average_theta = dill.loads(ins.parameters.tensors[0])
        self.prompts_alpha.data = average_theta.clone().detach()
        self.prompts_alpha.requires_grad = True
        train_sample_idxs, dev_sample_idxs = self.user_dict_train[self.idx], self.user_dict_dev[self.idx]
        print(f"Client {self.idx} execute local training on {len(train_sample_idxs)} samples...")
        print(f"Client {self.idx} train_sample_idxs {train_sample_idxs}")

        local_train_data = {
            "input_ids": torch.tensor(self.train_data["input_ids"].get(train_sample_idxs)),
            "attention_mask": torch.tensor(self.train_data["attention_mask"].get(train_sample_idxs)),
            "mask_pos": torch.tensor(self.train_data["mask_pos"].get(train_sample_idxs)),
            "labels": torch.tensor(self.train_data["labels"].get(train_sample_idxs)),
        }
        local_dev_data = {
            "input_ids": torch.tensor(self.dev_data["input_ids"].get(dev_sample_idxs)),
            "attention_mask": torch.tensor(self.dev_data["attention_mask"].get(dev_sample_idxs)),
            "mask_pos": torch.tensor(self.dev_data["mask_pos"].get(dev_sample_idxs)),
            "labels": torch.tensor(self.dev_data["labels"].get(dev_sample_idxs)),
        }
        if self.parallel:
            # expand training data to a larger batch for parallel evaluation
            self.train_data["input_ids"] = self.train_data["input_ids"].repeat(self.local_es.popsize, 1)
            self.train_data["attention_mask"] = self.train_data["attention_mask"].repeat(self.local_es.popsize, 1)
            self.train_data["mask_pos"] = self.train_data["mask_pos"].repeat(self.local_es.popsize)
            self.train_data["labels"] = self.train_data["labels"].repeat(self.local_es.popsize)

        local_train_data_aux = perturb_dataset(self.args, local_train_data, self.model_forward_api.config)

        self.model_forward_api.set_dataset(local_train_data, local_dev_data, local_train_data_aux)
        # Example training loop
        for _ in range(self.num_local_step):
            try:
                
                self.prompts_alpha.data = torch.clamp(self.prompts_alpha.data, min=1e-15)
                self.prompts_probs = F.gumbel_softmax(torch.log(self.prompts_alpha), tau=self.args.tau)
                prompts_dist = torch.distributions.Categorical(self.prompts_probs)
                batch = local_train_data
                with torch.no_grad():
                    bsz = len(batch['input_ids'])             # batch_size. 
                    label = batch["labels"].to(self.args.device)   
                    loss_list = []
                    prompts_discrete_indices_list = []
                    for k in range(self.args.sample_size):
                        prompts_discrete_indices = prompts_dist.sample() 
                        prompts_discrete_indices_list.append(prompts_discrete_indices) 
                        if self.args.use_ngram:
                            prompts_discrete_indices_ngram_list = []
                            indices_list = prompts_discrete_indices.int().tolist() # sampling index. 
                            for idx in indices_list:
                                prompts_discrete_indices_ngram_list.append(self.ngram_list[idx]) 
                            prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)
                            cur_input_ids = torch.cat([torch.zeros(bsz, 1, dtype=torch.long).to(self.args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(self.args.device), batch['input_ids'][:, 1:]], dim=1)
                        else: 
                            cur_input_ids = torch.cat([torch.zeros(bsz, 1, dtype=torch.long).to(self.args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(self.args.device), batch['input_ids'][:, 1:]], dim=1) # CLS + Discrete Prompt + input_ids

                        cur_attention_mask = torch.cat([torch.ones(bsz, 1).to(self.args.device), torch.ones(bsz, self.args.prompt_length).to(self.args.device), batch["attention_mask"][:, 1:]],dim=1) # [0, 1(prompt length), original_attention_mask]
                        mask_pos = np.where(np.array(cur_input_ids.cpu()) == self.tokenizer.mask_token_id)     # find mask position. 
                        mask_pos = torch.tensor(mask_pos[-1]) 
                        if self.model_forward_api.model_name in ["t5-small", "t5-base", "t5-large", "t5-3b"]:
                            logits = self.model_forward_api.model(
                                input_ids=self.train_data["input_ids"],
                                attention_mask=self.train_data["attention_mask"],
                                decoder_input_ids=self.train_data["decoder_input_ids"],
                                decoder_attention_mask=self.train_data["decoder_attention_mask"],
                            )["logits"]
                        elif self.model_forward_api.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "llama2"]:
                            logits = self.model_forward_api.model(
                                input_ids=self.train_data["input_ids"],
                                attention_mask=self.train_data["attention_mask"],
                            )["logits"]
                        else:
                            logits = self.model_forward_api.model(
                                input_ids=cur_input_ids,
                                attention_mask=cur_attention_mask,
                                mask_pos=mask_pos,
                            )["logits"]
                        loss, perf = self.model_forward_api.calc_metric(logits, self.train_data["labels"])
                        loss_list.append(loss.item())

                    loss_avg = sum(loss_list) / self.args.sample_size
                    
                    self.prompt_optimizer.zero_grad()

                    
                    # calculate the derivative w.r.t \alpha_{i,j} in Gumbel-softmax. 
                    derivative = (- self.prompts_probs / (self.prompts_alpha*self.args.tau)).repeat(self.args.sample_size, 1, 1)   # [5, 20, 200], -0 ~ -2000
                    
                    for k, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
                        for i in range(self.args.prompt_length):  #
                            derivative[k][i][prompts_discrete_indices[i]] = (1-self.prompts_probs[i][prompts_discrete_indices[i]])/(self.prompts_alpha[i][prompts_discrete_indices[i]]*self.args.tau)   

                    self.prompts_alpha.grad = torch.zeros_like(self.prompts_alpha)
                    for k in range(self.args.sample_size):
                        self.prompts_alpha.grad = self.prompts_alpha.grad + (1 / (self.args.sample_size - 1)) * (loss_list[k] - loss_avg) * derivative[k]

                    #torch.nn.utils.clip_grad_norm_(self.prompts_probs, 3)
                    #self.prompts_alpha.data = torch.clamp(self.prompts_alpha.data, min=1e-15)   # add clipping.  
                    self.prompt_optimizer.step()
                    #constrainScoreByWholeExact(self.prompts_probs)

                    self.completed_steps += 1
                    if self.completed_steps >= self.args.max_client_train_steps:
                        break
                
            except ApiCallLimitError:
                pass

        output_model =Parameters(tensors=[dill.dumps( self.prompts_alpha.clone())],tensor_type="")
        return FitRes(status=Status(
                    code=Code.OK,
                    message="Client fit",),
                parameters=output_model,
                num_examples=1,
                metrics={},)


    def evaluate(args,ins):
        return None