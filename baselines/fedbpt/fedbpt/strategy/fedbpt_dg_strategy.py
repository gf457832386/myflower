"""fedbpt: A Flower / PyTorch app."""

from logging import DEBUG, INFO, log
from typing import  List, Tuple, Union
import copy
import cma
from flwr.common import FitRes,FitIns
from flwr.server.strategy import Strategy
import numpy as np
from flwr.server.client_proxy import ClientProxy
import torch
from ..LMForwardAPI import LMForwardAPI
from ..utils import parameters2result,es2parameters
FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
from ..data_process import perturb_dataset

class FedBPTDGStrategy(Strategy):
    def __init__(self, args,seq_length,labels,vocab_size,start_round: int = 0,frac = 1,dgfit_iter=20):
        
        self.frac = frac
        self.seed = args.seed
        self.sigma = args.sigma
        self.intrinsic_dim = args.intrinsic_dim
        self.bound = args.bound
        self.num_clients = args.num_clients
        self.min_num_clients = args.min_clients
        self.num_rounds = args.num_rounds
        self.start_round = start_round
        self.m = max(int(self.frac * self.num_clients), 1)
        self.args = args
        self.cma_opts = {
            "seed": self.seed,
            "popsize": self.m,
            "maxiter": self.num_rounds,  # args.epochs,
            "verbose": -1,
            "CMA_mu": self.m,
        }
        if self.bound > 0:
            self.cma_opts["bounds"] = [-1 * self.bound, 1 * self.bound]
        self.global_es = cma.CMAEvolutionStrategy(self.intrinsic_dim * [0], self.sigma, inopts=self.cma_opts)
        self.server_prompts = [copy.deepcopy(self.global_es.mean)]
        if args.cat_or_add == "add":
            init_prompt_path = None
        else:
            init_prompt_path = "./nli_base_prompt.pt"
        self.model_forward_api = LMForwardAPI(args=args, init_prompt_path=init_prompt_path)
        self.local_sigma_current = self.global_es.sigma

        
        self.num_classes = len(labels)
        self.labels = labels
        self.dgfit_iter = dgfit_iter
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data_generators = [cma.CMAEvolutionStrategy((seq_length+2) * [0], self.sigma, inopts={
            "seed": self.seed,
            "popsize": 10,
            "maxiter": self.num_rounds,  # args.epochs,
            "verbose": -1,
            "CMA_mu": self.m,
            "bounds":[0,vocab_size+1]
        }) for i in range(self.num_classes)]

    def initialize_parameters(self, client_manager):
        global_model = es2parameters(self.global_es)
        return global_model

    def configure_fit(self, server_round, parameters, client_manager):
        
        global_model = es2parameters(self.global_es)
        clients = client_manager.sample(
            num_clients=self.num_clients, min_num_clients=self.min_num_clients
        )
        # Return client/config pairs
        
        ins = [(client,FitIns(parameters=global_model,config={"dim":self.intrinsic_dim,"current_round":server_round})) for client in clients]
        return ins
    
    def data_generator_fit(self,solutions,nums):
        for i in range(self.num_classes):
            train_step = 0
            while train_step<self.dgfit_iter:
                data = self.data_generators[i].ask()
                fitnesses = []
                for d in data:
                    train_data = {
                    "input_ids": torch.tensor(np.mod(np.round([d[:-2]]),self.vocab_size), dtype=torch.long),
                    "attention_mask": torch.tensor([[1]*self.seq_length], dtype=torch.long),
                    "mask_pos": torch.tensor([int(d[-2])%self.seq_length], dtype=torch.long),
                    "labels": torch.tensor(self.labels[i], dtype=torch.long),
                    }
                    self.model_forward_api.set_dataset(train_data, train_data, train_data)
                    losses = []
                    for x in solutions:
                        loss = self.model_forward_api.eval(x)
                        losses.append(loss)
                    n_sum = sum(nums)
                    fitness = 0
                    for i in range(len(nums)):
                        fitness+=nums[i]/n_sum*losses[i]
                    fitnesses.append(fitness)
                self.data_generators[i].tell(data,fitnesses)
                train_step+=1

    def global_data_fit(self):
        for i in range(self.num_classes):
            data = self.data_generators[i].ask()
            input_ids = []
            attention_masks= []
            mask_pos =[]
            labels=[]
            for d in data:
                input_ids.append(np.mod(np.round(d[:-2]),self.vocab_size))
                attention_masks.append([1]*self.seq_length)
                mask_pos.append(int(d[-2])%self.seq_length)
                labels.append(self.labels[i])
            train_data = {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
                    "mask_pos": torch.tensor(mask_pos, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    }
            train_data_aux = perturb_dataset(self.args, train_data, self.model_forward_api.config)
            self.model_forward_api.set_dataset(train_data, train_data, train_data_aux)
            train_step = 0
            while train_step<self.dgfit_iter:
                solutions = self.global_es.ask()
                if self.args.perturb != 0:
                    fitnesses = [self.model_forward_api.eval(x) / self.model_forward_api.eval_perturb(x) for x in solutions]
                else:
                    fitnesses = [self.model_forward_api.eval(x) for x in solutions]
                self.global_es.tell(solutions,fitnesses)
                train_step+=1

    def aggregate_fit(self, server_round, results, failures):
        global_solutions = []
        global_fitnesses = []
        client_sigma_list = []
        nums = []

        # get solutions from clients
        for crt in results:
            result = parameters2result(crt[1].parameters,crt[1].num_examples,self.args.local_iter)
            global_solutions.append(result["solutions"])
            global_fitnesses.append(result["fitnesses"])
            log(INFO, "fit_round %s: fitness=%s test acc = %s", server_round,global_fitnesses[-1],crt[1].metrics['test acc'])
            client_sigma_list.append(np.sum(np.array(result["local_sigmas"]) ** 2))
            local_cma_mu = result["local_cma_mu"]
            nums.append(result['local_data_num'])

        # self.data_generator_fit(global_solutions,nums)
        # Global update
        global_solutions = np.concatenate(global_solutions, axis=0)
        global_fitnesses = np.concatenate(global_fitnesses)
        log(INFO,f"Received {len(global_solutions)} solutions and {len(global_fitnesses)} fitnesses from clients")
        if len(global_solutions) != len(global_fitnesses):
            raise ValueError(
                f"Mismatch between {len(global_solutions)} solutions and {len(global_fitnesses)} fitnesses!"
            )

        # calculate global sigma
        global_sigma = np.sqrt(np.sum(np.array(client_sigma_list)) / self.m / local_cma_mu)
        if global_sigma is np.ndarray:
            global_sigma=global_sigma[0]
        self.global_es.sigma = global_sigma
        log(INFO,f"Check sigma before: {self.global_es.sigma}")
        global_sigma_old = self.global_es.sigma

        self.global_es.ask()
        self.global_es.tell(global_solutions, global_fitnesses)

        # self.global_data_fit()

        self.server_prompts.append(copy.deepcopy(self.global_es.mean))

        log(INFO,f"Check sigma after: {self.global_es.sigma}")
        global_sigma_new = self.global_es.sigma

        # set local sigma
        self.global_es.sigma = global_sigma_new / global_sigma_old * self.local_sigma_current

        self.local_sigma_current = self.global_es.sigma

        if self.global_es.sigma < 0.5:
            self.global_es.sigma = 0.5
            log(INFO,"Set sigma local: 0.5")
        if self.global_es.sigma > self.local_sigma_current:
            self.global_es.sigma = self.local_sigma_current
            log(INFO,"Set sigma local: not change")

        log(INFO,f"Check sigma local: {self.global_es.sigma}")
        global_model = es2parameters(self.global_es)
        return global_model,{ 'current_round':server_round}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return None
    def aggregate_evaluate(self, server_round, results, failures):
        return None

    def evaluate(self, server_round, parameters):
        return None




