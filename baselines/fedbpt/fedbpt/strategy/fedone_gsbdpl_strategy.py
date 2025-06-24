"""fedbpt: A Flower / PyTorch app."""

from logging import DEBUG, INFO, log
from typing import  List, Tuple, Union
import copy
import cma
import dill
from flwr.common import FitRes,FitIns
from flwr.server.strategy import Strategy
import numpy as np
from flwr.server.client_proxy import ClientProxy
from ..LMForwardAPI import LMForwardAPI
from ..utils import parameters2result,es2parameters
FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]

from flwr.common import FitIns,Parameters
class GumbelBDPLStrategy(Strategy):
    def __init__(self, args,start_round: int = 0,frac = 1):
        
        self.average_theta

    def initialize_parameters(self, client_manager):
        global_model = es2parameters(self.global_es)
        return global_model

    def configure_fit(self, server_round, parameters, client_manager):
        global_model = Parameters(tensor_type="",tensors=[dill.dumps(self.average_theta)])
        clients = client_manager.sample(
            num_clients=self.num_clients, min_num_clients=self.min_num_clients
        )
        # Return client/config pairs
        
        ins = [(client,FitIns(parameters=global_model,config={"dim":self.intrinsic_dim,"current_round":server_round})) for client in clients]
        return ins
        for epoch in range(args.num_train_epochs):
            # training. 
            client_prompts_probs_list = []
            client_dataset_len_list = []
            for client_idx in random.sample(range(args.num_clients), args.num_activated_clients):
                # Each client train and update.  
                client_prompts_probs = client_list[client_idx].local_training(args, model, tokenizer, average_theta, tracker)
                client_prompts_probs_list.append(client_prompts_probs) #print("client_prompts_probs: \n", client_prompts_probs)
                # get the weight for averaging. 
                client_dataset_len_nk = client_list[client_idx].get_len_dataset()
                client_dataset_len_list.append(client_dataset_len_nk) #print("weight: \n", weight)

            # Fed Average.
            sampled_client_dataset_len_sum_mt = sum(client_dataset_len_list) 
            average_theta = sum(nk/sampled_client_dataset_len_sum_mt * tensor for nk, tensor in zip(client_dataset_len_list, client_prompts_probs_list)) 
            if args.prompt_tuning_method == "prompt-tuning":
                model.prompt_encoder.default.embedding.weight.data = average_theta
            elif args.prompt_tuning_method == "prefix-tuning":
                model.trainable_params.data = average_theta 

            #print(average_theta)

            print(f"eval: {eval_result}")
            if eval_result >= best_eval_result:
                best_eval_result = eval_result
                best_theta = average_theta.clone().detach()
                if args.prompt_tuning_method == "GumbelBDPL":
                    best_prompt_prob = eval_prompt_prob.clone().detach()
                print("best theta")
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()
            if train_api_request.count >= args.api_limit:
                break
            #print(average_theta[0])

            # early stop. 
            if args.early_stop > 0:
                if eval_result > args.early_stop:
                    break
    
    def aggregate_fit(self, server_round, results, failures):
        global_solutions = []
        global_fitnesses = []
        client_sigma_list = []


        # get solutions from clients
        for crt in results:
            result = parameters2result(crt[1].parameters,crt[1].num_examples,self.args.local_iter)
            global_solutions.append(result["solutions"])
            global_fitnesses.append(result["fitnesses"])
            log(INFO, "fit_round %s: fitness=%s test acc = %s", server_round,global_fitnesses[-1],crt[1].metrics['test acc'])
            client_sigma_list.append(np.sum(np.array(result["local_sigmas"]) ** 2))
            local_cma_mu = result["local_cma_mu"]

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




