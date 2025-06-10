"""fedbpt: A Flower / PyTorch app."""

from logging import DEBUG, INFO, log
import pdb
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.server import Server, fit_clients
import copy
import cma
from flwr.common import Context, ndarrays_to_parameters,Parameters,FitRes,FitIns,EvaluateIns
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy
import numpy as np
from flwr.server.history import History
import timeit
from flwr.server.client_proxy import ClientProxy
from LMForwardAPI import LMForwardAPI
import data_process
from utils import parameters2es,result2parameters,parameters2result,es2parameters,Model
FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]


class FedBPTStrategy(Strategy):
    def __init__(self, args,test_data,start_round: int = 0,frac = 1):
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
        self.test_data = test_data
        # self.parameters = es2parameters(self.global_es)
        self.local_sigma_current = self.global_es.sigma

    def initialize_parameters(self, client_manager):
        global_model = es2parameters(self.global_es)
        return global_model

    def configure_fit(self, server_round, parameters, client_manager):
        
        global_model = es2parameters(self.global_es)
        clients = client_manager.sample(
            num_clients=self.num_clients, min_num_clients=self.min_num_clients
        )
        # Return client/config pairs
        
        res = [(client,FitIns(parameters=global_model,config={"dim":self.intrinsic_dim,"current_round":server_round})) for client in clients]
        return res
    # def initialize_parameters(self, client_manager):
    #     global_model = Model(params={"global_es": self.global_es}, current_round=0,metrics={},tensor_type=int,tensors=[])
    #     return global_model

    # def configure_fit(self, server_round, parameters, client_manager):
        
    #     global_model = Model(params={"global_es": self.global_es}, current_round=server_round,metrics={},tensor_type=int,tensors=[])
    #     clients = client_manager.sample(
    #         num_clients=self.num_clients, min_num_clients=self.min_num_clients
    #     )
    #     # Return client/config pairs
        
    #     res = [(client,FitIns(parameters=global_model,config={})) for client in clients]
    #     return res
    
    def aggregate_fit(self, server_round, results, failures):
        global_solutions = []
        global_fitnesses = []
        client_sigma_list = []


        # get solutions from clients
        for crt in results:
            result = parameters2result(crt[1].parameters,crt[1].num_examples,self.args.local_iter)
            # result = crt.parameters.params
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
        
        # clients = client_manager.sample(
        #     num_clients=self.num_clients, min_num_clients=self.min_num_clients
        # )
        # # Return client/config pairs
        
        # res = [(client,EvaluateIns(parameters=Parameters(tensor_type="",tensors=[]),config={})) for client in clients]
        # return res
        return None
    def aggregate_evaluate(self, server_round, results, failures):
        for crt in results:
            log(INFO, "evaluate_round %s: accuracy=%s", server_round,crt[1].metrics['accuracy'])
        return 0,{}

    def evaluate(self, server_round, parameters):
        test_acc = self.model_forward_api.eval(prompt_embedding=self.global_es.mean, test_data=self.test_data)
        return (None,{"accuracy":test_acc})

def server_fn(args,test_data):
    # Read from config
    num_rounds = args.num_rounds


    # Define strategy
    strategy = FedBPTStrategy(
        args,test_data
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


class ServerFedBPT(Server):
    """Sever for FedDyn."""

    def __init__(self, *, client_manager, strategy = None):
        super().__init__(client_manager=client_manager, strategy=strategy)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], FitResultsAndFailures]
    ]:
        """Perform a single round."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )
        
        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )
        # print(len(results),failures)
        aggregated_result = FedBPTStrategy.aggregate_fit(
            self.strategy,
            server_round,
            results,
            failures,
        )
        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)


