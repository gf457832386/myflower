"""fedbpt: A Flower / PyTorch app."""

from logging import DEBUG, INFO, log
from typing import  List, Tuple, Union
import copy
import cma
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
import random


class FedBPTClusterStrategy(Strategy):
    def __init__(self, args,start_round: int = 0,frac = 1):
        
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

    def initialize_parameters(self, client_manager):
        global_model = es2parameters(self.global_es)
        return global_model

    def configure_fit(self, server_round, parameters, client_manager):
        
        # 使用之前缓存的多套参数，为每个客户端随机选择一组
        clients = client_manager.sample(
            num_clients=self.num_clients, min_num_clients=self.min_num_clients
        )

        # 确保模型数量不超过客户端数
        model_pool = self.cross_models * (len(clients) // len(self.cross_models) + 1)
        random.shuffle(model_pool)

        ins = []
        for i, client in enumerate(clients):
            chosen_model = model_pool[i]  # 每个客户端拿不同的模型（重复利用）
            ins.append((
                client,
                FitIns(
                    parameters=chosen_model,
                    config={"dim": self.intrinsic_dim, "current_round": server_round}
                )
            ))

        # # 随机分配交叉聚合模型
        # ins = []
        # for client in clients:
        #     model_idx = random.randint(0, len(self.cross_models) - 1)
        #     chosen_model = self.cross_models[model_idx]
        #     ins.append((
        #         client,
        #         FitIns(
        #             parameters=chosen_model,
        #             config={"dim": self.intrinsic_dim, "current_round": server_round}
        #         )
        #     ))
        return ins
    
    def aggregate_fit(self, server_round, results, failures):
        local_means=[]
        local_Cs = []
        local_sigmas = []
        local_pcs = []


        # get solutions from clients
        for crt in results:
            params = parameters2result(crt[1].parameters)
            local_means.append(params[0])
            local_Cs.append(params[1])
            local_sigmas.append(params[2])
            local_pcs.append(params[3])

        # Global update
        log(INFO,f"Received {len(local_means)} local_es from clients")

        # 计算平均值
        num_clients = len(local_means)
        num_models = min(3, num_clients)  # 交叉聚合生成几套模型

        self.cross_models = []  # 用于下轮configure_fit

        for i in range(num_models):
            subset_size = max(1, num_clients // 2)  # 每次随机选择一半的客户端
            idxs = random.sample(range(num_clients), k=subset_size)
            cross_mean = np.mean([local_means[j] for j in idxs], axis=0)
            cross_C = np.mean([local_Cs[j] for j in idxs], axis=0)
            cross_sigma = np.mean([local_sigmas[j] for j in idxs])
            cross_pc = np.mean([local_pcs[j] for j in idxs], axis=0)

            # 构造临时 CMA 对象，用于序列化参数
            tmp_es = copy.deepcopy(self.global_es)
            tmp_es.mean = cross_mean
            tmp_es.C = cross_C
            tmp_es.sigma = cross_sigma
            tmp_es.pc = cross_pc

            # 转为参数用于下发
            global_model = es2parameters(tmp_es)
            self.cross_models.append(global_model)

        # 保存主模型（第0个）作为server prompt记录
        self.global_es.mean = local_means[0]
        self.global_es.C = local_Cs[0]
        self.global_es.sigma = local_sigmas[0]
        self.global_es.pc = local_pcs[0]

        self.server_prompts.append(copy.deepcopy(self.global_es.mean))

        return self.cross_models[0], {"current_round": server_round}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return None
    def aggregate_evaluate(self, server_round, results, failures):
        return None

    def evaluate(self, server_round, parameters):
        return None




