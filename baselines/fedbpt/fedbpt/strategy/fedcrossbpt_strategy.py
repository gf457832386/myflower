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


class FedCrossBPTStrategy(Strategy):
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
        self.next_generation_pool = []  # store new generation for next round

    def initialize_parameters(self, client_manager):
        return None
        # global_model = es2parameters(self.global_es)
        # return global_model

    def configure_fit(self, server_round, parameters, client_manager):
        
        clients = client_manager.sample(
            num_clients=self.num_clients, min_num_clients=self.min_num_clients
        )
        # ✅ 初始化 next_generation_pool（第一轮）
        if len(self.next_generation_pool) == 0:
            print(f"[Round {server_round}] Initializing next_generation_pool with {len(clients)} copies of global_es")
            dim = self.intrinsic_dim
            B = np.eye(dim)
            D = np.ones(dim)
            for _ in range(len(clients)):
                self.next_generation_pool.append({
                    "mean": copy.deepcopy(self.global_es.mean),
                    "sigma": self.global_es.sigma,
                    "B": B,
                    "D": D,
                })
        assert len(self.next_generation_pool) >= len(clients)

        random.shuffle(self.next_generation_pool)

        ins = []
        for i, client in enumerate(clients):
            cma_state = self.next_generation_pool[i]  # {"mean": ..., "sigma": ...}
            encoded = es2parameters(cma_state)        # 你需要确保支持字典序列化
            ins.append((client, FitIns(parameters=encoded, config={
                "dim": self.intrinsic_dim,
                "current_round": server_round
            })))
        return ins

    def aggregate_fit(self, server_round, results, failures):
        print(f"[Round {server_round}] {len(results)} successes, {len(failures)} failures.")
        

        # 提取 test acc 列表
        test_acc_list = [
            res.metrics["test acc"] for _, res in results
            if res.metrics and "test acc" in res.metrics
        ]
        # 计算平均 test acc
        avg_test_acc = sum(test_acc_list) / len(test_acc_list) if test_acc_list else 0.0
        print(f"[Round {server_round}] Average test acc from clients: {avg_test_acc:.4f}")
        # 打印最高的 test acc
        if test_acc_list:
            max_acc = max(test_acc_list)
            print(f"[Round {server_round}] Best test acc among clients: {max_acc:.4f}")
        else:
            print("No valid test acc reported.")
        print(test_acc_list)


        #服务器端聚合客户端的结果
        es_states = []  # 保存每个客户端返回的 {"mean": ..., "sigma": ...}
        fitnesses = []

        for i,(client, fitres) in enumerate(results):
            # fitres = crt[1]  # 解包二元组
            result = parameters2result(fitres.parameters, fitres.num_examples, self.args.local_iter)
            # result = parameters2result(crt[1].parameters, crt[1].num_examples, self.args.local_iter)
            mean = result["solutions"][0]
            sigma = result["local_sigmas"][-1]
            B = result.get("B", None)
            D = result.get("D", None)
            fitness = result["fitnesses"][0]

            print(f"\n[Round {server_round}] Client {i} upload:")
            print(f"  mean (norm): {np.linalg.norm(mean):.4f}")
            print(f"  sigma: {sigma:.4f}")
            if B is not None and D is not None:
                print(f"  B shape: {B.shape}, D shape: {D.shape}")
                print(f"  Covariance (approx) trace: {np.trace(B @ np.diag(D ** 2) @ B.T):.4f}")
            else:
                print("  B/D not provided by client.")
            print(f"  fitness: {fitness:.4f}")

            es_states.append({
                "mean": mean,
                "sigma": sigma,
                "B": B,
                "D": D,
            })
            fitnesses.append(fitness)

            # es_states.append({
            #     "mean": result["solutions"][0],         # shape=(intrinsic_dim,)
            #     "sigma": result["local_sigmas"][-1],     # 最后一个 sigma
            #     "B": result.get("B", None),
            #     "D": result.get("D", None)
            # })
            # fitnesses.append(result["fitnesses"][0])     # 对应 mean 的适应度

        fitnesses = np.array(fitnesses)
        sorted_indices = np.argsort(-fitnesses)
        top_k = len(sorted_indices) // 2
        top_indices = sorted_indices[:top_k]

        # 选出 top 50% 的 CMA-ES 状态
        # top_es_states = [es_states[i] for i in top_indices]
        top_es_states = [
            {
                "mean": es_states[i]["mean"],
                "sigma": es_states[i]["sigma"],
                "B": None,
                "D": None,
            }
            for i in top_indices
        ]

        # 两两随机交叉生成新的 CMA-ES 初始化参数
        offspring_states = []
        for _ in range(self.num_clients - top_k):
            p1, p2 = random.sample(top_es_states, 2)

            p1_index = None
            for j, state in enumerate(top_es_states):
                if np.allclose(state["mean"], p1["mean"]):
                    p1_index = j
                    break
            if p1_index is None:
                raise ValueError("Failed to find index of p1 in es_states")
            f1 = fitnesses[p1_index]

            p2_index = None
            for j, state in enumerate(top_es_states):
                if np.allclose(state["mean"], p2["mean"]):
                    p2_index = j
                    break
            if p2_index is None:
                raise ValueError("Failed to find index of p2 in es_states")
            f2 = fitnesses[p2_index]


            # f1 = fitnesses[sorted_indices.tolist().index(es_states.index(p1))]
            # f2 = fitnesses[sorted_indices.tolist().index(es_states.index(p2))]
            alpha = f1 / (f1 + f2 + 1e-8)
            # new_mean = alpha * p1["mean"] + (1 - alpha) * p2["mean"]
            # new_sigma = (p1["sigma"] + p2["sigma"]) / 2
            # 添加轻微扰动，避免陷入局部最优
            noise = np.random.randn(*p1["mean"].shape) * 0.01
            new_mean = alpha * p1["mean"] + (1 - alpha) * p2["mean"] + noise

            # 设置 sigma 下限，避免为 0
            new_sigma = max((p1["sigma"] + p2["sigma"]) / 2, 1e-3)
            # offspring_states.append({"mean": new_mean, "sigma": new_sigma})

            
            # 不保留任何协方差信息，完全由客户端自行估计
            new_state = {
                "mean": new_mean,
                "sigma": new_sigma,
                "B": None,
                "D": None,
            }
            offspring_states.append(new_state)

        # 更新下一轮 generation pool（共 num_clients 个 CMA-ES 初始状态）
        self.next_generation_pool = top_es_states + offspring_states

        print(f"\n[Round {server_round}] Next Generation Pool Summary:")
        for i, state in enumerate(self.next_generation_pool):
            mean = state["mean"]
            sigma = state["sigma"]
            B = state.get("B", None)
            D = state.get("D", None)
            print(f"  Candidate {i}:")
            print(f"    mean norm: {np.linalg.norm(mean):.4f}")
            print(f"    sigma: {sigma:.4f}")
            if B is not None and D is not None:
                print(f"    B shape: {B.shape}, D shape: {D.shape}")
                print(f"    Cov trace (approx): {np.trace(B @ np.diag(D ** 2) @ B.T):.4f}")
            else:
                print(f"    B/D not available.")

        print(f"[Round {server_round}] Selected top-{top_k}, generated {len(offspring_states)} offspring CMA states.")

        return None, {"current_round": server_round}


    def configure_evaluate(self, server_round, parameters, client_manager):
        return None
        # All clients will evaluate the global model
        clients = client_manager.all()
        return [(client, FitIns(parameters=parameters, config={})) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        return None
    
        acc_list = [res.metrics["accuracy"] for _, res in results if "accuracy" in res.metrics]
        avg_acc = sum(acc_list) / len(acc_list) if acc_list else 0.0
        print(f"Round {server_round} - Aggregated Global Accuracy from {len(acc_list)} clients: {avg_acc:.4f}")
        return 0.0, {"global_accuracy": avg_acc}

    def evaluate(self, server_round, parameters):
        return None


