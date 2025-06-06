"""fedloca: A Flower Baseline."""
# myfl/strategy.py
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from omegaconf import DictConfig

class FedLoCAStrategy(fl.server.strategy.FedAvg):
    """
    自定义 FedLoCA 策略：继承 FedAvg，只覆盖 aggregate_fit，
    并在其中调用您原来在 Alg_FedLoCA.py 里的聚合逻辑。
    """
    def __init__(
        self,
        server_lr: float,
        server_momentum: float,
        evaluate_fn,
        on_fit_config_fn,
        fraction_fit: float,
        min_fit_clients: int,
        min_available_clients: int,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
        )
        self.server_lr = server_lr
        self.server_momentum = server_momentum
        self.momentum_vec: List[np.ndarray] = []
        self.round = 0

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Parameters:
        # Flower 在启动时会调用：装载初始全局模型参数
        parameters = super().initialize_parameters(client_manager)
        self.momentum_vec = [np.zeros_like(arr) for arr in parameters_to_ndarrays(parameters)]
        return parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: list[
            Tuple[fl.server.client_proxy.ClientProxy, fl.server.strategy.FitRes]
        ],
        failures,
    ) -> Tuple[Parameters, Dict[str, float]]:
        """
        覆写 FedAvg 的聚合：先做普通 FedAvg，得到 weighted_avg_parameters；
        然后计算 FedLoCA 特有的“局部梯度投影 + 动量更新”并修正 server 权重。
        具体逻辑在 FedLoCA_Aggregator 中实现，可直接调用。
        """
        self.round = server_round
        if not results:
            return None, {}

        # 1. 提取各客户端上传的参数及样本数
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # 2. 普通 FedAvg 得到 new_params
        fedavg_ndarrays = []
        total_examples = sum(num_examples for _, num_examples in weights_results)
        for layer_i in range(len(weights_results[0][0])):
            # 按样本数加权求和
            weighted_sum = sum(
                w[layer_i] * (num_examples / total_examples)
                for w, num_examples in weights_results
            )
            fedavg_ndarrays.append(weighted_sum)
        # 转成 Parameters
        fedavg_params: Parameters = ndarrays_to_parameters(fedavg_ndarrays)

        # 3. 调用您原来的 FedLoCA 聚合器计算最终更新
        #    例如：FedLoCA_Aggregator(inputs...) 返回一个 List[np.ndarray]
        fedloca_corrected_ndarrays = FedLoCA_Aggregator(
            round_num=server_round,
            fedavg_ndarrays=fedavg_ndarrays,
            old_params=parameters_to_ndarrays(fedavg_params),  # 上一轮模型
            client_weights=weights_results,
            server_lr=self.server_lr,
            server_momentum=self.server_momentum,
            momentum_vec=self.momentum_vec,
        )
        # FedLoCA_Aggregator 应当更新 self.momentum_vec，并返回新的权重列表

        # 4. 更新 self.momentum_vec（假设 Federator 已经在内部做了动量更新）
        #    如果 Federator 返回的 tuple 包含更新后的动量，也可在这里赋值

        # 5. 将 corrected 参数列表转成 Flower Parameters
        aggregated_parameters: Parameters = ndarrays_to_parameters(fedloca_corrected_ndarrays)

        # 6. 如果需要聚合 metrics，可和 super().aggregate_fit 一样做一次
        metrics_aggregated = {}
        if results:
            # 简单地把客户端上的“loss”“accuracy”做加权平均
            # 这里只演示如何取第一个客户端的 metrics 作为全局指标
            first_metrics = results[0][1].metrics.get("accuracy")
            metrics_aggregated["accuracy"] = first_metrics or 0.0

        return aggregated_parameters, metrics_aggregated


def FedLoCA_Aggregator(
    round_num: int,
    fedavg_ndarrays: List[np.ndarray],
    old_params: List[np.ndarray],
    client_weights: List[ Tuple[List[np.ndarray], int] ],
    server_lr: float,
    server_momentum: float,
    momentum_vec: List[np.ndarray]
) -> List[np.ndarray]:
    """
    原来您在 fed_finetune.py 里实现的 FedLoCA 聚合逻辑。
    输入：
      - round_num: int，当前的联邦轮数
      - fedavg_ndarrays: List[np.ndarray]，普通 FedAvg 得到的参数列表
      - old_params: List[np.ndarray]，上一轮的全局参数列表
      - client_weights: [(ndarrays, num_examples), ...]，每个客户端上传的模型权重及样本数
      - server_lr: float，服务端学习率
      - server_momentum: float，服务端动量超参
      - momentum_vec: List[np.ndarray]，上一轮的动量向量
    输出：
      - new_params: List[np.ndarray]，更新后的参数列表
    您可以直接把原来脚本里“FedLoCA 具体算子”贴到这里，不做其它改动即可。
    """
    # 下面只是一个示意，您原先的真实实现可能更复杂
    # 1. 计算“伪梯度”：pseudo_grad = old_params - fedavg_ndarrays
    pseudo_grad = [old - new for old, new in zip(old_params, fedavg_ndarrays)]
    # 2. 更新动量：momentum_vec = server_momentum * momentum_vec + pseudo_grad
    new_momentum = [
        server_momentum * m + g for m, g in zip(momentum_vec, pseudo_grad)
    ]
    # 3. Nesterov 纠正： corrected_grad = pseudo_grad + server_momentum * new_momentum
    corrected_grad = [
        g + server_momentum * m for g, m in zip(pseudo_grad, new_momentum)
    ]
    # 4. 更新全局参数：new_params = old_params - server_lr * corrected_grad
    new_params = [
        old - server_lr * cg for old, cg in zip(old_params, corrected_grad)
    ]
    # 5. 返回新的参数，并把 momentum_vec 更新给策略
    #    注意：策略中需要接收新的 new_momentum，以便下一轮继续使用
    #    如果需要同时传回动量，也可以改为返回 (new_params, new_momentum)
    momentum_vec[:] = new_momentum
    return new_params

