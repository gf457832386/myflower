"""fedloca: A Flower Baseline."""
# myfl/strategy.py
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from omegaconf import OmegaConf
cfg = OmegaConf.load("fedloca/conf/base.yaml")

class FedLoCAStrategy(FedAvg):
    """
    自定义 FedLoCA 策略：继承 FedAvg，只覆盖 aggregate_fit，
    并在其中调用您原来在 Alg_FedLoCA.py 里的聚合逻辑。
    """
  
    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Parameters:
        # Flower 在启动时会调用：装载初始全局模型参数
        parameters = super().initialize_parameters(client_manager)
        self.momentum_vec = [np.zeros_like(arr) for arr in parameters_to_ndarrays(parameters)]
        return parameters

    # 覆写 FedAvg 的聚合逻辑
    def aggregate_fit(  
        self
        ):
        results=None

     
        
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

        

        return fedavg_params
