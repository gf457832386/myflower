from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, Scalar
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from typing import List, Tuple, Union
import numpy as np

class Fedloca(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[Union[BaseException, Tuple]],
    ) -> Tuple[Parameters, dict[str, Scalar]]:
        # Step 1: 过滤掉失败的客户端
        if not results:
            return None, {}
        
        # Step 2: 提取 LoRA 参数并按层聚合
        param_list = []
        num_examples_list = []

        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            param_list.append(weights)
            num_examples_list.append(fit_res.num_examples)

        # Step 3: Layer-wise 聚合
        num_layers = len(param_list[0])
        aggregated = []
        for layer_idx in range(num_layers):
            layer_weights = np.array([
                client[layer_idx] * num_examples_list[i]
                for i, client in enumerate(param_list)
            ])
            summed = np.sum(layer_weights, axis=0)
            total_examples = sum(num_examples_list)
            aggregated_layer = summed / total_examples
            aggregated.append(aggregated_layer)

        # Step 4: 返回聚合后的结果
        aggregated_parameters = ndarrays_to_parameters(aggregated)
        return aggregated_parameters
