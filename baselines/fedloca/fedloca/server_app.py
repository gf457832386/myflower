"""Phoebe: A Flower Baseline."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Server, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fedloca.model import Net, get_weights
from omegaconf import OmegaConf
cfg = OmegaConf.load("fedloca/conf/base.yaml")


#服务器流程：包括获得参数，聚合
def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    num_rounds = cfg.num_rounds
    fraction_fit = cfg.train_ratio

    # Initialize model parameters
    ndarrays = model_to_parameters
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=float(fraction_fit),
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        
    )
    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)