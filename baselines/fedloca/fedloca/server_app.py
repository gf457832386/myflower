"""Phoebe: A Flower Baseline."""

from flwr.common import Context, ndarrays_to_parameters
from omegaconf import OmegaConf
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from baselines.fedloca.fedloca.model import load_model
from baselines.fedloca.fedloca.strategy.alg_fedloca import Fedloca
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "conf/base.yaml"))
cfg = OmegaConf.load(base_path)


#服务器流程：包括获得参数，聚合
def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    num_rounds = cfg.num_rounds
    fraction_fit = cfg.train_ratio

    # Initialize model parameters
    model_parameters = load_model().get_weights()

    # Define strategy
    strategy = Fedloca(
        fraction_fit=float(fraction_fit),
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=model_parameters,
    )

    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)



