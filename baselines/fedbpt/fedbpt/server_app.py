"""fedbn: A Flower Baseline."""

import json
import flwr as fl
from omegaconf import DictConfig
from flwr.common import Context
from flwr.server import (
    ServerApp,
    ServerAppComponents,
    SimpleClientManager,
)
from flwr.server.server import Server
import hydra
from .strategy.fedbpt_strategy import FedBPTStrategy
from .strategy.fedavgbbpt_strategy import FedAvgBBTStrategy
from .utils import runcfg2args
def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from context
    print("### BEGIN: RUN CONFIG ####")
    run_config = context.run_config
    print(json.dumps(run_config, indent=4))
    print("### END: RUN CONFIG ####")
    cfg = runcfg2args(run_config)
    config=fl.server.ServerConfig(num_rounds=cfg.num_rounds)

    # Define Strategy
    if run_config['strategy']=="fedbpt":
        strategy = FedBPTStrategy(cfg)
    elif run_config['strategy']=="fedavgbbt":
        strategy = FedAvgBBTStrategy(cfg)
    else:
        strategy = FedBPTStrategy(cfg)
    server=Server(client_manager=SimpleClientManager(), strategy=strategy)
    return ServerAppComponents(server=server, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
