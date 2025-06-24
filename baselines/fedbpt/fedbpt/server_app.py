"""fedbn: A Flower Baseline."""

import json
import random
import flwr as fl
import numpy as np
from omegaconf import DictConfig
from flwr.common import Context
from flwr.server import (
    ServerApp,
    ServerAppComponents,
    SimpleClientManager,
)
from .data_process import construct_true_few_shot_data, split_data,data_processor
from flwr.server.server import Server
import hydra
from transformers import RobertaTokenizer
import torch
from .strategy.fedbpt_strategy import FedBPTStrategy
from .strategy.fedavgbbpt_strategy import FedAvgBBTStrategy
from .strategy.fedbpt_dg_strategy import FedBPTDGStrategy
from .utils import runcfg2args
def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from context
    print("### BEGIN: RUN CONFIG ####")
    run_config = context.run_config
    print(json.dumps(run_config, indent=4))
    print("### END: RUN CONFIG ####")
    args = runcfg2args(run_config)
    config=fl.server.ServerConfig(num_rounds=args.num_rounds)

    # Define Strategy
    if run_config['strategy']=="fedbpt":
        strategy = FedBPTStrategy(args)
    elif run_config['strategy']=="fedavgbbt":
        strategy = FedAvgBBTStrategy(args)
    elif run_config['strategy']=="fedbpt_dg":
        dp = data_processor(args)
        data_bundle = dp.get_data()
        if args.task_name in ["agnews", "yelpp", "dbpedia", "snli"]:
            train_data, test_data = data_bundle.get_dataset("train"), data_bundle.get_dataset("test")
        else:
            train_data, test_data = data_bundle.get_dataset("train"), data_bundle.get_dataset("validation")
        labels=[]
        # 填充数据，确保长度一致，并设置对应掩码
        for ds in [train_data, test_data]:
            ds.set_pad_val("attention_mask", 0)
            labels+=list(ds['labels'])
        tokenizers=RobertaTokenizer.from_pretrained("roberta-large")
        labels = set(labels)
        strategy = FedBPTDGStrategy(args,500,labels,tokenizers.vocab_size)
    else:
        strategy = FedBPTStrategy(args)
    server=Server(client_manager=SimpleClientManager(), strategy=strategy)
    return ServerAppComponents(server=server, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
