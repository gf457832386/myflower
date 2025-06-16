import copy
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import hydra
import flwr as fl
from flwr.server.server import Server
from flwr.server.client_manager import SimpleClientManager
from .utils import save_results_as_pickle
from .client.fedbpt_client import gen_client_fn
from .strategy.fedbpt_strategy import FedBPTStrategy
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg.onnx_model_path = eval(cfg.onnx_model_path)
    cfg.note = eval(cfg.note)
    cfg.init_score_path = eval(cfg.init_score_path)
    print(OmegaConf.to_yaml(cfg))

    # prepare function that will be used to spawn each client
    client_fn = gen_client_fn(
        args=cfg
    )

    # get function that will executed by the strategy's evaluate() method

    # get a function that will be used to construct the config that the client's
    # that are only defined at run time.

    # Start simulation
    strategy = FedBPTStrategy(cfg)
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        server=Server(
            client_manager=SimpleClientManager(), strategy=strategy
        ),
    )

    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results={})


if __name__ == "__main__":
    main()