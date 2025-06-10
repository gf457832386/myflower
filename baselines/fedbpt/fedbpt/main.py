import copy
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import hydra
import flwr as fl
from flwr.server.client_manager import SimpleClientManager



from client import *
from server import *
from utils import save_results_as_pickle
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg.onnx_model_path = eval(cfg.onnx_model_path)
    cfg.note = eval(cfg.note)
    cfg.init_score_path = eval(cfg.init_score_path)
    print(OmegaConf.to_yaml(cfg))

    # Initialize data processor

    data_processor = data_process.data_processor(cfg)


    data_bundle = data_processor.get_data()
    if cfg.task_name in ["agnews", "yelpp", "dbpedia", "snli"]:
        train_data, test_data = data_bundle.get_dataset("train"), data_bundle.get_dataset("test")
    else:
        train_data, test_data = data_bundle.get_dataset("train"), data_bundle.get_dataset("validation")

    # 提取少量代表性样本，返回下标
    train_data, dev_data = construct_true_few_shot_data(cfg, train_data, cfg.k_shot)

    # 填充数据，确保长度一致，并设置对应掩码
    for ds in [train_data, dev_data, test_data]:
        ds.set_pad_val(
            "input_ids", data_processor.tokenizer.pad_token_id if data_processor.tokenizer.pad_token_id is not None else 0
        )
        ds.set_pad_val("attention_mask", 0)
    print("# of train data: {}".format(len(train_data)))
    print("Example:")
    print(train_data[0])
    print("\n# of dev data: {}".format(len(dev_data)))
    print("Example:")
    print(dev_data[0])
    print("\n# of test data: {}".format(len(test_data)))
    print("Example:")
    print(test_data[0])

    # Split dataset，根据num_users分
    user_dict_train, user_dict_dev = split_data(cfg, train_data, dev_data)

    # prepare function that will be used to spawn each client
    client_fn = gen_client_fn(
        args=cfg,train_data=train_data,dev_data=dev_data,test_data=test_data,user_dict_train=user_dict_train,user_dict_dev=user_dict_dev
    )

    # get function that will executed by the strategy's evaluate() method

    # get a function that will be used to construct the config that the client's
    # that are only defined at run time.



    # Start simulation
    strategy = FedBPTStrategy(cfg,test_data=test_data)
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        server=ServerFedBPT(
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