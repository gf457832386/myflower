# main.py
import pickle
from pathlib import Path

import flwr as fl
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from dataset import twenty_newsgroup
from partition import create_lda_partitions
from client_app import generate_client_fn
from server_app import get_on_fit_config, get_evaluate_fn
from strategy import FedLoCAStrategy

@hydra.main(config_path="conf", config_name="base")
def main(cfg: DictConfig) -> None:

    #创建文件夹
    # 取出模型保存路径和结果保存路径
    model_dir = Path(cfg.model_path)
    results_dir = Path(cfg.results_path)

    # 如果目录不存在，就递归创建
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 固定随机种子，保证可重现
    np.random.seed(cfg.dataset.seed)

    # 1. 打印完整配置，方便调试
    print(OmegaConf.to_yaml(cfg))

    # 2. 加载并划分处理数据集



    x_all, y_all, x_test, y_test, input_shape, num_classes = instantiate(cfg.dataset)
    # 3. 非 IID 划分：得到 length = num_clients 的列表，每项 (x_part, y_part)
    partitions = create_lda_partitions((x_all, y_all), cfg.num_clients, cfg.noniid.concentration, cfg.dataset.seed)
    print(f">>> Data partition finished: {cfg.num_clients} clients, each share ~{len(partitions[0][0])} samples.")

    # 4. 生成 Flower 客户端函数
    client_fn = generate_client_fn(partitions, cfg.model, num_classes, device="cpu")

    # 5. 构造服务端评估函数：需要一个裸模型实例（带 LoRA）
    model_for_eval = instantiate(cfg.model)
    evaluate_fn = get_evaluate_fn(
        model=model_for_eval,
        x_test=x_test,
        y_test=y_test,
        num_rounds=cfg.num_rounds,
        num_classes=num_classes,
        device="cpu",
    )

    # 6. 构造 on_fit_config_fn：每轮分发给客户端的超参
    on_fit_config_fn = get_on_fit_config(cfg)

    # 7. 实例化自定义 FedLoCA 策略
    strategy = FedLoCAStrategy(
        server_lr=cfg.server.server_lr,
        server_momentum=cfg.server.server_momentum,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        fraction_fit=cfg.strategy.fraction_fit,
        min_fit_clients=cfg.strategy.min_fit_clients,
        min_available_clients=cfg.strategy.min_available_clients,
    )

    # 8. 启动 Flower Simluation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.num_cpus, "num_gpus": cfg.num_gpus},
    )

    # 9. 保存结果
    final_acc = history.metrics_centralized["accuracy"][-1][1]
    save_dir = HydraConfig.get().runtime.output_dir
    strategy_name = strategy.__class__.__name__
    dataset_name = "20Newsgroup"
    filename = f"results_{strategy_name}_{dataset_name}_clients{cfg.num_clients}_rounds{cfg.num_rounds}_acc{final_acc:.4f}.pkl"
    results_path = Path(save_dir) / filename
    with open(results_path, "wb") as f:
        pickle.dump({"history": history}, f)
    print(f">>> Results saved to {results_path}")

if __name__ == "__main__":
    main()
