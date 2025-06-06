"""Phoebe: A Flower Baseline."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from datasets import load_dataset


FDS = None  # Cache FederatedDataset


def twenty_newsgroup(
    data_path: str,
    val_set_size: float,
    test_set_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple, int]:
    """
    通过 HuggingFace datasets 加载 20Newsgroup 数据集，
    并根据 val_set_size/test_set_size 进行 train/val/test 划分，
    最后返回 (X_train, y_train, X_test, y_test, input_shape, num_classes)。
    """
    # 1. 从原始代码中复用 data_processers.py 里的数据预处理函数
    from baselines.fedloca.fedloca.data_processers import preprocess_20newsgroup

    # 假设 preprocess_20newsgroup 返回一个字典 {'train':DatasetDict, 'test':DatasetDict}
    dataset = load_dataset("newsgroup", "20news-bydate")
    # 调用您自己 data_processers.py 里的预处理
    processed = preprocess_20newsgroup(dataset)

    # processed["train"] 已包含分词、截断、encode 等逻辑，类似您原来做的
    # processed["test"] 是原始测试集
    train_val = processed["train"].train_test_split(
        test_size=val_set_size, seed=seed
    )
    dataset_train = train_val["train"]  # ~ (1 - val_set_size) 部分
    dataset_val = train_val["test"]     # ~ val_set_size 部分
    dataset_test = processed["test"]    # ~全部测试集

    print("Loaded 20Newsgroup dataset:")
    print("Train set size:", len(dataset_train))
    print("Val set size:  ", len(dataset_val))
    print("Test set size: ", len(dataset_test))

    # 将 HF Dataset 转成 numpy 数组或 torch.Tensor（根据后续 client.py 接受的格式）
    # 这里假定 tokenizer 已经在 preprocess_20newsgroup 中统一做过 encode，
    # 且每个样本 item["input_ids"] 是长度为 cutoff_len 的 list[int]，
    # item["label"] 是 0~19 之间的整数 label。

    def dataset_to_numpy(ds):
        X = np.array(ds["input_ids"])
        y = np.array(ds["label"])
        return X, y

    x_train, y_train = dataset_to_numpy(dataset_train)
    x_val, y_val = dataset_to_numpy(dataset_val)
    x_test, y_test = dataset_to_numpy(dataset_test)

    # 2. 合并 train/val，为了后续分区先合并到一起，再在 client_fn 中 90%:10% 划分
    x_all = np.concatenate([x_train, x_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)

    # 3. 返回给 main.py 用于全局划分
    input_shape = list(x_all.shape[1:])  # e.g. [cutoff_len]
    num_classes = len(set(y_all.tolist()))
    return x_all, y_all, x_test, y_test, tuple(input_shape), num_classes


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = FDS.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader
