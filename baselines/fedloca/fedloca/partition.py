# myfl/partition.py
import numpy as np
from typing import Tuple, List
from collections import defaultdict


def create_lda_partitions(
    dataset: Tuple[np.ndarray, np.ndarray],
    num_partitions: int,
    concentration: float,
    seed: int = 1234,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    基于 Dirichlet 分布的 LDA 非 IID 分区，复用您原来 partition.py 里的逻辑。
    输入：dataset = (x_all, y_all)，num_partitions = num_clients，
         concentration = Dirichlet 分布浓度系数。
    输出：length = num_partitions 的列表，每项 (x_part, y_part)。
    """
    x_all, y_all = dataset
    num_classes = len(set(y_all.tolist()))
    np.random.seed(seed)

    # 1. 按 label 索引排序，便于后续按照 Dirichlet 采样
    idx_by_label = defaultdict(list)
    for idx, label in enumerate(y_all):
        idx_by_label[int(label)].append(idx)

    # 2. 对每个类别，用 Dirichlet 分布生成一个长度为 num_partitions 的向量，
    #    然后把该类别样本分配给各 partition。
    partitions_indices = {i: [] for i in range(num_partitions)}
    for label, idx_list in idx_by_label.items():
        np.random.shuffle(idx_list)
        # 为这个类别生成 Dirichlet 向量
        alpha = [concentration] * num_partitions
        proportions = np.random.dirichlet(alpha)
        # 按比例分配样本
        cumulative = (np.cumsum(proportions) * len(idx_list)).astype(int)
        prev = 0
        for part_id, cut in enumerate(cumulative):
            selected = idx_list[prev:cut]
            partitions_indices[part_id].extend(selected)
            prev = cut

    # 3. 将 indices 转成真实的 (x, y) 对
    partitions = []
    for i in range(num_partitions):
        part_idx = partitions_indices[i]
        x_part = x_all[part_idx]
        y_part = y_all[part_idx]
        partitions.append((x_part, y_part))

    return partitions
