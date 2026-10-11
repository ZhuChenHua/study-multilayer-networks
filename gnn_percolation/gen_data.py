"""
在真实网络 Cora 上做随机渗流，用 BFS 求最大连通分量作标签，返回 PyG Data。

只做两件事：读拓扑 + 算标签。因为 Cora 自带的标签是论文主题，
而我们需要的是“攻击后是否还在巨连通分量”，只能模拟算出来。
"""

import os
import pickle
import sys
sys.path.append("E:\\project\\study-multilayer-networks\\paper_1\\code")

import torch
from torch_geometric.data import Data
from percolation_lecture import largest_component

_CORA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Cora", "raw", "ind.cora.graph")


def load_cora():
    """读取 Cora 邻接矩阵（对称、无自环，bool）。"""
    with open(_CORA, "rb") as f:
        g = pickle.load(f, encoding="latin1")   # {节点: [邻居]}
    N = len(g)
    src, dst = [], []
    for u, nbrs in g.items():
        for v in nbrs:
            src.append(u)
            dst.append(v)
    A = torch.zeros(N, N, dtype=torch.bool)
    A[torch.tensor(src), torch.tensor(dst)] = True
    A = A | A.T
    A.fill_diagonal_(False)
    return A


def make_data(q=0.55, seed=42, noise=0.05):
    """随机移除 q 比例节点，返回带 train/val/test mask 的 Data。"""
    torch.manual_seed(seed)
    A = load_cora()
    N = A.shape[0]

    alive = torch.rand(N) >= q              # 谁幸存
    giant = largest_component(A, alive)     # 标签：是否在最大连通分量

    deg = A.sum(1).float()
    deg = deg / deg.max()
    x = torch.stack([deg + noise * torch.randn(N), alive.float(), torch.zeros(N)], dim=1)

    data = Data(x=x, edge_index=A.nonzero().t().contiguous(), y=giant.long())

    # 只在幸存节点上划分
    idx = torch.where(alive)[0]
    perm = idx[torch.randperm(len(idx))]
    n_tr, n_va = int(0.3 * len(perm)), int(0.2 * len(perm))
    data.train_mask = _mask(N, perm[:n_tr])
    data.val_mask = _mask(N, perm[n_tr:n_tr + n_va])
    data.test_mask = _mask(N, perm[n_tr + n_va:])
    data.alive = alive
    data.giant = giant
    return data


def _mask(N, idx):
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    return m


if __name__ == "__main__":
    d = make_data()
    print(d)
    print("giant/alive =", round(d.giant.sum().item() / d.alive.sum().item(), 3))
