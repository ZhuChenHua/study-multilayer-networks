"""
加载训练好的模型，在测试节点上评估。
"""

import os

import torch
from gen_data import make_data
from model import GCN


def main():
    data = make_data(q=0.55, seed=42)

    model = GCN(in_dim=3, hidden=64, out_dim=2, dropout=0.2)
    model.load_state_dict(torch.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_gcn.pt"),
        weights_only=True
    ))
    model.eval()

    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        pred_S = (pred.bool() & data.alive).sum().item() / data.num_nodes
        true_S = data.giant.sum().item() / data.num_nodes

    print(f"测试准确率: {test_acc:.4f}")
    print(f"S 真实 {true_S:.4f} | 预测 {pred_S:.4f}")


if __name__ == "__main__":
    main()
