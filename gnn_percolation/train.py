"""
在 Cora 渗流数据上训练 GCN（标准 PyG 节点分类流程）。
"""

import os
import torch
import torch.nn.functional as F
from model import GCN
from gen_data import make_data


def main():
    data = make_data(q=0.55, seed=42)
    pos = data.giant[data.alive].float().mean().item()
    print(f"{data}")
    print(f"巨分量/幸存者 = {pos:.3f}  (多数类基线 acc = {max(pos, 1-pos):.3f})")

    model = GCN(in_dim=3, hidden=64, out_dim=2, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val, best_state = 0.0, None
    for epoch in range(1, 201):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss {loss:.4f} | 验证Acc {val_acc:.4f}")

    # 测试
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        pred_S = (pred.bool() & data.alive).sum().item() / data.num_nodes
        true_S = data.giant.sum().item() / data.num_nodes
    print(f"\n最佳验证Acc: {best_val:.4f}")
    print(f"测试准确率   : {test_acc:.4f}")
    print(f"S 真实 {true_S:.4f} | 预测 {pred_S:.4f}")

    torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_gcn.pt"))


if __name__ == "__main__":
    main()
