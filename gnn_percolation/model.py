"""
GNN 模型定义：基于 PyTorch Geometric 的标准 3 层 GCN。

使用 torch_geometric.nn.GCNConv（Kipf & Welling, 2017）：
    H' = sigma( D^{-1/2} A D^{-1/2} H W )
GCNConv 内部自动添加自环，无需手动处理邻接矩阵。
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """输入 -> hidden -> hidden -> out_dim"""
    def __init__(self, in_dim=3, hidden=64, out_dim=2, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x  # (N, out_dim) logits
