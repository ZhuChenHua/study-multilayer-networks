"""
跟着论文 §2 -> §3 -> §4 讲用的最简实现，只用 PyTorch 张量（默认精度）。

图是邻接矩阵，连通分支用最小标号传播。为最简，只保留 ER 情形：论文式 (3) 的通用
生成函数 G_0/G_1 与式 (4)(5) 的 u 方程，对 ER 都化为 er_G 与 er_S。
"""

import torch

# ---------------------------------------------------------------------------- 第 2 节


def er_graph(N, c):
    """生成平均度为 c 的 ER 随机图，返回对称邻接矩阵（布尔张量）。
    Args:
        N: 节点数
        c: 目标平均度
    Returns:
        N x N 的布尔邻接矩阵
    """
    p = c / (N - 1)  # 任意一对节点间连边的概率
    upper = torch.rand(N, N) < p  # 只在上三角采样，避免每条边被算两次
    upper = upper.triu(1)  # 清掉对角线与下三角，只留 i < j 的边
    return upper | upper.T  # 转置相加，得到无向图的对称邻接矩阵


def random_failure(A, q):
    """随机攻击：每个节点以概率 q 独立失效，返回幸存节点的布尔掩码。
    Args:
        A: 邻接矩阵（图）
        q: 移除概率
    Returns:
        长度为 N 的布尔掩码，True 表示幸存
    """
    return torch.rand(A.shape[0]) >= q  # 每个节点以 1-q 的概率保留


def largest_component(A, alive):
    """式 (1)(2)：最小标号传播求 alive 诱导子图的最大连通分支，返回布尔掩码。
    Args:
        A:     邻接矩阵（图）
        alive: 幸存节点的布尔掩码
    Returns:
        最大连通分支的布尔掩码
    """
    n = A.shape[0]
    label = torch.arange(n)  # 每个节点先用自身编号做标号
    edge = alive[:, None] & alive[None, :] & A  # 只保留两端都幸存的实际边
    while True:
        nbr = torch.where(edge, label[None, :], n)  # 邻居的标号，非邻居记为哨兵 n
        new = torch.minimum(label, nbr.min(dim=1).values)  # 自己与邻居取最小标号
        if torch.equal(new, label):  # 标号不再下降：同一分支已同号
            break
        label = new
    counts = torch.bincount(label[alive], minlength=n)  # 每个分支的节点数
    return alive & (label == counts.argmax())  # 最大分支对应的节点


# ---------------------------------------------------------------------------- 第 3 节


def er_G(x, c):
    """式 (3)(6)：ER 的生成函数 G_0(x) = G_1(x) = e^{-c(1-x)}。
    Args:
        x: 自变量（标量或张量）
        c: 平均度
    Returns:
        生成函数在 x 处的取值
    """
    x = torch.as_tensor(x)  # 统一成张量
    return torch.exp(-c * (1 - x))  # 直接代入闭式


def er_S(q, c, M=1, steps=2000):
    """式 (7)(9)：不动点迭代解 S = p(1 - e^{-cS})^M。
    Args:
        q:     移除概率
        c:     平均度
        M:     层数，M=1 即单层
        steps: 迭代步数
    Returns:
        巨连通分支占比 S（从 S = 1 出发，从 0 会被平凡根钉死）
    """
    q = torch.as_tensor(q)
    p = 1 - q  # 保留概率
    S = torch.ones_like(p)  # 初值取 1，避开平凡根 S = 0
    for _ in range(steps):
        S = p * (1 - torch.exp(-c * S)) ** M  # 反复代入自洽方程
    return S


# ---------------------------------------------------------------------------- 第 4 节


def mc_gc(layers, alive):
    """§4：每层最大连通分支取交集，迭代到不动点，返回 MCGC 布尔掩码。
    Args:
        layers: 邻接矩阵组成的列表，每个元素是一层网络
        alive:  初始幸存节点的布尔掩码
    Returns:
        MCGC 的布尔掩码
    """
    current = alive
    while True:
        new = largest_component(layers[0], current)  # 第一层的最大分支
        for A in layers[1:]:
            new = new & largest_component(A, current)  # 再与其余层取交集
        if torch.equal(new, current):  # 集合不再缩小：到达不动点
            return new
        current = new


def y_c(M, steps=500):
    """式 (10)(11)：解驻点方程 e^x = 1 + Mx 的非平凡根 x_c，再代回式 (11)。
    Args:
        M:     层数
        steps: 不动点 x <- ln(1 + Mx) 的迭代步数
    Returns:
        临界参数 y_c
    """
    if M == 1:
        return torch.tensor(1.0)  # 单层：y_c = 1
    x = torch.tensor(1.0)
    for _ in range(steps):
        x = torch.log(1 + M * x)  # 迭代求 e^x = 1 + Mx 的非零根
    return x / (1 - torch.exp(-x)) ** M  # 代回 y_c = x / (1 - e^{-x})^M


def q_c(M, c):
    """式 (12)：临界移除概率 q_c = 1 - y_c / c。
    Args:
        M: 层数
        c: 平均度
    Returns:
        临界移除概率 q_c
    """
    return 1 - y_c(M) / c  # 由 y = c(1-q) 反解


# -------------------------------------- 演示：每块对应一段公式

if __name__ == "__main__":
    torch.manual_seed(0)  # 固定随机性

    # ===== 第 2 节  式 (1)(2)：er_graph / random_failure / largest_component =====
    N, c = 2000, 4.0
    A = er_graph(N, c)
    for q in (0.0, 0.3, 0.5, 0.9):
        alive = random_failure(A, q)
        giant = largest_component(A, alive)
        R = giant.sum().item() / N
        print(f"q={q:.2f}  R(q)={R:.4f}")

    # ===== 第 3 节  式 (3)(6)：er_G =====
    # c = 4.0
    # for x in (0.0, 0.5, 1.0):
    #     print(f"x={x:.1f}  G0=G1={er_G(x, c):.4f}")

    # ===== 第 3 节  式 (7)(8)：er_S =====
    # c = 4.0
    # for q in (0.0, 0.3, 0.5, 0.75, 0.9):
    #     print(f"q={q:.2f}  S={er_S(q, c):.4f}   (q_c=1-1/c={1 - 1 / c:.2f})")

    # ===== 第 4 节  式 (9)：mc_gc =====
    # N, c = 2000, 4.0
    # layers = [er_graph(N, c) for _ in range(2)]
    # for q in (0.0, 0.2, 0.3, 0.5):
    #     alive = random_failure(layers[0], q)
    #     giant = mc_gc(layers, alive)
    #     S = giant.sum().item() / N
    #     print(f"q={q:.2f}  MCGC={S:.4f}  理论={er_S(q, c, 2):.4f}")

    # ===== 第 4 节  式 (10)(11)(12)：y_c / q_c =====
    # for M in (1, 2, 3, 4):
    #     print(f"M={M}  y_c={y_c(M):.4f}  q_c(c=6)={q_c(M, 6.0):.4f}  q_c(c=4)={q_c(M, 4.0):.4f}")
