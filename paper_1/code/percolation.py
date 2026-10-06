"""
论文实验所需的最小实现。函数按论文小节的顺序排列，自上而下即 §2 -> §3.1 -> §3.2 -> §4。

  第 2 节 连通分支与随机攻击模型
    random_failure         每个节点以概率 q 独立移除
    largest_component      R(q) = |C_max(G_q)| / N
  第 3.1 节 生成函数
    generating_functions   G_0(x) = sum_k P(k)x^k 与 G_1(x) = sum_k kP(k)/<k> x^{k-1}
    edge_giant_prob        u = q + p G_1(u) 的非平凡根
    giant_fraction         S = p[1 - G_0(u)]
  第 3.2 节 ER 网络
    er_degree_distribution P(k) = e^{-c} c^k / k!
    er_layer               生成平均度为 c 的 ER 网络（第 5 节取 N=10^4, c=4）
    theory_S               S = p(1-e^{-cS})^M 的最大非平凡根（M=1 即单层 ER 结果）
  第 4 节 多层相互依赖网络
    mc_gc                  MCGC 级联迭代的不动点
    y_critical             y_c = max_x x/(1-e^{-x})^M
    q_critical             q_c = 1 - y_c / c

论文 12 个编号公式的对应关系：
  式 (1)(2)  C_max 与 R(q)              -> largest_component（式 (2) 的除以 N 在调用处做）
  式 (3)     G_0 与 G_1                 -> generating_functions
  式 (4)     u = q + p G_1(u)           -> edge_giant_prob
  式 (5)     S = p[1 - G_0(u)]          -> giant_fraction
  式 (6)     G_0 = G_1 = e^{-c(1-x)}    -> 由式 (3) 配 Poisson P(k) 即得，不另写函数
  式 (7)     S = p(1-e^{-cS})           -> theory_S(M=1)
  式 (8)     p_c = 1/c, q_c = 1 - 1/c   -> q_critical(M=1, c)（p_c = 1 - q_c，不另写）
  式 (9)     S = p(1-e^{-cS})^M         -> theory_S(M)
  式 (10)    g_y = 0 且 g_y' = 0        -> y_critical 解其精确等价形式 e^x = 1 + Mx
  式 (11)    y_c = max_x x/(1-e^{-x})^M -> y_critical
  式 (12)    q_c = 1 - y_c / c          -> q_critical

本文件是纯函数库，供 figures.py 导入出图用；逐函数演示见 percolation_lecture.py。
"""

import random
from collections import deque
from math import lgamma

import numpy as np

# ---------------------------------------------------------------------------- 第 2 节


def random_failure(adj, q, seed=None):
    """
    随机攻击：每个节点以概率 q 独立失效，返回幸存节点列表。
    Args:
        adj: 邻接表（图）
        q:   移除概率
        seed: 随机数种子，None 表示不固定随机性
    Returns:
        幸存节点列表
    """

    rng = random.Random(seed)
    return [i for i in range(len(adj)) if rng.random() >= q]


def largest_component(adj, active):
    """
    BFS 求诱导子图 G[active] 的最大连通分支 C_max。
    Args:
        adj: 邻接表
        active: 幸存节点列表
    Returns:
        best: 最大连通分支的节点列表
    """

    alive, seen, best = set(active), set(), []
    for start in active:
        if start in seen:
            continue
        component, queue = [], deque([start])
        seen.add(start)
        while queue:
            u = queue.popleft()
            component.append(u)
            for v in adj[u]:
                if v in alive and v not in seen:
                    seen.add(v)
                    queue.append(v)
        if len(component) > len(best):
            best = component
    return best


# -------------------------------------------------------------------------- 第 3.1 节


def generating_functions(P):
    """
    第 3.1 节的生成函数。它们都是多项式，系数直接由度分布 P 给出：

        G_0(x) = sum_k P(k) x^k,
        G_1(x) = sum_k k P(k) / <k> x^{k-1}.

    区别只在“怎么走到这个节点”，这也是论文里 G_1 那个因子 k 的来源：

        G_0  随便走到一个节点（无偏）  ->  该节点度数为 k 的概率是 P(k)
        G_1  沿一条边走到另一端（有偏） ->  该端点度数为 k 的概率是 k P(k) / <k>

    走边等价于先随机抽一条边，而高度数节点的边更多、被抽中的概率更大，所以 G_1 的
    系数整体偏向大 k，这叫 size-biased。这也解释了论文里的用法：判断“一条边能否
    通向巨连通分支”要沿边到达，解 u = q + p G_1(u) 时用 G_1；而“某个节点是否在
    巨连通分支里”是对所有节点无偏计数，S = p[1 - G_0(u)] 用 G_0。

    返回 (G0, G1) 两个可调用对象，均接受标量或数组 x，且都满足 G_0(1) = G_1(1) = 1。
    """

    P = np.asarray(P, dtype=float)
    degrees = np.arange(len(P))
    mean_k = float(np.dot(P, degrees))
    if mean_k <= 0:
        raise ValueError("度分布的平均度 <k> 必须为正")

    # G_1 的指数是 k - 1，而系数数组按 x^i 排列，所以 i = k - 1：只保留 k >= 1，
    # 丢掉贡献恒为 0 的 k = 0 项，g1_coef[i] 便正好等于 (i+1) P(i+1) / <k>。
    g1_coef = degrees[1:] * P[1:] / mean_k

    # np.polynomial.polynomial.polyval 是升幂求值：系数数组第 i 项乘以 x^i，与
    # np.polyval 的降幂约定相反，这里容易看错。
    polyval = np.polynomial.polynomial.polyval

    return lambda x: polyval(x, P), lambda x: polyval(x, g1_coef)


def edge_giant_prob(q, P):
    """
    第 3.1 节式 (4)：解 u = q + p G_1(u)，即“随机一条边不能通向巨连通分支”的概率。

    u = 1 恒为平凡根。当 p<k> > 1 时 (0, 1) 内另有唯一非平凡根，取网格上第一个变号区间
    再二分细化（与 theory_S 取最后一个变号区间相对照：一个取最小根，一个取最大根）；
    不存在非平凡根时返回 1.0，对应没有巨连通分支。
    """

    _, G1 = generating_functions(P)
    f = lambda u: q + (1 - q) * G1(u) - u
    grid = np.linspace(0.0, 1.0, 20001)
    values = f(grid)
    sign_change = np.nonzero(values[:-1] * values[1:] < 0)[0]
    if sign_change.size == 0:
        return 1.0
    lo, hi, f_lo = (
        grid[sign_change[0]],
        grid[sign_change[0] + 1],
        values[sign_change[0]],
    )
    for _ in range(60):
        mid = (lo + hi) / 2
        if f_lo * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2)


def giant_fraction(q, P):
    """第 3.1 节式 (5)：巨连通分支比例 S = p[1 - G_0(u)]，其中 u = edge_giant_prob(q, P)。"""

    G0, _ = generating_functions(P)
    return (1 - q) * (1 - G0(edge_giant_prob(q, P)))


# -------------------------------------------------------------------------- 第 3.2 节


def er_degree_distribution(c, k_max=None):
    """
    第 3.2 节：平均度为 c 的 ER 网络的度分布 P(k) = e^{-c} c^k / k!。

    截断到 k_max 并重新归一化；缺省 k_max 取 c + 10 sqrt(c) + 12，此时截断概率可忽略。

    论文式 (6) 的闭式 G_0(x) = G_1(x) = e^{-c(1-x)} 不另写函数：它由式 (3) 配上本函数
    的 Poisson P(k) 直接得到，把 P 传给 generating_functions 即可复现。
    """

    if c <= 0:
        raise ValueError("平均度 c 必须为正")
    k_max = int(c + 10 * c**0.5 + 12) if k_max is None else k_max
    ks = np.arange(k_max + 1)
    P = np.exp(-c + ks * np.log(c) - np.array([lgamma(k + 1) for k in ks]))
    return P / P.sum()


def er_layer(N, c, seed=None):
    """第 3.2 节：平均度为 c 的 ER 随机图（边数 m = cN/2，随机抽边去重），返回邻接表。"""

    rng = random.Random(seed)
    m = round(c * N / 2)
    edges = set()
    while len(edges) < m:
        u, v = rng.randrange(N), rng.randrange(N)
        if u != v:
            edges.add((u, v) if u < v else (v, u))
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def theory_S(q, c, M=1):
    """
    第 3.2、4 节式 (7)、式 (9)：理论巨分支比例 S = p(1-e^{-cS})^M 的最大非平凡根。

    M = 1 时退化为单层 ER 结果 S = p(1-e^{-cS})。M >= 2 时阈值附近有一大一小两个正根，
    物理上稳定的是较大的那个，因此先在网格上找最后一个变号区间，再二分细化。
    """

    p = 1 - q
    f = lambda s: p * (1 - np.exp(-c * s)) ** M - s
    grid = np.linspace(1e-12, 1.0, 20001)
    values = f(grid)
    sign_change = np.nonzero(values[:-1] * values[1:] < 0)[0]
    if sign_change.size == 0:
        return 0.0
    lo, hi, f_lo = (
        grid[sign_change[-1]],
        grid[sign_change[-1] + 1],
        values[sign_change[-1]],
    )
    for _ in range(60):
        mid = (lo + hi) / 2
        if f_lo * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2)


# ---------------------------------------------------------------------------- 第 4 节


def mc_gc(layers, active, max_iter=100):
    """
    第 4 节 MCGC：反复取各层最大连通分支的交集，直到集合不再变化。

    各层的最大连通分支都由同一份上轮幸存集合 current 出发再取交集，而不是把上一层的
    输出直接喂给下一层；收敛时得到的就是论文所述级联迭代的不动点。
    """

    current = set(active)
    for _ in range(max_iter):
        new = set(largest_component(layers[0], current))
        for adj in layers[1:]:
            new &= set(largest_component(adj, current))
        if new == current:
            break
        current = new
    return current


def y_critical(M):
    """
    第 4 节式 (10)、式 (11)：临界参数 y_c = c p_c = c(1-q_c)。

    M = 1 时退化为 y_c = 1。M > 1 时式 (10) 的两个条件 g_y(x) = x - y(1-e^{-x})^M = 0
    与 g_y'(x) = 0 联立，消去 y 后等价于驻点方程 e^x = 1 + Mx，这里用二分法求其非平凡根
    x_c > 0；再代回式 (11) 的 y_c = x/(1-e^{-x})^M。（直接用 g_y 与 g_y' 做二元求根更脆，
    而两者的解完全相同：把 x_c 代回可得 g_y(x_c) = 0、g_y'(x_c) = 0 至机器精度。）
    M = 2 时给出 x_c ≈ 1.2564、y_c ≈ 2.4554。
    """

    if M == 1:
        return 1.0
    lo, hi = 1e-9, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.exp(mid) - 1 - M * mid > 0:
            hi = mid
        else:
            lo = mid
    x = (lo + hi) / 2
    return float(x / (1 - np.exp(-x)) ** M)


def q_critical(M, c):
    """
    第 4 节式 (12)：临界移除概率 q_c = 1 - y_c / c；M = 1 时即式 (8) 的 q_c = 1 - 1/c。

    式 (8) 里的 p_c = 1/c 不另写函数，它恒等于 1 - q_c。
    """

    return 1 - y_critical(M) / c
