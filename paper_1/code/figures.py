"""
生成论文的三张图（英文标注，与 main.tex 中的 ACL 排版一致）。

图 1 model              引言的模型示意图：(a) 单层 ER 的最大连通分支，(b) 两层相互依赖的 MCGC
图 2 giant_component    单层 ER 与两层相互依赖 ER 的 R(q)：模拟点 + 理论曲线
图 3 phase_transition   M = 1..4 的 MCGC 曲线 S(q) 与各自的阈值 q_c

连通分支一律用 percolation.py 里的 BFS 版本求解。这里试过把整个 Monte Carlo 批量张量化
（torch 的“钩连+指针跳跃”并行连通分支，一次算完整批失效掩码），结论是本仓库规模下不划算：
N <= 10^4、E <= 2*10^4 时一次 BFS 只要约 15 ms，而指针跳跃需要 30 轮全边扫描，逐样本提前
退出后 figure1 只快 2%（16.4 s vs 16.8 s），figure2 反而慢 75%（2.1 s vs 1.2 s）。连通分支
是典型的指针追逐型访存，数组框架的张量算子开销摊不开，没有算术强度可以复用。

运行：python paper_1/code/figures.py
输出：paper_1/figures/*.pdf 与 *.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from percolation import er_layer, largest_component, mc_gc, q_critical, random_failure, theory_S, y_critical

N, C, RUNS, Q_MAX = 10_000, 4.0, 20, 0.95
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def simulate_single_and_double(qs, runs=RUNS, N=N, c=C):
    """第 5 节实验一：对每个 q 做 runs 次独立实验，返回单层 R(q) 与两层 MCGC 比例的均值。"""

    single, double = np.zeros(len(qs)), np.zeros(len(qs))
    for run in range(runs):
        layers = [er_layer(N, c, seed=1000 + run), er_layer(N, c, seed=5000 + run)]
        for i, q in enumerate(qs):
            alive = random_failure(layers[0], q, seed=run * 97 + i)
            single[i] += len(largest_component(layers[0], alive)) / N
            double[i] += len(mc_gc(layers, alive)) / N
    return single / runs, double / runs


def simulate_mcgc(M, qs, runs=8, N=4000, c=6.0):
    """第 5 节实验二：M 层相互依赖网络的 MCGC 比例均值。"""

    data = np.zeros(len(qs))
    for run in range(runs):
        layers = [er_layer(N, c, seed=9000 * M + 137 * run + i) for i in range(M)]
        for i, q in enumerate(qs):
            alive = random_failure(layers[0], q, seed=7919 * run + 31 * i + M)
            data[i] += len(mc_gc(layers, alive)) / N
    return data / runs


def figure1():
    qs = np.linspace(0, Q_MAX, 21)
    single, double = simulate_single_and_double(qs)
    dense = np.linspace(0, Q_MAX, 600)
    q1, q2 = q_critical(1, C), q_critical(2, C)

    fig, ax = plt.subplots(figsize=(3.1, 2.0))
    for M, sim, color, marker, name in ((1, single, "tab:blue", "o", "single layer"),
                                        (2, double, "tab:red", "s", "two interdependent layers")):
        q_c = q1 if M == 1 else q2
        ax.plot(dense, [theory_S(q, C, M) for q in dense], color=color, lw=1.5, label=name)
        ax.plot(qs, sim, marker, color=color, ms=3.2, mfc="none", mew=0.8)
        ax.axvline(q_c, color=color, ls="--", lw=0.8, alpha=0.7)
        ax.annotate(rf"$q_c={q_c:.3f}$", xy=(q_c, 0.99 if M == 2 else 0.86),
                    ha="center", va="top", color=color, fontsize=8)
    ax.set_xlabel(r"removed fraction $q$")
    ax.set_ylabel(r"component fraction of $N$")
    ax.set(xlim=(0, Q_MAX), ylim=(0, 1.02))
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    return fig, f"q_c = {q1:.2f} (single layer), {q2:.4f} (two layers)"


def figure2():
    qs, Ms = np.linspace(0, 0.9, 13), [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(3.1, 2.05))
    for M in Ms:
        ax.plot(qs, [theory_S(q, 6.0, M) for q in qs], lw=1.4, label=rf"$M={M}$")
        ax.axvline(q_critical(M, 6.0), ls="--", lw=0.7, alpha=0.4)
    for M, marker in ((1, "o"), (2, "s")):
        ax.plot(qs, simulate_mcgc(M, qs), marker, ls="none", ms=3.0, mfc="none",
                 mew=0.8, color=f"C{M - 1}")
    ax.set(xlim=(0, 0.9), ylim=(-0.02, 1.02))
    ax.set_xlabel(r"removed fraction $q$")
    ax.set_ylabel(r"MCGC fraction $S_{\mathrm{MCGC}}$")
    ax.legend(loc="lower left", ncol=2, columnspacing=0.8, fontsize=8, framealpha=1.0,
              facecolor="white", edgecolor="none")
    return fig, "y_c = " + ", ".join(f"M={M}: {y_critical(M):.4f}" for M in Ms)


def spring_pos(adj, seed=0):
    """小网络的弹簧布局，归一化到 [-1, 1]^2。"""

    graph = nx.Graph()
    graph.add_nodes_from(range(len(adj)))
    graph.add_edges_from((u, v) for u, nbrs in enumerate(adj) for v in nbrs if v > u)
    pos = nx.spring_layout(graph, seed=seed)
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return {i: (2 * (pos[i][0] - x0) / (x1 - x0) - 1, 2 * (pos[i][1] - y0) / (y1 - y0) - 1)
            for i in pos}


def draw_network(ax, adj, pos, alive, giant, color, alive_all=None):
    """画出单层网络：巨分支（giant）用彩色实心点，其余幸存点灰色，失效点打叉。"""

    alive_all = alive_all if alive_all is not None else alive
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            if v <= u:
                continue
            if u in giant and v in giant:
                ax.plot(*zip(pos[u], pos[v]), color=color, lw=1.3, zorder=2)
            elif u in alive and v in alive:
                ax.plot(*zip(pos[u], pos[v]), color="0.85", lw=0.6, zorder=1)
    small = [pos[i] for i in alive_all if i not in giant]
    if small:
        ax.scatter(*zip(*small), s=10, c="0.55", zorder=3)
    ax.scatter(*zip(*(pos[i] for i in giant)), s=26, c=color, zorder=4)
    dead = [pos[i] for i in range(len(adj)) if i not in alive_all]
    if dead:
        ax.scatter(*zip(*dead), s=16, c="0.55", marker="x", lw=0.8, zorder=3)
    ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.15))
    ax.axis("off")


def figure_model():
    """引言用的模型示意图：(a) 单层 ER 的最大连通分支；(b) 两层相互依赖的 MCGC。"""

    n, c, q = 45, 4.0, 0.25
    layer_a, layer_b = er_layer(n, c, seed=3), er_layer(n, c, seed=4)
    pos_a, pos_b = spring_pos(layer_a, seed=1), spring_pos(layer_b, seed=2)
    shrink = lambda pos, dx: {i: (0.42 * x + dx, y) for i, (x, y) in pos.items()}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.1, 3.25))

    alive = set(random_failure(layer_a, q, seed=1))
    draw_network(ax1, layer_a, pos_a, alive, set(largest_component(layer_a, alive)), "tab:blue")
    ax1.set_title(r"(a) single layer: giant component $C_{\max}$", fontsize=9)

    alive = set(random_failure(layer_a, q, seed=1))
    mc = mc_gc([layer_a, layer_b], alive)
    for i in range(n):
        ax2.plot([0.42 * pos_a[i][0] - 0.58, 0.42 * pos_b[i][0] + 0.58],
                 [pos_a[i][1], pos_b[i][1]], color="0.8", lw=0.5, ls=":", zorder=0)
    draw_network(ax2, layer_a, shrink(pos_a, -0.58), alive, mc, "tab:red", alive_all=set(range(n)))
    draw_network(ax2, layer_b, shrink(pos_b, 0.58), alive, mc, "tab:red", alive_all=set(range(n)))
    ax2.set_title(r"(b) two interdependent layers: MCGC", fontsize=9)

    fig.tight_layout()
    return fig, f"N = {n}, c = {c}, q = {q}: |C_max| = {len(largest_component(layer_a, set(random_failure(layer_a, q, seed=1))))}, |MCGC| = {len(mc)}"


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
    })
    os.makedirs(OUT, exist_ok=True)
    for name, fig, note in (("model", *figure_model()),
                            ("giant_component", *figure1()),
                            ("phase_transition", *figure2())):
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"{name}.{suffix}"), bbox_inches="tight")
        print(f"{name}: {note}")
    print("figures written to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
