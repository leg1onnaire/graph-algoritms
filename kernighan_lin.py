import itertools
import time
from collections import defaultdict
from itertools import islice
from operator import itemgetter
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
from networkx.algorithms.community.community_utils import is_partition
import matplotlib.pyplot as plt

G = nx.Graph()

data = [
    (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (5, 0), (6, 0), (6, 4),
    (6, 5), (7, 0), (7, 1), (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
    (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2), (13, 3), (16, 5),
    (16, 6), (17, 0), (17, 1), (19, 0), (19, 1), (21, 0), (21, 1), (25, 23), (25, 24),
    (27, 2), (27, 23), (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8), (31, 0),
    (31, 24), (31, 25), (31, 28), (32, 2), (32, 8), (32, 14), (32, 15), (32, 18), (32, 20),
    (32, 22), (32, 23), (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13), (33, 14),
    (33, 15), (33, 18), (33, 19), (33, 20), (33, 22), (33, 23), (33, 26), (33, 27), (33, 28),
    (33, 29), (33, 30), (33, 31), (33, 32)
]




for edge in data:
    G.add_edge(int(edge[0]), int(edge[1]))
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

__all__ = ['kernighan_lin_bisection']

def _compute_delta(G, A, B, weight):
    delta = defaultdict(float)
    for u, v, d in G.edges(data=True):
        w = d.get(weight, 1)
        if u in A:
            if v in A:
                delta[u] -= w
                delta[v] -= w
            elif v in B:
                delta[u] += w
                delta[v] += w
        elif u in B:
            if v in A:
                delta[u] += w
                delta[v] += w
            elif v in B:
                delta[u] -= w
                delta[v] -= w
    return delta

def _update_delta(delta, G, A, B, u, v, weight):
    for _, nbr, d in G.edges(u, data=True):
        w = d.get(weight, 1)
        if nbr in A:
            delta[nbr] += 2 * w
        if nbr in B:
            delta[nbr] -= 2 * w
    for _, nbr, d in G.edges(v, data=True):
        w = d.get(weight, 1)
        if nbr in A:
            delta[nbr] -= 2 * w
        if nbr in B:
            delta[nbr] += 2 * w
    return delta

def _kernighan_lin_pass(G, A, B, weight):
    multigraph = G.is_multigraph()
    delta = _compute_delta(G, A, B, weight)
    swapped = set()
    gains = []
    while len(swapped) < len(G):
        gain = []
        for u in A - swapped:
            for v in B - swapped:
                try:
                    if multigraph:
                        w = sum(d.get(weight, 1) for d in G[u][v].values())
                    else:
                        w = G[u][v].get(weight, 1)
                except KeyError:
                    w = 0
                gain.append((delta[u] + delta[v] - 2 * w, u, v))
        if len(gain) == 0:
            break
        maxg, u, v = max(gain, key=itemgetter(0))
        swapped |= {u, v}
        gains.append((maxg, u, v))
        delta = _update_delta(delta, G, A - swapped, B - swapped, u, v, weight)
    return gains

@py_random_state(4)
@not_implemented_for('directed')
def kernighan_lin_bisection(G, partition=None, max_iter=10, weight='weight',seed=None):
    if partition is None:
        nodes = list(G)
        h = len(nodes) // 2
        print("h: ", h)
        partition = (nodes[:h], nodes[h:])
        print("partition: ", partition)
    try:
        A, B = set(partition[0]), set(partition[1])
    except:
        raise ValueError('partition must be two sets')
    if not is_partition(G, (A, B)):
        raise nx.NetworkXError('partition invalid')
    for i in range(max_iter):
        gains = _kernighan_lin_pass(G, A, B, weight)
        csum = list(itertools.accumulate(g for g, u, v in gains))
        max_cgain = max(csum)
        if max_cgain <= 0:
            break
        index = csum.index(max_cgain)
        nodesets = islice(zip(*gains[:index + 1]), 1, 3)
        anodes, bnodes = (set(s) for s in nodesets)
        A |= bnodes
        A -= anodes
        B |= anodes
        B -= bnodes
        return A, B

def main() -> None:
    start_time = time.time()  # Record the start time
    kl = kernighan_lin_bisection(G, partition=None, max_iter=10, weight='weight',seed=None)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time

    print(f"Elapsed time: {elapsed_time} seconds")
    print(f'= {kl}')
    print(f'= Subset1 : {len(kl[0])}')
    print(f'= Subset2 : {len(kl[1])}')


    cut_edges = []
    for u, v in G.edges():
        if (u in kl[0] and v in kl[1]) or (u in kl[1] and v in kl[0]):
            cut_edges.append((u, v))

    num_cut_edges = len(cut_edges)
    print(f"Number of cut edges: {num_cut_edges}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Original Graph
    pos_original = nx.spring_layout(G)
    node_colors_original = ['b' if n in kl[0] else 'b' for n in G.nodes()]
    nx.draw(G, pos_original, with_labels=True, node_color=node_colors_original, ax=axs[0])
    axs[0].set_title('Original Graph')


    # Original Graph
    pos_original = nx.spring_layout(G)
    node_colors_original = ['b' if n in kl[0] else 'g' for n in G.nodes()]
    nx.draw(G, pos_original, with_labels=True, node_color=node_colors_original, ax=axs[1])
    axs[1].set_title('Partitioned Graph')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()