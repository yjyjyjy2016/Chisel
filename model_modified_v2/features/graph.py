import numpy as np
import torch
import networkx as nx
import sys
import os
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)


def quaternion(v1, v2):
    """Single-pair quaternion (kept for compatibility)."""
    w = torch.dot(v1, v2) + torch.norm(v1) * torch.norm(v2)
    u = torch.cross(v1, v2)
    w = torch.unsqueeze(w, dim=-1)
    q = torch.cat([w, u], dim=-1)
    q = q / torch.norm(q)
    return q


def compute_rotation_movment(traj, top, length=400):
    """Vectorized computation of pairwise distances, movements, and quaternions.

    Original: O(N^2) Python for-loop, ~minutes for N=500.
    Fixed: vectorized torch ops, ~seconds for N=500.

    Only distances are used downstream (for centrality), so we prioritize
    getting distances right and fast. Quaternions are computed in batch too.
    """
    CAs_index = top.select("backbone and name CA")
    Ns_index = top.select("backbone and name N")
    Cs_index = top.select("backbone and name C")

    CAs = torch.from_numpy(traj.xyz[0, CAs_index].copy())
    Ns = torch.from_numpy(traj.xyz[0, Ns_index].copy())
    Cs = torch.from_numpy(traj.xyz[0, Cs_index].copy())

    real_length = min(len(CAs), len(Ns), len(Cs))
    CAs = CAs[:real_length]
    Ns = Ns[:real_length]
    Cs = Cs[:real_length]

    # Vectorized pairwise distances: (real_length, real_length)
    distances = torch.zeros((length, length), dtype=torch.float)
    dist_block = torch.cdist(CAs.unsqueeze(0), CAs.unsqueeze(0)).squeeze(0)
    distances[:real_length, :real_length] = dist_block

    # Vectorized movement: (CAs[i] - CAs[j]) / dist
    movement = torch.zeros((length, length, 3), dtype=torch.float)
    diff = CAs.unsqueeze(1) - CAs.unsqueeze(0)  # (N, N, 3)
    safe_dist = dist_block.clone()
    safe_dist[safe_dist == 0] = 1.0  # avoid div by zero on diagonal
    mov_block = diff / safe_dist.unsqueeze(-1)
    movement[:real_length, :real_length] = mov_block

    # Vectorized quaternions from direction vectors
    CA_N = Ns - CAs  # (N, 3)
    CA_C = Cs - CAs  # (N, 3)
    dircts = torch.cross(CA_C, CA_N)  # (N, 3)

    quternions = torch.zeros((length, length, 4), dtype=torch.float)
    # Diagonal: identity quaternion [1, 0, 0, 0]
    for i in range(real_length):
        quternions[i, i] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    # Off-diagonal: batched quaternion computation
    # q = [w, u] where w = dot(d_i, d_j) + ||d_i|| * ||d_j||, u = cross(d_i, d_j)
    if real_length > 1:
        norms = torch.norm(dircts, dim=1)  # (N,)
        # dot products: (N, N)
        dots = torch.mm(dircts, dircts.T)
        # norm products: (N, N)
        norm_prods = norms.unsqueeze(1) * norms.unsqueeze(0)
        # w values: (N, N)
        w_vals = dots + norm_prods

        # cross products: vectorized via broadcasting
        # cross(dircts[i], dircts[j]) for all i,j
        d1 = dircts.unsqueeze(1).expand(-1, real_length, -1)  # (N, N, 3)
        d2 = dircts.unsqueeze(0).expand(real_length, -1, -1)  # (N, N, 3)
        u_vals = torch.cross(d1, d2, dim=2)  # (N, N, 3)

        # Combine: (N, N, 4)
        q_block = torch.cat([w_vals.unsqueeze(-1), u_vals], dim=-1)
        # Normalize
        q_norms = torch.norm(q_block, dim=-1, keepdim=True).clamp(min=1e-8)
        q_block = q_block / q_norms

        quternions[:real_length, :real_length] = q_block
        # Re-set diagonal
        for i in range(real_length):
            quternions[i, i] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    return distances, movement, quternions


def compute_shortestpath_centerilty(distances, length=400):
    """Vectorized shortest path + centrality computation.

    Original: O(N^2) calls to nx.shortest_path_length, extremely slow.
    Fixed: scipy.sparse.csgraph.shortest_path does all-pairs BFS at once.

    Betweenness centrality still uses networkx but is only O(N^2) itself.
    """
    shape = distances.shape
    weight = torch.where(distances != 0.0, 1 / distances, torch.tensor(0.0))
    graph_matrix = torch.where(weight > 0.8333333, 1.0, 0.0).numpy()

    # Betweenness centrality via networkx (this is O(N*E), acceptable)
    graph = nx.from_numpy_array(graph_matrix)
    centerity = np.array(
        list(nx.centrality.betweenness_centrality(graph).values()))

    # All-pairs shortest path via scipy (MUCH faster than N^2 nx calls)
    sparse_graph = csr_matrix(graph_matrix)
    sp_matrix = shortest_path(sparse_graph, method='D', directed=False,
                              unweighted=True)
    # Replace inf (unreachable) with 0
    sp_matrix[np.isinf(sp_matrix)] = 0.0

    # Pad/truncate to requested shape
    N = sp_matrix.shape[0]
    if N < shape[0]:
        shortest_path_length = np.zeros(shape)
        shortest_path_length[:N, :N] = sp_matrix
    else:
        shortest_path_length = sp_matrix[:shape[0], :shape[1]]

    return shortest_path_length, centerity


if __name__ == "__main__":
    import time as _time
    print("Vectorized graph features - test")
    # Quick benchmark
    N = 200
    d = torch.rand(N, N)
    d = (d + d.T) / 2
    d.fill_diagonal_(0)
    start = _time.time()
    sp, cen = compute_shortestpath_centerilty(d, length=N)
    elapsed = _time.time() - start
    print(f"N={N}: shortest_path + centrality in {elapsed:.2f}s")
    print(f"  sp shape: {sp.shape}, centrality shape: {cen.shape}")

