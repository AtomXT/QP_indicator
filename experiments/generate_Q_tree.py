#!/usr/bin/env python3
"""
Generate datasets for QP-with-indicators experiment on TREE graphs:

    min_{x,z}  1/2 (d-x)^T Q (d-x) + lambda^T z
    s.t.       x_i (1 - z_i) = 0,  z_i in {0,1}

Experiment design:
- Q structure: random weighted TREE graph, implemented via a weighted tree Laplacian
              so Q is PSD and its off-diagonal sparsity pattern is exactly a tree.
- Condition number fixed: kappa(Q) ~= target_kappa (default 100)
- Normalize scale across settings: trace(Q)/n = 1
- For each n: generate R independent instances (default 5)

Outputs:
- One .npz per instance, containing:
    Q as CSR (data, indices, indptr, shape),
    d vector,
    metadata dict (as a JSON string).
"""

import argparse
import heapq
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


@dataclass
class InstanceMeta:
    n: int                      # problem size (dimension of Q, x, z)
    rep: int                    # replicate index for this n
    seed: int                   # RNG seed used to generate this instance
    graph_type: str             # here: "random_tree"

    target_kappa: float         # desired condition number of Q
    achieved_kappa_est: float   # estimated condition number after construction

    trace_per_dim_after_norm: float  # trace(Q)/n after normalization (should be ~1)

    num_edges: int              # for a tree, this should be n-1 when n >= 2
    offdiag_nnz: int            # number of nonzero off-diagonal entries in Q
    diag_nnz: int               # number of nonzero diagonal entries in Q
    Q_nnz_total: int            # total number of nonzeros in Q


def random_tree_edges(
    n: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Generate a uniformly random labeled tree on nodes {0, ..., n-1}
    using a Prüfer sequence.

    Returns a list of undirected edges (u, v), length n-1 for n >= 2.
    """
    if n <= 1:
        return []

    if n == 2:
        return [(0, 1)]

    prufer = rng.integers(0, n, size=n - 2)
    degree = np.ones(n, dtype=np.int64)
    for v in prufer:
        degree[v] += 1

    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)

    edges: List[Tuple[int, int]] = []
    for v in prufer:
        u = heapq.heappop(leaves)
        edges.append((u, int(v)))

        degree[u] -= 1
        degree[v] -= 1
        if degree[v] == 1:
            heapq.heappush(leaves, int(v))

    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    edges.append((u, v))

    return edges


def tree_laplacian_psd(
    n: int,
    rng: np.random.Generator,
    weight_low: float = 0.1,
    weight_high: float = 1.0,
) -> Tuple[sp.csr_matrix, int]:
    """
    Build a sparse weighted TREE graph Laplacian L:
      - sample a random labeled tree
      - assign each tree edge a positive random weight
      - set L_ij = -w_ij on edges, and L_ii to weighted degree

    Then L is PSD and has exactly a tree sparsity pattern off-diagonal.
    Returns L (CSR) and num_edges.
    """
    if n <= 1:
        return sp.csr_matrix((n, n), dtype=float), 0

    edges = random_tree_edges(n=n, rng=rng)
    m = len(edges)  # should be n-1

    i = np.fromiter((u for u, _ in edges), dtype=np.int64, count=m)
    j = np.fromiter((v for _, v in edges), dtype=np.int64, count=m)
    w = rng.uniform(weight_low, weight_high, size=m)

    # Off-diagonals: -w at (i,j) and (j,i)
    row = np.concatenate([i, j])
    col = np.concatenate([j, i])
    data = np.concatenate([-w, -w])

    # Diagonal: weighted degrees
    deg = np.zeros(n, dtype=float)
    np.add.at(deg, i, w)
    np.add.at(deg, j, w)

    L_off = sp.coo_matrix((data, (row, col)), shape=(n, n))
    L = (L_off + sp.diags(deg, offsets=0)).tocsr()
    return L, m


def build_Q_with_kappa(
    L: sp.csr_matrix,
    target_kappa: float,
    eps: float = 1e-12,
    max_eig_iter: int = 3000,
    tol: float = 1e-6,
) -> Tuple[sp.csr_matrix, float]:
    """
    Given Laplacian L (PSD, smallest eigenvalue = 0),
    set Q = L + alpha I so that kappa(Q) = (lambda_max(L)+alpha)/alpha ~= target_kappa.

    alpha = lambda_max(L) / (target_kappa - 1)

    Returns Q and the achieved condition number based on the computed lambda_max(L).
    """
    n = L.shape[0]

    if L.nnz == 0:
        alpha = 1.0
        Q = sp.eye(n, format="csr") * alpha
        return Q, 1.0

    lmax = float(
        eigsh(
            L, k=1, which="LA", return_eigenvectors=False,
            tol=tol, maxiter=max_eig_iter
        )[0]
    )

    if target_kappa <= 1.0 + 1e-12:
        Q = sp.eye(n, format="csr")
        return Q, 1.0

    alpha = max(lmax / (target_kappa - 1.0), eps)
    Q = (L + sp.eye(n, format="csr") * alpha).tocsr()

    # Scaling by a positive scalar does not change condition number,
    achieved = (lmax + alpha) / alpha
    return Q, achieved


def normalize_trace(Q: sp.csr_matrix) -> sp.csr_matrix:
    """
    Scale Q so that trace(Q)/n = 1.
    """
    n = Q.shape[0]
    tr = float(Q.diagonal().sum())
    if tr <= 0:
        return Q
    scale = 1.0 / (tr / n)
    return (Q * scale).tocsr()


def save_instance(
    out_path: str,
    Q: sp.csr_matrix,
    d: np.ndarray,
    meta: InstanceMeta,
) -> None:
    """
    Save one instance as a compressed NPZ with CSR arrays + vectors + JSON metadata.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "Q_data": Q.data,
        "Q_indices": Q.indices,
        "Q_indptr": Q.indptr,
        "Q_shape": np.array(Q.shape, dtype=np.int64),
        "d": d.astype(np.float64),
        "meta_json": np.array(json.dumps(asdict(meta)), dtype=object),
    }
    np.savez_compressed(out_path, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="../data/Q_tree")
    parser.add_argument(
        "--n_list",
        type=str,
        default="100,200,300,500,1000,2000,5000,10000",
        help="Comma-separated list of n values."
    )
    parser.add_argument("--reps", type=int, default=5, help="Instances per n.")
    parser.add_argument("--kappa", type=float, default=100.0, help="Target condition number of Q.")
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed.")
    parser.add_argument("--weight_low", type=float, default=0.1, help="Lower edge-weight bound.")
    parser.add_argument("--weight_high", type=float, default=1.0, help="Upper edge-weight bound.")
    args = parser.parse_args()

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    reps = args.reps

    rng_master = np.random.default_rng(args.seed)

    manifest = []
    for n in n_list:
        for rep in range(reps):
            # Derive a deterministic per-instance seed
            seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)

            # 1) Generate TREE Laplacian L (PSD, off-diagonal support is exactly a tree)
            L, m_edges = tree_laplacian_psd(
                n=n,
                rng=rng,
                weight_low=args.weight_low,
                weight_high=args.weight_high,
            )

            # 2) Form Q = L + alpha I to hit target kappa, then normalize trace(Q)/n = 1
            Q_raw, achieved_kappa = build_Q_with_kappa(L, target_kappa=args.kappa)
            Q = normalize_trace(Q_raw)

            trace_per_dim = float(Q.diagonal().sum()) / n

            # 3) Generate d and normalize ||d||_2 = 1
            d = rng.normal(size=n)
            d_norm = np.linalg.norm(d)
            if d_norm > 0:
                d = d / d_norm

            # Metadata / sparsity stats
            offdiag_nnz = int(Q.nnz - np.count_nonzero(Q.diagonal()))
            diag_nnz = int(np.count_nonzero(Q.diagonal()))
            meta = InstanceMeta(
                n=n,
                rep=int(rep),
                seed=seed,
                graph_type="random_tree",
                target_kappa=float(args.kappa),
                achieved_kappa_est=float(achieved_kappa),
                trace_per_dim_after_norm=float(trace_per_dim),
                num_edges=int(m_edges),
                offdiag_nnz=offdiag_nnz,
                diag_nnz=diag_nnz,
                Q_nnz_total=int(Q.nnz),
            )

            # Save
            fname = f"inst_n{n}_tree_rep{rep:02d}.npz"
            out_path = os.path.join(args.out_dir, f"n={n}", fname)
            save_instance(out_path, Q, d, meta)

            manifest.append({"path": out_path, **asdict(meta)})

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(manifest)} instances.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()