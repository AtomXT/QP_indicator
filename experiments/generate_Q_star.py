#!/usr/bin/env python3
"""
Generate datasets for QP-with-indicators experiment:

    min_{x,z}  1/2 (d-x)^T Q (d-x) + lambda^T z
    s.t.       x_i (1 - z_i) = 0,  z_i in {0,1}

Experiment design:
- Q structure: star graph centered at node 0
- Condition number fixed: kappa(Q) ~= target_kappa (default 100)
- Normalize scale across settings: trace(Q)/n = 1
- For each (n): generate R independent instances (default 5)

Outputs:
- One .npz per instance, containing:
    Q as CSR (data, indices, indptr, shape),
    d vector,
    metadata dict (as a JSON string).
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


@dataclass
class InstanceMeta:
    n: int
    rep: int
    seed: int
    graph_type: str
    center: int

    target_kappa: float
    achieved_kappa_est: float

    trace_per_dim_after_norm: float

    num_edges: int
    offdiag_nnz: int
    diag_nnz: int
    Q_nnz_total: int


def star_laplacian_psd(
    n: int,
    rng: np.random.Generator,
    center: int = 0,
    weight_low: float = 0.1,
    weight_high: float = 1.0,
) -> Tuple[sp.csr_matrix, int]:
    """
    Build a weighted star graph Laplacian L on nodes 0,1,...,n-1:
      edges = (center, j) for every j != center

    For each star edge, draw weight w_j > 0 and set
      L_{center,j} = L_{j,center} = -w_j
      diagonal entries accumulate incident weights

    Returns:
      L : CSR sparse matrix, PSD
      num_edges : n - 1
    """
    if n <= 1:
        return sp.csr_matrix((n, n), dtype=float), 0
    if not (0 <= center < n):
        raise ValueError("center must satisfy 0 <= center < n.")

    leaves = np.array([j for j in range(n) if j != center], dtype=np.int64)
    num_edges = leaves.size
    weights = rng.uniform(weight_low, weight_high, size=num_edges)

    row = np.concatenate([np.full(num_edges, center, dtype=np.int64), leaves])
    col = np.concatenate([leaves, np.full(num_edges, center, dtype=np.int64)])
    data = np.concatenate([-weights, -weights])

    deg = np.zeros(n, dtype=float)
    deg[center] = float(np.sum(weights))
    deg[leaves] = weights

    L_off = sp.coo_matrix((data, (row, col)), shape=(n, n))
    L = (L_off + sp.diags(deg, offsets=0)).tocsr()
    return L, int(num_edges)


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

    Returns Q and an estimated achieved kappa.
    """
    n = L.shape[0]

    if L.nnz == 0:
        alpha = 1.0
        Q = sp.eye(n, format="csr") * alpha
        return Q, 1.0

    lmax = float(
        eigsh(L, k=1, which="LA", return_eigenvectors=False, tol=tol, maxiter=max_eig_iter)[0]
    )

    if target_kappa <= 1.0 + 1e-12:
        Q = sp.eye(n, format="csr")
        return Q, 1.0

    alpha = max(lmax / (target_kappa - 1.0), eps)
    Q = (L + sp.eye(n, format="csr") * alpha).tocsr()

    achieved = (lmax + alpha) / max(alpha, eps)
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
    parser.add_argument("--out_dir", type=str, default="../data/Q_star")
    parser.add_argument("--n_list", type=str, default="100,200,300,500,1000,2000,5000,10000",
                        help="Comma-separated list of n values.")
    parser.add_argument("--reps", type=int, default=5, help="Instances per n.")
    parser.add_argument("--kappa", type=float, default=100.0, help="Target condition number of Q.")
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed.")
    parser.add_argument("--center", type=int, default=0, help="Center node of the star graph.")
    parser.add_argument("--weight_low", type=float, default=0.1, help="Lower edge-weight bound.")
    parser.add_argument("--weight_high", type=float, default=1.0, help="Upper edge-weight bound.")
    args = parser.parse_args()

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    reps = args.reps

    rng_master = np.random.default_rng(args.seed)

    manifest = []
    for n in n_list:
        for rep in range(reps):
            seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)

            L, m_edges = star_laplacian_psd(
                n=n,
                rng=rng,
                center=int(args.center),
                weight_low=float(args.weight_low),
                weight_high=float(args.weight_high),
            )

            Q_raw, achieved_kappa = build_Q_with_kappa(L, target_kappa=args.kappa)
            Q = normalize_trace(Q_raw)

            trace_per_dim = float(Q.diagonal().sum()) / n

            d = rng.normal(size=n)
            d_norm = np.linalg.norm(d)
            if d_norm > 0:
                d = d / d_norm

            offdiag_nnz = int(Q.nnz - np.count_nonzero(Q.diagonal()))
            diag_nnz = int(np.count_nonzero(Q.diagonal()))
            meta = InstanceMeta(
                n=n,
                rep=int(rep),
                seed=seed,
                graph_type="star",
                center=int(args.center),
                target_kappa=float(args.kappa),
                achieved_kappa_est=float(achieved_kappa),
                trace_per_dim_after_norm=float(trace_per_dim),
                num_edges=int(m_edges),
                offdiag_nnz=offdiag_nnz,
                diag_nnz=diag_nnz,
                Q_nnz_total=int(Q.nnz),
            )

            fname = f"inst_n{n}_star_rep{rep:02d}.npz"
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
