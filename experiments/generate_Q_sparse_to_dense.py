#!/usr/bin/env python3
"""
Generate datasets for QP-with-indicators experiment:

    min_{x,z}  1/2 (d-x)^T Q (d-x) + lambda^T z
    s.t.       x_i (1 - z_i) = 0,  z_i in {0,1}

Experiment design:
- Q structure: Erdős–Rényi sparsity pattern, implemented via a weighted graph Laplacian
              so Q is PSD and has EXACTLY the ER edge sparsity on off-diagonals.
- Condition number fixed: kappa(Q) ~= target_kappa (default 100)
- Normalize scale across settings: trace(Q)/n = 1
- For each (n, delta): generate R independent instances (default 10)

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
    n: int                      # problem size (dimension of Q, x, z)
    delta: float                # ER edge probability controlling sparsity
    rep: int                    # replicate index for this (n, delta) setting
    seed: int                   # RNG seed used to generate this instance

    target_kappa: float         # desired condition number of Q
    achieved_kappa_est: float   # estimated condition number after construction

    trace_per_dim_after_norm: float  # trace(Q)/n after normalization (should be ~1)

    num_edges: int              # number of ER edges (controls off-diagonal sparsity)
    offdiag_nnz: int            # number of nonzero off-diagonal entries in Q
    diag_nnz: int               # number of nonzero diagonal entries in Q
    Q_nnz_total: int            # total number of nonzeros in Q


def er_laplacian_psd(
    n: int,
    delta: float,
    rng: np.random.Generator,
    weight_low: float = 0.1,
    weight_high: float = 1.0,
) -> Tuple[sp.csr_matrix, int]:
    """
    Build a sparse weighted ER graph Laplacian L:
      For each undirected edge (i,j), weight w_ij > 0
      L_ij = -w_ij, L_ii += w_ij, L_jj += w_ij
    Then L is PSD and sparse with off-diagonal nnz determined by edges.
    Returns L (CSR) and num_edges.
    """
    # Sample upper-triangular edges with prob delta
    # Expected edges ~ delta * n(n-1)/2
    # We do it by sampling a mask on pairs; for large n, this is still OK up to ~ a few thousand.
    # If you go much larger, switch to edge sampling via binomial counts.
    triu_i, triu_j = np.triu_indices(n, k=1)
    mask = rng.random(size=triu_i.size) < delta
    i = triu_i[mask]
    j = triu_j[mask]
    m = i.size  # number of edges

    if m == 0:
        # No edges: Laplacian is zero
        return sp.csr_matrix((n, n), dtype=float), 0

    w = rng.uniform(weight_low, weight_high, size=m)

    # Off-diagonals: -w at (i,j) and (j,i)
    row = np.concatenate([i, j])
    col = np.concatenate([j, i])
    data = np.concatenate([-w, -w])

    # Diagonal: degree sums
    deg = np.zeros(n, dtype=float)
    np.add.at(deg, i, w)
    np.add.at(deg, j, w)  # Because it is undirected

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

    Returns Q and an estimated achieved kappa using eigsh on Q.
    """
    n = L.shape[0]

    if L.nnz == 0:
        # L = 0 => choose alpha=1 so Q=I (kappa=1)
        alpha = 1.0
        Q = sp.eye(n, format="csr") * alpha
        return Q, 1.0

    # Estimate lambda_max(L)
    # L is symmetric PSD; use largest algebraic eigenvalue
    lmax = float(eigsh(L, k=1, which="LA", return_eigenvectors=False, tol=tol, maxiter=max_eig_iter)[0])

    if target_kappa <= 1.0 + 1e-12:
        # Degenerate request: kappa=1 -> Q proportional to identity; pick alpha=1, ignore L
        Q = sp.eye(n, format="csr")
        return Q, 1.0

    alpha = max(lmax / (target_kappa - 1.0), eps)  # Because the smallest eigenvalue of Laplacian is 0.
    Q = (L + sp.eye(n, format="csr") * alpha).tocsr()

    # Optional: estimate achieved kappa on Q (cost: two eigsh calls)
    qmin = float(eigsh(Q, k=1, which="SA", return_eigenvectors=False, tol=tol, maxiter=max_eig_iter)[0])
    qmax = float(eigsh(Q, k=1, which="LA", return_eigenvectors=False, tol=tol, maxiter=max_eig_iter)[0])
    achieved = qmax / max(qmin, eps)
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
    parser.add_argument("--out_dir", type=str, default="../data/Q_sparsity")
    parser.add_argument("--n_list", type=str, default="100,500,1000",
                        help="Comma-separated list of n values.")
    parser.add_argument("--delta_list", type=str, default="0.01,0.05,0.1,0.5",
                        help="Comma-separated list of ER densities (edge probabilities).")
    parser.add_argument("--reps", type=int, default=10, help="Instances per (n,delta).")
    parser.add_argument("--kappa", type=float, default=100.0, help="Target condition number of Q.")
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed.")
    args = parser.parse_args()

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    delta_list = [float(x.strip()) for x in args.delta_list.split(",") if x.strip()]
    reps = args.reps

    rng_master = np.random.default_rng(args.seed)

    manifest = []
    for n in n_list:
        for delta in delta_list:
            for rep in range(reps):
                # Derive a deterministic per-instance seed
                seed = int(rng_master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed)

                # 1) Generate ER Laplacian L (PSD, sparse exactly by edges)
                L, m_edges = er_laplacian_psd(n=n, delta=delta, rng=rng)

                # 2) Form Q = L + alpha I to hit target kappa, then normalize trace(Q)/n=1
                Q_raw, achieved_kappa = build_Q_with_kappa(L, target_kappa=args.kappa)
                Q = normalize_trace(Q_raw)

                # After trace normalization, trace(Q)/n should be 1 (up to numerical tolerance)
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
                    delta=float(delta),
                    rep=int(rep),
                    seed=seed,
                    target_kappa=float(args.kappa),
                    achieved_kappa_est=float(achieved_kappa),
                    trace_per_dim_after_norm=float(trace_per_dim),
                    num_edges=int(m_edges),
                    offdiag_nnz=offdiag_nnz,
                    diag_nnz=diag_nnz,
                    Q_nnz_total=int(Q.nnz),
                )

                # Save
                fname = f"inst_n{n}_delta{delta:g}_rep{rep:02d}.npz"
                out_path = os.path.join(args.out_dir, f"n={n}", f"delta={delta:g}", fname)
                save_instance(out_path, Q, d, meta)

                manifest.append({"path": out_path, **asdict(meta)})

    # Save a manifest for easy loading/analysis
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest)} instances.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()