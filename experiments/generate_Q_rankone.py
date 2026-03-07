#!/usr/bin/env python3
"""
Generate datasets for sparse QP-with-indicators + rank-one term (NO mu in dataset):

    min_{x,z}  1/2 x^T Q x + (a^T x)^2 + mu^T z
    s.t.       x_i (1 - z_i) = 0,  z_i in {0,1}

Dataset contains ONLY the instance structure (Q, a) + metadata.
The penalty vector mu (>=0) should be provided at solve time (e.g., swept over).

Design:
- Q structure: Erdős–Rényi sparsity pattern via a weighted graph Laplacian
              => Q is PSD and has EXACT ER edge sparsity on off-diagonals.
- Condition number control: kappa(Q) ~= target_kappa (default 100)
- Scale normalization: trace(Q)/n = 1
- Rank-one vector a: generated dense by default; normalized to ||a||_2 = 1
- For each (n, delta): generate R independent instances (default 10)

Outputs:
- One .npz per instance, containing:
    Q as CSR (data, indices, indptr, shape),
    a vector,
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
    delta: float
    rep: int
    seed: int

    target_kappa: float
    achieved_kappa_est: float
    trace_per_dim_after_norm: float

    # a generation / scaling
    a_dist: str
    a_sparse_frac: float
    a_norm_after: float

    # sparsity stats
    num_edges: int
    offdiag_nnz: int
    diag_nnz: int
    Q_nnz_total: int


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
    L is PSD and sparse with off-diagonal nnz determined by edges.
    Returns L (CSR) and num_edges.
    """
    # Sample upper-triangular edges with prob delta (EXACT ER mask realization)
    triu_i, triu_j = np.triu_indices(n, k=1)
    mask = rng.random(size=triu_i.size) < delta
    i = triu_i[mask]
    j = triu_j[mask]
    m = i.size

    if m == 0:
        return sp.csr_matrix((n, n), dtype=float), 0

    w = rng.uniform(weight_low, weight_high, size=m)

    # Off-diagonals: -w at (i,j) and (j,i)
    row = np.concatenate([i, j])
    col = np.concatenate([j, i])
    data = np.concatenate([-w, -w])

    # Diagonal: degree sums
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

    Returns Q and an estimated achieved kappa using eigsh on Q.
    """
    n = L.shape[0]

    if L.nnz == 0:
        # L = 0 => choose alpha=1 so Q=I (kappa=1)
        alpha = 1.0
        Q = sp.eye(n, format="csr") * alpha
        return Q, 1.0

    # Estimate lambda_max(L)
    lmax = float(
        eigsh(
            L,
            k=1,
            which="LA",
            return_eigenvectors=False,
            tol=tol,
            maxiter=max_eig_iter,
        )[0]
    )

    if target_kappa <= 1.0 + 1e-12:
        Q = sp.eye(n, format="csr")
        return Q, 1.0

    alpha = max(lmax / (target_kappa - 1.0), eps)
    Q = (L + sp.eye(n, format="csr") * alpha).tocsr()

    # Estimate achieved kappa (two eigsh calls)
    qmin = float(
        eigsh(
            Q,
            k=1,
            which="SA",
            return_eigenvectors=False,
            tol=tol,
            maxiter=max_eig_iter,
        )[0]
    )
    qmax = float(
        eigsh(
            Q,
            k=1,
            which="LA",
            return_eigenvectors=False,
            tol=tol,
            maxiter=max_eig_iter,
        )[0]
    )
    achieved = qmax / max(qmin, eps)
    return Q, achieved


def normalize_trace(Q: sp.csr_matrix) -> sp.csr_matrix:
    """Scale Q so that trace(Q)/n = 1."""
    n = Q.shape[0]
    tr = float(Q.diagonal().sum())
    if tr <= 0:
        return Q
    scale = 1.0 / (tr / n)
    return (Q * scale).tocsr()


def generate_a(
    n: int,
    rng: np.random.Generator,
    a_dist: str,
    a_sparse_frac: float,
) -> np.ndarray:
    """
    Generate a vector a, optionally sparsified, then normalized to ||a||_2 = 1.
    """
    if a_dist == "normal":
        a = rng.normal(size=n)
    elif a_dist == "rademacher":
        a = rng.choice([-1.0, 1.0], size=n)
    else:
        raise ValueError(f"Unknown a_dist: {a_dist}")

    if a_sparse_frac > 0.0:
        k = max(1, int(round(a_sparse_frac * n)))
        idx = rng.choice(n, size=k, replace=False)
        a2 = np.zeros(n, dtype=float)
        a2[idx] = a[idx]
        a = a2

    norm = float(np.linalg.norm(a))
    if norm <= 0:
        # Extremely unlikely unless n=0; still guard
        a = np.zeros(n, dtype=float)
    else:
        a = a / norm
    return a.astype(np.float64)


def save_instance(
    out_path: str,
    Q: sp.csr_matrix,
    a: np.ndarray,
    meta: InstanceMeta,
) -> None:
    """Save one instance as a compressed NPZ with CSR arrays + vectors + JSON metadata."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "Q_data": Q.data,
        "Q_indices": Q.indices,
        "Q_indptr": Q.indptr,
        "Q_shape": np.array(Q.shape, dtype=np.int64),
        "a": a.astype(np.float64),
        "meta_json": np.array(json.dumps(asdict(meta)), dtype=object),
    }
    np.savez_compressed(out_path, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="../data/Q_rankone")
    parser.add_argument("--n_list", type=str, default="500",
                        help='Comma-separated list of n values, e.g. "100,500,1000".')
    parser.add_argument("--delta_list", type=str, default="0.1",
                        help='Comma-separated list of ER densities, e.g. "0.01,0.1".')
    parser.add_argument("--reps", type=int, default=5, help="Instances per (n,delta).")
    parser.add_argument("--kappa", type=float, default=100.0, help="Target condition number of Q.")
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed.")

    # rank-one vector a
    parser.add_argument("--a_dist", type=str, default="normal",
                        choices=["normal", "rademacher"],
                        help="Distribution for a before normalization.")
    parser.add_argument("--a_sparse_frac", type=float, default=0.0,
                        help="If >0, make a sparse by keeping this fraction of entries, then renormalize. "
                             "E.g. 0.1 keeps 10% nonzeros. Default 0 (dense).")

    args = parser.parse_args()

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    delta_list = [float(x.strip()) for x in args.delta_list.split(",") if x.strip()]
    reps = int(args.reps)

    if not (0.0 <= args.a_sparse_frac < 1.0):
        raise ValueError("a_sparse_frac must be in [0, 1). Use 0 for dense.")

    rng_master = np.random.default_rng(args.seed)

    manifest = []
    for n in n_list:
        for delta in delta_list:
            for rep in range(reps):
                seed = int(rng_master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed)

                # 1) Generate ER Laplacian L (PSD, sparse)
                L, m_edges = er_laplacian_psd(n=n, delta=delta, rng=rng)

                # 2) Form Q = L + alpha I to hit target kappa, then normalize trace(Q)/n=1
                Q_raw, achieved_kappa = build_Q_with_kappa(L, target_kappa=args.kappa)
                Q = normalize_trace(Q_raw)
                trace_per_dim = float(Q.diagonal().sum()) / n

                # 3) Generate a and normalize ||a||_2=1
                a = generate_a(
                    n=n,
                    rng=rng,
                    a_dist=args.a_dist,
                    a_sparse_frac=args.a_sparse_frac,
                )
                a_norm_after = float(np.linalg.norm(a))

                # Sparsity stats
                diag = Q.diagonal()
                offdiag_nnz = int(Q.nnz - np.count_nonzero(diag))
                diag_nnz = int(np.count_nonzero(diag))

                meta = InstanceMeta(
                    n=int(n),
                    delta=float(delta),
                    rep=int(rep),
                    seed=int(seed),
                    target_kappa=float(args.kappa),
                    achieved_kappa_est=float(achieved_kappa),
                    trace_per_dim_after_norm=float(trace_per_dim),
                    a_dist=str(args.a_dist),
                    a_sparse_frac=float(args.a_sparse_frac),
                    a_norm_after=float(a_norm_after),
                    num_edges=int(m_edges),
                    offdiag_nnz=int(offdiag_nnz),
                    diag_nnz=int(diag_nnz),
                    Q_nnz_total=int(Q.nnz),
                )

                fname = f"inst_n{n}_delta{delta:g}_rep{rep:02d}.npz"
                out_path = os.path.join(args.out_dir, f"n={n}", f"delta={delta:g}", fname)
                save_instance(out_path, Q, a, meta)

                manifest.append({"path": out_path, **asdict(meta)})

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest)} instances.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()