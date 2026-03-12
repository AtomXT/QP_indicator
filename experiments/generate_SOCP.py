#!/usr/bin/env python3
"""
Generate datasets for SOCP-with-indicators experiment:

    min_{x, x0, z}  c^T x + x0 + mu^T z
    s.t.            x_i (1 - z_i) = 0,   z_i in {0,1}
                    x0 >= sqrt(sigma^2 + x^T Q x)

Equivalently, since x0 is tight at optimality,

    min_{x, z}  c^T x + sqrt(sigma^2 + x^T Q x) + mu^T z
    s.t.        x_i (1 - z_i) = 0,   z_i in {0,1}

Experiment design:
- Q structure: Erdős–Rényi sparsity pattern, implemented via a weighted graph Laplacian
               so Q is PSD and has EXACTLY the ER edge sparsity on off-diagonals.
- Condition number fixed: kappa(Q) ~= target_kappa
- Normalize scale across settings: trace(Q)/n = 1
- For each (n, delta): generate R independent instances

Outputs:
- One .npz per instance, containing:
    Q as CSR (data, indices, indptr, shape),
    c vector,
    sigma scalar,
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

    num_edges: int
    offdiag_nnz: int
    diag_nnz: int
    Q_nnz_total: int

    sigma: float
    c_dist: str


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
    triu_i, triu_j = np.triu_indices(n, k=1)
    mask = rng.random(size=triu_i.size) < delta
    i = triu_i[mask]
    j = triu_j[mask]
    m = i.size

    if m == 0:
        return sp.csr_matrix((n, n), dtype=float), 0

    w = rng.uniform(weight_low, weight_high, size=m)

    row = np.concatenate([i, j])
    col = np.concatenate([j, i])
    data = np.concatenate([-w, -w])

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
    set Q = L + alpha I so that
        kappa(Q) = (lambda_max(L)+alpha)/alpha ~= target_kappa

    alpha = lambda_max(L) / (target_kappa - 1)

    Returns Q and an estimated achieved kappa using eigsh on Q.
    """
    n = L.shape[0]

    if L.nnz == 0:
        alpha = 1.0
        Q = sp.eye(n, format="csr") * alpha
        return Q, 1.0

    lmax = float(
        eigsh(
            L, k=1, which="LA",
            return_eigenvectors=False,
            tol=tol, maxiter=max_eig_iter
        )[0]
    )

    if target_kappa <= 1.0 + 1e-12:
        Q = sp.eye(n, format="csr")
        return Q, 1.0

    alpha = max(lmax / (target_kappa - 1.0), eps)
    Q = (L + sp.eye(n, format="csr") * alpha).tocsr()

    qmin = float(
        eigsh(
            Q, k=1, which="SA",
            return_eigenvectors=False,
            tol=tol, maxiter=max_eig_iter
        )[0]
    )
    qmax = float(
        eigsh(
            Q, k=1, which="LA",
            return_eigenvectors=False,
            tol=tol, maxiter=max_eig_iter
        )[0]
    )
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


def generate_c(
    n: int,
    rng: np.random.Generator,
    dist: str = "normal",
    normalize_l2: bool = True,
) -> np.ndarray:
    """
    Generate linear-cost vector c.
    """
    if dist == "normal":
        c = rng.normal(size=n)
    elif dist == "uniform":
        c = rng.uniform(-1.0, 1.0, size=n)
    else:
        raise ValueError(f"Unsupported c_dist: {dist}")

    if normalize_l2:
        norm_c = np.linalg.norm(c)
        if norm_c > 0:
            c = c / norm_c
    return c.astype(np.float64)


def save_instance(
    out_path: str,
    Q: sp.csr_matrix,
    c: np.ndarray,
    sigma: float,
    meta: InstanceMeta,
) -> None:
    """
    Save one instance as a compressed NPZ with CSR arrays + vectors + scalar + JSON metadata.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "Q_data": Q.data,
        "Q_indices": Q.indices,
        "Q_indptr": Q.indptr,
        "Q_shape": np.array(Q.shape, dtype=np.int64),
        "c": c.astype(np.float64),
        "sigma": np.array([sigma], dtype=np.float64),
        "meta_json": np.array(json.dumps(asdict(meta)), dtype=object),
    }
    np.savez_compressed(out_path, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="../data/SOCP_indicator")
    parser.add_argument("--n_list", type=str, default="500",
                        help='Comma-separated list, e.g. "100,500,1000"')
    parser.add_argument("--delta_list", type=str, default="0.01,0.1,0.5,0.99",
                        help='Comma-separated ER edge probabilities')
    parser.add_argument("--reps", type=int, default=5,
                        help="Number of instances per (n, delta)")
    parser.add_argument("--kappa", type=float, default=100.0,
                        help="Target condition number of Q")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Scalar sigma in sqrt(sigma^2 + x^T Q x)")
    parser.add_argument("--c_dist", type=str, default="uniform", choices=["normal", "uniform"],
                        help="Distribution for c")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Base RNG seed")
    args = parser.parse_args()

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    delta_list = [float(x.strip()) for x in args.delta_list.split(",") if x.strip()]
    reps = args.reps

    rng_master = np.random.default_rng(args.seed)

    manifest = []
    for n in n_list:
        for delta in delta_list:
            for rep in range(reps):
                seed = int(rng_master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed)

                # 1) Generate sparse PSD Q
                L, m_edges = er_laplacian_psd(n=n, delta=delta, rng=rng)
                Q_raw, achieved_kappa = build_Q_with_kappa(L, target_kappa=args.kappa)
                Q = normalize_trace(Q_raw)
                trace_per_dim = float(Q.diagonal().sum()) / n

                # 2) Generate c
                c = generate_c(n=n, rng=rng, dist=args.c_dist, normalize_l2=True)

                # 3) Sigma
                sigma = float(args.sigma)

                # 4) Stats / metadata
                diag = Q.diagonal()
                offdiag_nnz = int(Q.nnz - np.count_nonzero(diag))
                diag_nnz = int(np.count_nonzero(diag))

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
                    sigma=sigma,
                    c_dist=args.c_dist,
                )

                # 5) Save
                fname = f"inst_n{n}_delta{delta:g}_rep{rep:02d}.npz"
                out_path = os.path.join(args.out_dir, f"n={n}", f"delta={delta:g}", fname)
                save_instance(out_path, Q, c, sigma, meta)

                manifest.append({
                    "path": out_path,
                    **asdict(meta),
                })

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(manifest)} instances.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()