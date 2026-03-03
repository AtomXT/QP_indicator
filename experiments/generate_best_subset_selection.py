#!/usr/bin/env python3
"""
Generate datasets for Cardinality-Constrained Best Subset Selection (BSS) experiment:

    min_{beta,z}  1/2 ||y - X beta||_2^2 + (gamma/2)||beta||_2^2
    s.t.          beta_i (1 - z_i) = 0,  z_i in {0,1}
                  sum_i z_i <= k

We fix:
- p = 500
- correlation in X: AR(1) with rho (default 0.6)
- ridge gamma (default 1e-2)
- noise sigma (default 0.5)
- amplitude a (default 1.0)

We vary:
- n in {250, 1000}  (configurable via --n_list)
- k in a list (configurable via --k_list)
- reps per (n,k)

We also precompute quantities used by both formulations:
- G = X^T X (dense)
- c = X^T y
and CORe bound ingredients:
- ridge solution beta_ridge = (G + gamma I)^{-1} c
- bar_beta_i = M * |beta_ridge_i| + beta_min
- H_i = sum_{j != i} |G_ij| * bar_beta_j + |c_i|

Outputs:
- One .npz per instance, containing:
    X (float32), y (float64),
    G (float64), c (float64),
    beta_star (float64), support (int64),
    bar_beta (float64), H (float64),
    metadata dict (as JSON string).

Also writes a manifest.json listing all instances + metadata.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class InstanceMeta:
    # Design parameters
    n: int
    p: int
    rep: int
    seed: int

    rho: float
    gamma: float
    sigma: float
    amplitude: float

    # Truth
    s_true: int
    support_sorted: List[int]

    # Basic sanity stats
    X_col_standardized: bool
    y_norm: float
    beta_star_l2: float
    beta_ridge_l2: float

    # Matrix stats
    G_trace: float
    G_diag_min: float
    G_diag_max: float


def ar1_rows(n: int, p: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate X with AR(1) covariance across columns:
      Cov(X[:,i], X[:,j]) = rho^{|i-j|}
    using the recursion (stationary AR(1) in feature index):
      x_0 ~ N(0,1)
      x_j = rho x_{j-1} + sqrt(1-rho^2) e_j, e_j~N(0,1)
    Returns X as float64.
    """
    if not (0.0 <= abs(rho) < 1.0):
        raise ValueError("rho must satisfy |rho| < 1 for stationary AR(1).")

    X = np.empty((n, p), dtype=np.float64)
    eps = rng.normal(size=(n, p))
    X[:, 0] = eps[:, 0]
    scale = np.sqrt(max(1.0 - rho * rho, 0.0))
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + scale * eps[:, j]
    return X


def standardize_columns(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Column-standardize: mean 0, variance 1 (ddof=0).
    """
    mu = X.mean(axis=0)
    Xc = X - mu
    var = (Xc * Xc).mean(axis=0)
    std = np.sqrt(np.maximum(var, eps))
    return Xc / std


def make_beta_star(p: int, amplitude: float, rng: np.random.Generator, s_true: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sparse ground-truth beta_star with fixed sparsity.

      - s_true is independent of k (cardinality constraint).
      - Entries on support are +/- amplitude.
      - Allows k < s_true (misspecified regime).

    Returns:
      beta_star (p,), support_sorted (indices of true support)
    """
    if not (1 <= s_true <= p):
        raise ValueError("s_true must satisfy 1 <= s_true <= p.")

    support = rng.choice(p, size=s_true, replace=False)
    support.sort()

    signs = rng.choice([-1.0, 1.0], size=s_true)

    beta_star = np.zeros(p, dtype=np.float64)
    beta_star[support] = signs * float(amplitude)

    return beta_star, support

def compute_core_bounds(G: np.ndarray, c: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute:
      beta_ridge = (G + gamma I)^{-1} c
      bar_beta_i = M*|beta_ridge_i| + beta_min
      H_i = sum_{j != i} |G_ij| bar_beta_j + |c_i|
    Returns (beta_ridge, bar_beta, H).
    """
    p = G.shape[0]
    A = G + gamma * np.eye(p, dtype=np.float64)
    beta_ridge = np.linalg.solve(A, c)

    M, beta_min = 10, 1e-3
    bar_beta = M * np.abs(beta_ridge) + beta_min

    absG = np.abs(G)
    absdiag = np.abs(np.diag(G))
    # absG @ bar_beta includes diagonal term; remove it to get sum_{j!=i}
    H = absG @ bar_beta - absdiag * bar_beta + np.abs(c)
    return beta_ridge, bar_beta, H


def save_instance(
    out_path: str,
    X: np.ndarray,
    y: np.ndarray,
    G: np.ndarray,
    c: np.ndarray,
    beta_star: np.ndarray,
    support: np.ndarray,
    beta_ridge: np.ndarray,
    bar_beta: np.ndarray,
    H: np.ndarray,
    meta: InstanceMeta,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "X": X.astype(np.float32),  # saves space; solvers can cast back to float64
        "y": y.astype(np.float64),
        "G": G.astype(np.float64),
        "c": c.astype(np.float64),
        "beta_star": beta_star.astype(np.float64),
        "support": support.astype(np.int64),
        "beta_ridge": beta_ridge.astype(np.float64),
        "bar_beta": bar_beta.astype(np.float64),
        "H": H.astype(np.float64),
        "meta_json": np.array(json.dumps(asdict(meta)), dtype=object),
    }
    np.savez_compressed(out_path, **payload)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="../data/bss_cardinality",
                        help="Output directory for instances.")
    parser.add_argument("--p_list", type=str, default="500",
                        help="Comma-separated list of p values.")
    parser.add_argument("--s_true", type=int, default=20,
                        help="True sparsity of beta_star (independent of k).")
    parser.add_argument("--n_list", type=str, default="250,1000", help="Comma-separated list of n values.")
    parser.add_argument("--reps", type=int, default=5, help="Instances per (n,p).")
    parser.add_argument("--rho", type=float, default=0.6, help="AR(1) correlation parameter.")
    parser.add_argument("--gamma", type=float, default=1e-2, help="Ridge parameter (stabilized).")
    parser.add_argument("--sigma", type=float, default=0.5, help="Noise std.")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Nonzero coefficient amplitude.")
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed.")
    parser.add_argument("--standardize_X", action="store_true", default=True,
                        help="Column-standardize X (on by default).")
    parser.add_argument("--no_standardize_X", action="store_false", dest="standardize_X",
                        help="Disable column standardization of X.")


    args = parser.parse_args()

    p_list = parse_int_list(args.p_list)
    n_list = parse_int_list(args.n_list)
    reps = int(args.reps)

    rng_master = np.random.default_rng(int(args.seed))

    manifest = []
    total = 0
    for p in p_list:
        for n in n_list:
            for rep in range(reps):
                seed = int(rng_master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed)

                # 1) X
                X = ar1_rows(n=n, p=p, rho=float(args.rho), rng=rng)
                if args.standardize_X:
                    X = standardize_columns(X)

                # 2) beta_star and y
                beta_star, support = make_beta_star(p=p, amplitude=float(args.amplitude), rng=rng, s_true=20)
                eps = rng.normal(scale=float(args.sigma), size=n)
                y = X @ beta_star + eps

                # 3) G and c
                G = X.T @ X
                c = X.T @ y

                # 4) CORe bound ingredients
                beta_ridge, bar_beta, H = compute_core_bounds(G=G, c=c, gamma=float(args.gamma))

                # 5) Metadata
                meta = InstanceMeta(
                    n=int(n),
                    p=int(p),
                    rep=int(rep),
                    seed=int(seed),
                    rho=float(args.rho),
                    gamma=float(args.gamma),
                    sigma=float(args.sigma),
                    amplitude=float(args.amplitude),
                    s_true=int(len(support)),
                    support_sorted=[int(i) for i in support.tolist()],
                    X_col_standardized=bool(args.standardize_X),
                    y_norm=float(np.linalg.norm(y)),
                    beta_star_l2=float(np.linalg.norm(beta_star)),
                    beta_ridge_l2=float(np.linalg.norm(beta_ridge)),
                    G_trace=float(np.trace(G)),
                    G_diag_min=float(np.min(np.diag(G))),
                    G_diag_max=float(np.max(np.diag(G))),
                )

                # 6) Save
                fname = f"inst_p{p}_n{n}_rep{rep:02d}.npz"
                out_path = os.path.join(args.out_dir, f"p={p}", f"n={n}", fname)
                save_instance(
                    out_path=out_path,
                    X=X,
                    y=y,
                    G=G,
                    c=c,
                    beta_star=beta_star,
                    support=support,
                    beta_ridge=beta_ridge,
                    bar_beta=bar_beta,
                    H=H,
                    meta=meta,
                )

                manifest.append({"path": out_path, **asdict(meta)})
                total += 1

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {total} instances.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()