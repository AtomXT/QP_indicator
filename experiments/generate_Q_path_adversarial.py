#!/usr/bin/env python3
"""
Generate near-worst-case path instances for QP-with-indicators:

    min_{x,z}  1/2 x^T Q x + c^T x + lambda^T z
    s.t.       x_i (1 - z_i) = 0,  z_i in {0,1}

Experiment design:
- Q structure: tridiagonal path graph with Q_ii = 1 and Q_{i,i+1} = rho
- Default rho = 0.45, so Q is uniformly positive definite
- Default tau = 0.75, with lambda_i = tau^2 / 2 because Q_ii = 1
- Variants:
    1) exact_adversarial_path: c_i = -1
    2) perturbed_c_adversarial_path: c_i = -1 + eps_i,
       eps_i ~ Uniform[-perturbation, perturbation]

Outputs:
- One .npz per instance, containing:
    Q as CSR (data, indices, indptr, shape),
    c vector,
    lambda vector,
    metadata dict (as a JSON string).
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DEFAULT_OUT_DIR = os.path.join(project_root, "data", "Q_path_adversarial")

EXACT_VARIANT = "exact_adversarial_path"
PERTURBED_C_VARIANT = "perturbed_c_adversarial_path"
DEFAULT_VARIANTS = (EXACT_VARIANT, PERTURBED_C_VARIANT)
VARIANT_SEED_IDS = {
    EXACT_VARIANT: 0,
    PERTURBED_C_VARIANT: 1,
}


@dataclass
class InstanceMeta:
    n: int
    rep: int
    seed: int
    variant: str
    graph_type: str

    rho: float
    tau: float
    lambda_value: float
    perturbation_level: float

    min_eig_est: float
    max_eig_est: float
    trace_per_dim: float
    tau_gt_one_minus_rho: bool

    num_edges: int
    offdiag_nnz: int
    diag_nnz: int
    Q_nnz_total: int


def parse_list(s: str, cast=str):
    """
    Accepts:
      "1,2,3"
      "[1,2,3]"
      "1"
    """
    s = s.strip()
    if s.startswith("["):
        arr = json.loads(s)
        return [cast(x) for x in arr]
    if "," in s:
        return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]
    return [cast(s)]


def float_label(value: float) -> str:
    return f"{float(value):g}"


def instance_seed(base_seed: int, variant: str, n: int, rep: int) -> int:
    """
    Stable seed for a single instance, independent of generation order.
    """
    if variant not in VARIANT_SEED_IDS:
        raise ValueError(f"Unknown variant: {variant}")
    seq = np.random.SeedSequence([int(base_seed), VARIANT_SEED_IDS[variant], int(n), int(rep)])
    rng = np.random.default_rng(seq)
    return int(rng.integers(0, 2**31 - 1))


def path_tridiagonal_Q(n: int, rho: float) -> sp.csr_matrix:
    """
    Build Q with Q_ii = 1 and Q_{i,i+1} = Q_{i+1,i} = rho.
    """
    diag = np.ones(n, dtype=float)
    if n <= 1:
        return sp.diags(diag, offsets=0, format="csr")

    off = float(rho) * np.ones(n - 1, dtype=float)
    return sp.diags([off, diag, off], offsets=[-1, 0, 1], shape=(n, n), format="csr")


def tridiagonal_toeplitz_eigs(n: int, rho: float) -> Tuple[float, float]:
    """
    Eigenvalue range for diag=1 and constant first off-diagonal=rho.
    """
    if n <= 0:
        return np.nan, np.nan
    k = np.arange(1, n + 1, dtype=float)
    eigs = 1.0 + 2.0 * float(rho) * np.cos(k * np.pi / (n + 1.0))
    return float(eigs.min()), float(eigs.max())


def is_path_graph_Q(Q: sp.csr_matrix, rho: float, tol: float = 1e-10) -> bool:
    """
    Check the intended path support and weights.
    """
    n = Q.shape[0]
    expected = path_tridiagonal_Q(n, rho)
    diff = (Q - expected).tocoo()
    if diff.nnz == 0:
        return True
    return bool(np.max(np.abs(diff.data)) <= tol)


def build_instance(
    n: int,
    rep: int,
    seed: int,
    variant: str,
    rho: float = 0.45,
    tau: float = 0.75,
    perturbation: float = 0.01,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, InstanceMeta]:
    """
    Build one exact or c-perturbed adversarial path instance.
    """
    if variant not in DEFAULT_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if n < 1:
        raise ValueError("n must be positive.")
    if perturbation < 0:
        raise ValueError("perturbation must be nonnegative.")

    Q = path_tridiagonal_Q(n=n, rho=rho)
    min_eig, max_eig = tridiagonal_toeplitz_eigs(n=n, rho=rho)
    if min_eig <= 0:
        raise ValueError(
            f"Q is not positive definite for n={n}, rho={rho}: min eigenvalue {min_eig}."
        )

    rng = np.random.default_rng(seed)
    if variant == EXACT_VARIANT:
        c = -np.ones(n, dtype=float)
        perturbation_level = 0.0
    else:
        eps = rng.uniform(-perturbation, perturbation, size=n)
        c = -np.ones(n, dtype=float) + eps
        perturbation_level = float(perturbation)

    lambda_value = float(tau) ** 2 / 2.0
    lam = lambda_value * np.ones(n, dtype=float)

    offdiag_nnz = int(Q.nnz - np.count_nonzero(Q.diagonal()))
    diag_nnz = int(np.count_nonzero(Q.diagonal()))
    meta = InstanceMeta(
        n=int(n),
        rep=int(rep),
        seed=int(seed),
        variant=variant,
        graph_type="path",
        rho=float(rho),
        tau=float(tau),
        lambda_value=lambda_value,
        perturbation_level=perturbation_level,
        min_eig_est=min_eig,
        max_eig_est=max_eig,
        trace_per_dim=float(Q.diagonal().sum()) / n,
        tau_gt_one_minus_rho=bool(float(tau) > 1.0 - float(rho)),
        num_edges=max(0, int(n) - 1),
        offdiag_nnz=offdiag_nnz,
        diag_nnz=diag_nnz,
        Q_nnz_total=int(Q.nnz),
    )
    return Q, c, lam, meta


def instance_path(
    out_dir: str,
    variant: str,
    n: int,
    rep: int,
    rho: float,
    tau: float,
    perturbation: float,
) -> str:
    pert_for_path = 0.0 if variant == EXACT_VARIANT else float(perturbation)
    return os.path.join(
        out_dir,
        variant,
        f"n={int(n)}",
        f"rho={float_label(rho)}",
        f"tau={float_label(tau)}",
        f"perturb={float_label(pert_for_path)}",
        f"inst_n{int(n)}_{variant}_rep{int(rep):02d}.npz",
    )


def save_instance(
    out_path: str,
    Q: sp.csr_matrix,
    c: np.ndarray,
    lam: np.ndarray,
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
        "c": c.astype(np.float64),
        "lambda": lam.astype(np.float64),
        "meta_json": np.array(json.dumps(asdict(meta)), dtype=object),
    }
    np.savez_compressed(out_path, **payload)


def generate_instance_file(
    out_dir: str,
    n: int,
    rep: int,
    seed: int,
    variant: str,
    rho: float = 0.45,
    tau: float = 0.75,
    perturbation: float = 0.01,
    overwrite: bool = False,
) -> Tuple[str, InstanceMeta]:
    out_path = instance_path(
        out_dir=out_dir,
        variant=variant,
        n=n,
        rep=rep,
        rho=rho,
        tau=tau,
        perturbation=perturbation,
    )
    if os.path.exists(out_path) and not overwrite:
        obj = np.load(out_path, allow_pickle=True)
        meta = InstanceMeta(**json.loads(str(obj["meta_json"])))
        return out_path, meta

    Q, c, lam, meta = build_instance(
        n=n,
        rep=rep,
        seed=seed,
        variant=variant,
        rho=rho,
        tau=tau,
        perturbation=perturbation,
    )
    if not is_path_graph_Q(Q, rho=rho):
        raise ValueError(f"Generated Q does not match the requested path graph for {out_path}.")
    if not np.allclose(lam, float(tau) ** 2 / 2.0):
        raise ValueError("Generated lambda is inconsistent with tau^2 / 2.")

    save_instance(out_path, Q, c, lam, meta)
    return out_path, meta


def generate_grid(
    out_dir: str,
    n_list: Iterable[int],
    reps: int,
    variants: Iterable[str],
    rho: float = 0.45,
    tau_list: Iterable[float] = (0.75,),
    perturbation: float = 0.01,
    seed: int = 2026,
    overwrite: bool = False,
) -> List[dict]:
    manifest = []

    for variant in variants:
        for n in n_list:
            for rep in range(reps):
                for tau in tau_list:
                    inst_seed = instance_seed(seed, variant, int(n), int(rep))
                    out_path, meta = generate_instance_file(
                        out_dir=out_dir,
                        n=int(n),
                        rep=int(rep),
                        seed=inst_seed,
                        variant=variant,
                        rho=rho,
                        tau=float(tau),
                        perturbation=perturbation,
                        overwrite=overwrite,
                    )
                    manifest.append({"path": out_path, **asdict(meta)})
                    print(f"Saved {variant}, n={n}, tau={tau}, rep={rep}: {out_path}")

    manifest_path = os.path.join(out_dir, "manifest.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(manifest)} instances.")
    print(f"Manifest: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--n_list",
        type=str,
        default="100,1000",
        help="Comma-separated list of n values."
    )
    parser.add_argument("--reps", type=int, default=5, help="Instances per n and variant.")
    parser.add_argument(
        "--variant_list",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variant list."
    )
    parser.add_argument("--rho", type=float, default=0.45)
    parser.add_argument(
        "--tau_list",
        type=str,
        default="0.6,0.7,0.8,0.9",
        help='e.g. "0.7" or "0.7,0.9" or "[0.7,0.9]"'
    )
    parser.add_argument("--perturbation", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    n_list = parse_list(args.n_list, int)
    tau_list = parse_list(args.tau_list, float)
    variants = parse_list(args.variant_list, str)
    generate_grid(
        out_dir=args.out_dir,
        n_list=n_list,
        reps=args.reps,
        variants=variants,
        rho=args.rho,
        tau_list=tau_list,
        perturbation=args.perturbation,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
