#!/usr/bin/env python3
"""
Generate datasets for the 2D-GMRF denoising experiment:

    min_x  sigma^{-2} ||x_tilde - x||_2^2
           + sum_{vertical edges} (x_i - x_j)^2
           + sum_{horizontal edges} (x_i - x_j)^2
           + gamma ||x||_0

The saved quadratic is in the standard form used by Run_2D_MRF.py:

    1/2 x^T Q x + c^T x + const + gamma ||x||_0

so

    Q     = 2 * (sigma^{-2} I + L_grid),
    c     = -2 * sigma^{-2} x_tilde,
    const = sigma^{-2} ||x_tilde||_2^2.

Experiment design:
- Grid size p is configurable via --grid_size_list.
- Noise variance sigma^2 is configurable via --sigma2_list.
- The latent signal is generated from local Gaussian shocks on s-by-s blocks.
- For each (p, sigma^2): generate R independent instances.

Outputs:
- One CSV per instance, matching src.utils.load_instance_2d_mrf:
    Row 1: dimension n = p^2
    Row 2: additive constant
    Row 3: support of the latent signal
    Row 4: latent signal vec(X)
    Row 5: linear term c
    Rows 6-(n+5): dense rows of Q
- A manifest.json with per-instance metadata.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DEFAULT_OUT_DIR = os.path.join(project_root, "data", "2D-MRF")


@dataclass
class InstanceMeta:
    grid_size: int
    n: int
    sigma2: float
    rep: int
    seed: int

    shock_size: int
    num_shocks: int
    shock_upper_left_1based: List[List[int]]

    support_size: int
    signal_l2: float
    observation_l2: float
    constant: float

    vectorization: str
    quadratic_form: str

    num_grid_edges: int
    offdiag_nnz: int
    diag_nnz: int
    Q_nnz_total: int
    Q_diag_min: float
    Q_diag_max: float


def parse_list(s: str, cast):
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
        return [cast(x.strip()) for x in s.split(",") if x.strip()]
    return [cast(s)]


def sigma2_to_str(sigma2: float) -> str:
    """
    Match the filename convention used by load_instance_2d_mrf, which searches
    with Python's default float string representation.
    """
    return str(float(sigma2))


def default_shock_size(grid_size: int) -> int:
    local_defaults = {
        10: 3,
        20: 4,
        40: 5,
        100: 10,
    }
    return local_defaults.get(grid_size, max(1, grid_size // 10))


def default_num_shocks(grid_size: int) -> int:
    local_defaults = {
        10: 2,
        20: 3,
        40: 5,
        100: 10,
    }
    return local_defaults.get(grid_size, max(1, grid_size // 10))


def shock_precision(shock_size: int) -> np.ndarray:
    """
    Build Omega_s on an s-by-s grid:
        diagonal entries are 4,
        horizontal/vertical neighbor entries are -1.
    """
    s = int(shock_size)
    n = s * s
    omega = 4.0 * np.eye(n, dtype=np.float64)

    for i in range(s):
        for j in range(s):
            idx = i * s + j
            if i + 1 < s:
                nbr = (i + 1) * s + j
                omega[idx, nbr] = -1.0
                omega[nbr, idx] = -1.0
            if j + 1 < s:
                nbr = i * s + (j + 1)
                omega[idx, nbr] = -1.0
                omega[nbr, idx] = -1.0

    return omega


def sample_shock(
    shock_size: int,
    shock_cholesky: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample W with vec(W) ~ N(0, Omega_s^{-1}).
    """
    z = rng.normal(size=shock_size * shock_size)
    w = np.linalg.solve(shock_cholesky.T, z)
    return w.reshape((shock_size, shock_size))


def generate_signal(
    grid_size: int,
    shock_size: int,
    num_shocks: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Generate the latent p-by-p signal from local Gaussian shocks.
    Returns the signal and 1-based upper-left block positions.
    """
    if shock_size < 1 or shock_size > grid_size:
        raise ValueError("shock_size must satisfy 1 <= shock_size <= grid_size.")
    if num_shocks < 1:
        raise ValueError("num_shocks must be positive.")

    x = np.zeros((grid_size, grid_size), dtype=np.float64)
    omega = shock_precision(shock_size)
    shock_cholesky = np.linalg.cholesky(omega)

    positions: List[List[int]] = []
    max_start = grid_size - shock_size + 1

    for _ in range(num_shocks):
        i0 = int(rng.integers(0, max_start))
        j0 = int(rng.integers(0, max_start))
        shock = sample_shock(shock_size=shock_size, shock_cholesky=shock_cholesky, rng=rng)
        x[i0:i0 + shock_size, j0:j0 + shock_size] += shock
        positions.append([i0 + 1, j0 + 1])

    return x, positions


def grid_quadratic_stats(grid_size: int, sigma2: float) -> Tuple[int, int, int, int, float, float]:
    n = grid_size * grid_size
    num_edges = 2 * grid_size * (grid_size - 1)
    offdiag_nnz = 2 * num_edges
    diag_nnz = n
    q_nnz_total = diag_nnz + offdiag_nnz

    if n == 1:
        min_degree = 0
        max_degree = 0
    elif grid_size == 2:
        min_degree = 2
        max_degree = 2
    else:
        min_degree = 2
        max_degree = 4

    sigma2_inv = 1.0 / float(sigma2)
    diag_min = 2.0 * (sigma2_inv + min_degree)
    diag_max = 2.0 * (sigma2_inv + max_degree)

    return num_edges, offdiag_nnz, diag_nnz, q_nnz_total, diag_min, diag_max


def format_float(x: float) -> str:
    return f"{float(x):.17g}"


def format_float_vector(values: np.ndarray) -> str:
    return ",".join(format_float(v) for v in values)


def format_int_vector(values: np.ndarray) -> str:
    return ",".join(str(int(v)) for v in values)


def write_quadratic_rows(f, grid_size: int, sigma2: float) -> None:
    """
    Stream dense rows of Q without materializing the full dense matrix.
    """
    n = grid_size * grid_size
    sigma2_inv = 1.0 / float(sigma2)
    row = np.zeros(n, dtype=np.float64)

    for idx in range(n):
        row.fill(0.0)
        i, j = divmod(idx, grid_size)
        degree = 0

        if i > 0:
            row[idx - grid_size] = -2.0
            degree += 1
        if i + 1 < grid_size:
            row[idx + grid_size] = -2.0
            degree += 1
        if j > 0:
            row[idx - 1] = -2.0
            degree += 1
        if j + 1 < grid_size:
            row[idx + 1] = -2.0
            degree += 1

        row[idx] = 2.0 * (sigma2_inv + degree)
        f.write(format_float_vector(row) + "\n")


def save_instance(
    out_path: str,
    x_true: np.ndarray,
    x_obs: np.ndarray,
    support: np.ndarray,
    sigma2: float,
    overwrite: bool,
) -> Tuple[np.ndarray, float]:
    """
    Save one instance as a CSV compatible with load_instance_2d_mrf.
    Returns c and const for metadata.
    """
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"Instance already exists: {out_path}. Pass --overwrite to replace it.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    grid_size = x_true.shape[0]
    n = grid_size * grid_size
    x_vec = x_true.reshape(-1)
    x_obs_vec = x_obs.reshape(-1)
    support_vec = support.reshape(-1).astype(np.int64)

    c = -2.0 * x_obs_vec / float(sigma2)
    const = float(np.dot(x_obs_vec, x_obs_vec) / float(sigma2))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        f.write(format_float(const) + "\n")
        f.write(format_int_vector(support_vec) + "\n")
        f.write(format_float_vector(x_vec) + "\n")
        f.write(format_float_vector(c) + "\n")
        write_quadratic_rows(f, grid_size=grid_size, sigma2=sigma2)

    return c, const


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Output directory for 2D-MRF CSV instances.")
    parser.add_argument("--grid_size_list", type=str, default="10,20,40",
                        help='Comma-separated list of p values, e.g. "10,20,40".')
    parser.add_argument("--sigma2_list", type=str, default="0.05,0.1,0.2",
                        help='Comma-separated list of sigma^2 values, e.g. "0.05,0.1,0.2".')
    parser.add_argument("--reps", type=int, default=5,
                        help="Instances per (grid_size, sigma2).")
    parser.add_argument("--rep_start", type=int, default=101,
                        help="First replicate id. Defaults to existing 2D-MRF convention.")
    parser.add_argument("--shock_size", type=int, default=None,
                        help="Side length s of each shock block. If omitted, use local defaults by grid size.")
    parser.add_argument("--num_shocks", type=int, default=None,
                        help="Number h of Gaussian shocks. If omitted, use local defaults by grid size.")
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing instance files.")
    args = parser.parse_args()

    grid_size_list = parse_list(args.grid_size_list, int)
    sigma2_list = parse_list(args.sigma2_list, float)
    reps = int(args.reps)

    if reps < 1:
        raise ValueError("reps must be positive.")
    for sigma2 in sigma2_list:
        if sigma2 <= 0:
            raise ValueError("All sigma2 values must be positive.")

    planned_instances = []
    for grid_size in grid_size_list:
        if grid_size < 1:
            raise ValueError("grid_size must be positive.")

        shock_size = int(args.shock_size) if args.shock_size is not None else default_shock_size(grid_size)
        num_shocks = int(args.num_shocks) if args.num_shocks is not None else default_num_shocks(grid_size)

        if shock_size < 1:
            raise ValueError("shock_size must be positive.")
        if shock_size > grid_size:
            raise ValueError("shock_size must be no larger than grid_size.")
        if num_shocks < 1:
            raise ValueError("num_shocks must be positive.")

        for sigma2 in sigma2_list:
            sigma2_str = sigma2_to_str(sigma2)
            for rep_offset in range(reps):
                rep = int(args.rep_start + rep_offset)
                fname = f"syntGrid{grid_size}-{shock_size}-{shock_size}-{sigma2_str}-{rep}_quad.csv"
                out_path = os.path.join(args.out_dir, fname)
                planned_instances.append((grid_size, shock_size, num_shocks, sigma2, rep, out_path))

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    if not args.overwrite:
        existing_paths = [out_path for *_, out_path in planned_instances if os.path.exists(out_path)]
        if os.path.exists(manifest_path):
            existing_paths.append(manifest_path)
        if existing_paths:
            preview = "\n".join(existing_paths[:10])
            extra = "" if len(existing_paths) <= 10 else f"\n... and {len(existing_paths) - 10} more"
            raise FileExistsError(
                "Refusing to overwrite existing files. Pass --overwrite to replace them:\n"
                f"{preview}{extra}"
            )

    rng_master = np.random.default_rng(int(args.seed))

    manifest = []
    total = 0
    for grid_size, shock_size, num_shocks, sigma2, rep, out_path in planned_instances:
        seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)

        # 1) Generate latent signal and support
        x_true, shock_positions = generate_signal(
            grid_size=grid_size,
            shock_size=shock_size,
            num_shocks=num_shocks,
            rng=rng,
        )
        support = (np.abs(x_true) > 0.0).astype(np.int64)

        # 2) Add iid N(0, sigma^2) observation noise
        noise = rng.normal(scale=np.sqrt(float(sigma2)), size=x_true.shape)
        x_obs = x_true + noise

        # 3) Save the standard-form quadratic instance
        _, const = save_instance(
            out_path=out_path,
            x_true=x_true,
            x_obs=x_obs,
            support=support,
            sigma2=sigma2,
            overwrite=bool(args.overwrite),
        )

        # 4) Metadata / sparsity stats
        num_edges, offdiag_nnz, diag_nnz, q_nnz_total, diag_min, diag_max = grid_quadratic_stats(
            grid_size=grid_size,
            sigma2=sigma2,
        )
        n = grid_size * grid_size
        meta = InstanceMeta(
            grid_size=int(grid_size),
            n=int(n),
            sigma2=float(sigma2),
            rep=int(rep),
            seed=int(seed),
            shock_size=int(shock_size),
            num_shocks=int(num_shocks),
            shock_upper_left_1based=shock_positions,
            support_size=int(np.count_nonzero(support)),
            signal_l2=float(np.linalg.norm(x_true.reshape(-1))),
            observation_l2=float(np.linalg.norm(x_obs.reshape(-1))),
            constant=float(const),
            vectorization="row-major",
            quadratic_form="0.5*x^T*Q*x + c^T*x + const",
            num_grid_edges=int(num_edges),
            offdiag_nnz=int(offdiag_nnz),
            diag_nnz=int(diag_nnz),
            Q_nnz_total=int(q_nnz_total),
            Q_diag_min=float(diag_min),
            Q_diag_max=float(diag_max),
        )

        manifest.append({"path": out_path, **asdict(meta)})
        total += 1

    os.makedirs(args.out_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {total} instances.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
