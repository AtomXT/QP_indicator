#!/usr/bin/env python3
"""
Run near-worst-case path QP-with-indicator experiments.

Instances have objective

    min_{x,z}  1/2 x^T Q x + c^T x + lambda^T z
    s.t.       -M z_i <= x_i <= M z_i,  z_i in {0,1}.

The CORe formulation follows the same g-based three-way disjunction used in the
existing path experiments, but c is loaded directly from the adversarial instance
instead of being derived as -Q.T @ d.
"""

import argparse
import json
import os
import time
from typing import Dict, Tuple

import gurobipy as gp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from gurobipy import GRB

from src.Parametric import Para_Algo

try:
    from experiments.generate_Q_path_adversarial import (
        DEFAULT_OUT_DIR,
        generate_instance_file,
        instance_path,
        instance_seed,
        is_path_graph_Q,
    )
except ImportError:
    from generate_Q_path_adversarial import (
        DEFAULT_OUT_DIR,
        generate_instance_file,
        instance_path,
        instance_seed,
        is_path_graph_Q,
    )


current_dir = os.path.dirname(os.path.abspath(__file__))

np.set_printoptions(linewidth=200)

# Edit this list to choose which adversarial variants the runner executes.
VARIANT_LIST = [
    "exact_adversarial_path",
    # "perturbed_c_adversarial_path",
]

root_bound = [np.inf, -np.inf]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--n_list", type=str, default="100",
                   help='e.g. "100" or "100,200,400" or "[100,200,400]"')
    p.add_argument("--rep_list", type=str, default="0",
                   help='e.g. "0" or "0,1,2" or "[0,1,2]"')

    p.add_argument("--rho", type=float, default=0.45)
    p.add_argument("--tau_list", type=str, default="0.6,0.7,0.8",
                   help='e.g. "0.7" or "0.7,0.8" or "[0.7,0.8]"')
    p.add_argument("--perturbation", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2026)

    p.add_argument("--timelimit", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--big_m_init", type=float, default=1000.0)
    p.add_argument("--job_name", type=str, default="adversarial_path",
                   help="Name of this job (used as output csv filename).")

    return p.parse_args()


def parse_list(s: str, cast=float):
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


def record_root_lb(model, where):
    if where == GRB.Callback.MIPNODE:
        nodecnt = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
        if nodecnt == 0:
            lb = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
            ub = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
            if lb >= root_bound[1]:
                root_bound[1] = lb
            if ub <= root_bound[0]:
                root_bound[0] = ub


def safe_gap(ub, lb):
    if ub is None or lb is None:
        return np.nan
    if not np.isfinite(ub) or not np.isfinite(lb):
        return np.nan
    denom = max(1.0, abs(float(ub)))
    return abs(float(ub) - float(lb)) / denom


def load_instance_Q_path_adversarial(
    n: int,
    rep: int,
    variant: str,
    rho: float,
    tau: float,
    perturbation: float,
    seed: int,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict, str]:
    """
    Load one adversarial path instance, generating it if it is missing.
    """
    path = instance_path(
        out_dir=DEFAULT_OUT_DIR,
        variant=variant,
        n=n,
        rep=rep,
        rho=rho,
        tau=tau,
        perturbation=perturbation,
    )
    if not os.path.exists(path):
        inst_seed = instance_seed(seed, variant, n, rep)
        path, _ = generate_instance_file(
            out_dir=DEFAULT_OUT_DIR,
            n=n,
            rep=rep,
            seed=inst_seed,
            variant=variant,
            rho=rho,
            tau=tau,
            perturbation=perturbation,
        )

    obj = np.load(path, allow_pickle=True)
    Q = sp.csr_matrix(
        (obj["Q_data"], obj["Q_indices"], obj["Q_indptr"]),
        shape=tuple(obj["Q_shape"])
    )
    c = obj["c"].astype(float)
    lam = obj["lambda"].astype(float)
    meta = json.loads(str(obj["meta_json"]))
    return Q, c, lam, meta, path


def validate_instance(Q: sp.csr_matrix, c: np.ndarray, lam: np.ndarray, meta: Dict) -> None:
    n = int(meta["n"])
    rho = float(meta["rho"])
    tau = float(meta["tau"])
    if Q.shape != (n, n):
        raise ValueError(f"Q has shape {Q.shape}, expected {(n, n)}.")
    if c.size != n:
        raise ValueError(f"c has length {c.size}, expected {n}.")
    if lam.size != n:
        raise ValueError(f"lambda has length {lam.size}, expected {n}.")
    if not is_path_graph_Q(Q, rho=rho):
        raise ValueError("Q does not have the requested path-graph tridiagonal structure.")
    if float(meta["min_eig_est"]) <= 0:
        raise ValueError("Q is not positive definite according to metadata.")
    if not np.allclose(lam, tau**2 / 2.0):
        raise ValueError("lambda_i is not tau^2 / 2.")


def tighten_big_m(Q, c, lam, big_m_init, threads, verbose_solver):
    n = c.size
    model_relax = gp.Model()
    z_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z")
    x_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")

    model_relax.addConstrs(x_relax[i] <= big_m_init * z_relax[i] for i in range(n))
    model_relax.addConstrs(x_relax[i] >= -big_m_init * z_relax[i] for i in range(n))
    model_relax.setObjective(x_relax.T @ Q @ x_relax / 2 + c.T @ x_relax + lam.T @ z_relax, GRB.MINIMIZE)
    model_relax.params.OutputFlag = 0
    model_relax.params.Threads = threads
    model_relax.optimize()

    if model_relax.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL} or model_relax.SolCount == 0:
        print(f"Could not tighten Big-M from relaxation; using initial Big-M {big_m_init}.")
        return float(big_m_init)

    x_relax_vals = np.array([x_relax[i].X for i in range(n)])
    max_abs_x = float(np.max(np.abs(x_relax_vals)))
    tightened = min(float(big_m_init), max(float(big_m_init) * 1e-6, 2.0 * max_abs_x))
    if not verbose_solver:
        print(f"The relaxed obj is {model_relax.ObjVal}.")
        print(f"The largest absolute value of x is {max_abs_x}.")
        print(f"Use new Big-M {tightened}.")
    return tightened


def cor_reform(Q, c, lam, M=100, timelimit=None, threads=None, verbose=False):
    """
    Build the g-based three-way convex-hull CORe formulation.
    """
    Q = Q.toarray()
    c = np.asarray(c, dtype=float).reshape(-1)
    lam = np.asarray(lam, dtype=float).reshape(-1)
    n = c.size
    xbar = M * np.ones(n)

    N = [[] for _ in range(n)]
    for i in range(n):
        nz = np.nonzero(Q[i, :])[0]
        for j in nz:
            if j != i and Q[i, j] != 0.0:
                N[i].append(j)

    tau = np.sqrt(2.0 * lam * np.diag(Q))

    m = gp.Model("g_hull_miqp_adversarial_path")
    m.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)
    if threads:
        m.Params.Threads = threads

    x = m.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    g = m.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g")
    m._x = x
    m._g = g

    xp = m.addVars(n, lb=-GRB.INFINITY, ub=0, name="xp")
    xm = m.addVars(n, lb=0, ub=GRB.INFINITY, name="xm")

    g0 = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g0")
    gpv = m.addVars(n, lb=0, ub=GRB.INFINITY, name="gplus")
    gmv = m.addVars(n, lb=0, ub=GRB.INFINITY, name="gminus")

    z0 = m.addMVar(n, vtype=GRB.BINARY, name="z0")
    m._z0 = z0
    zp = m.addMVar(n, vtype=GRB.BINARY, name="zplus")
    zm = m.addMVar(n, vtype=GRB.BINARY, name="zminus")
    m._zp = zp
    m._zm = zm

    for i in range(n):
        expr = c[i]
        if N[i]:
            expr += gp.quicksum(Q[i, j] * x[j] for j in N[i])
        m.addConstr(g[i] == expr, name=f"g_def[{i}]")

    for i in range(n):
        m.addConstr(x[i] == xp[i] + xm[i], name=f"disagg_x[{i}]")
        m.addConstr(g[i] == g0[i] + gpv[i] - gmv[i], name=f"disagg_g[{i}]")
        m.addConstr(z0[i] + zp[i] + zm[i] == 1, name=f"regime_sum[{i}]")

        m.addConstr(g0[i] <= tau[i] * z0[i], name=f"off_g0_ub[{i}]")
        m.addConstr(g0[i] >= -tau[i] * z0[i], name=f"off_g0_lb[{i}]")

        Qii = Q[i, i]
        m.addConstr(Qii * xp[i] + gpv[i] == 0, name=f"onp_stat[{i}]")
        m.addConstr(gpv[i] >= tau[i] * zp[i], name=f"onp_g_lb[{i}]")

        m.addConstr(Qii * xm[i] - gmv[i] == 0, name=f"onm_stat[{i}]")
        m.addConstr(-gmv[i] <= -tau[i] * zm[i], name=f"onm_g_ub[{i}]")

        m.addConstr(xp[i] >= -xbar[i] * zp[i], name=f"xp_lb[{i}]")
        m.addConstr(xm[i] <= xbar[i] * zm[i], name=f"xm_ub[{i}]")

    obj = 0.5 * x @ Q @ x + c @ x + lam @ (zp + zm)
    m.setObjective(obj, GRB.MINIMIZE)
    return m


def original_form(Q, c, lam, M, timelimit, threads, verbose):
    n = c.size
    m = gp.Model("original_miqp_adversarial_path")
    z = m.addMVar(n, vtype=GRB.BINARY, name="z")
    x = m.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    m._z = z
    m._x = x
    m.setObjective(x.T @ Q @ x / 2 + c.T @ x + lam.T @ z, GRB.MINIMIZE)
    m.addConstrs(x[i] <= M * z[i] for i in range(n))
    m.addConstrs(x[i] >= -M * z[i] for i in range(n))
    m.params.OutputFlag = 1 if verbose else 0
    m.params.Threads = threads
    m.params.TimeLimit = timelimit
    return m


def model_result_row(model, n, tau, rep, formulation, variant, rho, perturbation, seed, lambda_value, instance_path_value):
    if model.SolCount > 0:
        end_ub = float(model.ObjVal)
    else:
        end_ub = np.nan
    end_lb = float(model.ObjBound) if hasattr(model, "ObjBound") else np.nan
    end_gap = safe_gap(end_ub, end_lb)

    root_ub, root_lb = root_bound[0], root_bound[1]
    root_gap = safe_gap(root_ub, root_lb)
    if not np.isfinite(root_gap) and model.NodeCount <= 1 and model.SolCount > 0:
        root_ub, root_lb, root_gap = end_ub, end_lb, safe_gap(end_ub, end_lb)

    if formulation == "original" and model.SolCount > 0:
        z_vals = np.array([model._z[i].X for i in range(n)])
    elif formulation == "opt" and model.SolCount > 0:
        z_vals = np.array([1.0 - model._z0[i].X for i in range(n)])
    else:
        z_vals = np.array([])
    nnz = int(np.count_nonzero(np.abs(z_vals) > 1e-7)) if z_vals.size else 0

    return [
        n, tau, rep, formulation,
        root_ub, root_lb, root_gap,
        end_ub, end_lb, end_gap,
        nnz, 100.0 * nnz / n,
        model.NodeCount, model.Runtime,
        variant, rho, perturbation, seed, lambda_value,
        int(model.Status), instance_path_value,
    ]


def run_tree_benchmark(Q, c, lam, M, n, tau, rep, variant, rho, perturbation, seed, lambda_value, instance_path_value):
    # Para_Algo is used elsewhere with Laplacian-style negative edge couplings.
    # The adversarial construction has positive path couplings, so use the
    # bipartite sign change y_i = (-1)^i x_i before calling the tree routine.
    signs = np.ones(n)
    signs[1::2] = -1.0
    Q_signed = signs[:, None] * Q.toarray() * signs[None, :]
    c_signed = signs * c

    start = time.time()
    tree_obj, x_tree_signed = Para_Algo(Q_signed, c_signed, lam, M)
    tree_time = time.time() - start
    x_tree = signs * x_tree_signed
    z_tree_vals = np.array([1 if abs(x_tree[i]) > 1e-7 else 0 for i in range(n)])
    nnz = int(np.count_nonzero(z_tree_vals))
    return [
        n, tau, rep, "tree",
        np.nan, np.nan, np.nan,
        float(tree_obj), np.nan, np.nan,
        nnz, 100.0 * nnz / n,
        0, tree_time,
        variant, rho, perturbation, seed, lambda_value,
        np.nan, instance_path_value,
    ]


def main():
    args = parse_args()

    n_list = parse_list(args.n_list, int)
    rep_list = parse_list(args.rep_list, int)
    tau_list = parse_list(args.tau_list, float)
    variant_list = VARIANT_LIST
    formulations = ["original", "opt", "tree"]

    timelimit = args.timelimit
    threads = args.threads
    big_m_init = float(args.big_m_init)
    verbose_solver = True

    results = []
    columns = [
        "n", "tau", "rep", "formulation",
        "root_ub", "root_lb", "root_gap",
        "end_ub", "end_lb", "end_gap",
        "nnz", "active_pct", "node_count", "time",
        "variant", "rho", "perturbation_level", "seed", "lambda_value",
        "status", "instance_path",
    ]

    for variant in variant_list:
        for n in n_list:
            for rep in rep_list:
                for tau in tau_list:
                    try:
                        Q, c, lam, meta, inst_path = load_instance_Q_path_adversarial(
                            n=n,
                            rep=rep,
                            variant=variant,
                            rho=args.rho,
                            tau=tau,
                            perturbation=args.perturbation,
                            seed=args.seed,
                        )
                        validate_instance(Q, c, lam, meta)
                        inst_seed = int(meta["seed"])
                        lambda_value = float(meta["lambda_value"])

                        print([n, tau, rep, variant])
                        print(
                            f"Q path check passed; min eig ~= {meta['min_eig_est']}, "
                            f"lambda_i = {lambda_value}."
                        )

                        BIG_M = tighten_big_m(
                            Q=Q,
                            c=c,
                            lam=lam,
                            big_m_init=big_m_init,
                            threads=threads,
                            verbose_solver=verbose_solver,
                        )

                        for formulation in formulations:
                            global root_bound
                            root_bound = [np.inf, -np.inf]

                            if formulation == "original":
                                model = original_form(
                                    Q=Q,
                                    c=c,
                                    lam=lam,
                                    M=BIG_M,
                                    timelimit=timelimit,
                                    threads=threads,
                                    verbose=verbose_solver,
                                )
                                model.optimize(record_root_lb)
                                row = model_result_row(
                                    model, n, tau, rep, "original", variant, args.rho,
                                    float(meta["perturbation_level"]), inst_seed, lambda_value, inst_path
                                )
                                results.append(row)
                                print("--------------------------------")
                                print("solve the problem in original formulation")
                                print(f"The obj is {row[7]}. Runtime: {row[13]}.")
                                print(f"Active percentage: {row[11]}%. Nodes: {row[12]}.")
                                print("--------------------------------")

                            elif formulation == "opt":
                                model = cor_reform(
                                    Q=Q,
                                    c=c,
                                    lam=lam,
                                    M=BIG_M,
                                    timelimit=timelimit,
                                    threads=threads,
                                    verbose=verbose_solver,
                                )
                                model.optimize(record_root_lb)
                                row = model_result_row(
                                    model, n, tau, rep, "opt", variant, args.rho,
                                    float(meta["perturbation_level"]), inst_seed, lambda_value, inst_path
                                )
                                results.append(row)
                                print("--------------------------------------------------")
                                print("Solve the optimal solution in the proposed formulation")
                                print(f"The obj is {row[7]}. Runtime: {row[13]}.")
                                print(f"Active percentage: {row[11]}%. Nodes: {row[12]}.")
                                print("--------------------------------------------------")

                            elif formulation == "tree":
                                row = run_tree_benchmark(
                                    Q=Q,
                                    c=c,
                                    lam=lam,
                                    M=BIG_M,
                                    n=n,
                                    tau=tau,
                                    rep=rep,
                                    variant=variant,
                                    rho=args.rho,
                                    perturbation=float(meta["perturbation_level"]),
                                    seed=inst_seed,
                                    lambda_value=lambda_value,
                                    instance_path_value=inst_path,
                                )
                                results.append(row)
                                print("--------------------------------")
                                print("solve the problem in tree algorithm")
                                print(f"The obj is {row[7]}. Runtime: {row[13]}.")
                                print(f"Active percentage: {row[11]}%.")
                                print("--------------------------------")

                            results_df = pd.DataFrame(results, columns=columns)
                            print(results_df)
                            out_csv = os.path.join(
                                current_dir,
                                "..",
                                "experiments_results",
                                f"path_Q_adversarial_{args.job_name}.csv",
                            )
                            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
                            results_df.to_csv(out_csv, index=False)
                    except Exception as e:
                        print(f"Error: {e}")
                        continue


if __name__ == "__main__":
    main()
