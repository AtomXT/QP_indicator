#!/usr/bin/env python3
"""
Run the accelerometer GHMM benchmark from the tree-parametric algorithm paper.

The model uses variables ordered as

    v = [x_1, ..., x_T, r_1, ..., r_n],

where x_t is the hidden activity level for one 10-reading raw block. The
parametric tree run keeps the explicit robust variables r_i = -w_i so all
nonzero off-diagonal entries in Q are negative. The robust opt run projects
out w and uses the capped quadratic observation loss with an epigraph.
"""

import argparse
import os
import sys
import time

import gurobipy as gp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from gurobipy import GRB


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.Parametric import Para_Algo

np.set_printoptions(linewidth=200)

root_bound = [np.inf, -np.inf]
ACCELEROMETER_BLOCK_SIZE = 10


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--max_readings", type=int, default=None,
                   help="Optional smoke-test cap on raw readings before block truncation.")

    p.add_argument("--gamma", type=float, default=400.0)
    p.add_argument("--lambda_outlier", type=float, default=100.0)
    p.add_argument("--sigma2", type=float, default=2.0)
    p.add_argument("--nu", type=float, default=1.0)

    p.add_argument("--inference_modes", type=str, default="robust,nonrobust",
                   help='Comma-separated list from {"robust", "nonrobust"}.')
    p.add_argument("--formulations", type=str, default="opt,tree",
                   help='Comma-separated list from {"opt", "tree"}.')
    p.add_argument("--timelimit", type=float, default=300.0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--big_m_init", type=float, default=1000.0)
    p.add_argument("--job_name", type=str, default="accelerometer_default",
                   help="Name of this job (used as output csv filename).")

    return p.parse_args()


def parse_inference_modes(s):
    modes = [x.strip() for x in s.split(",") if x.strip()]
    allowed = {"robust", "nonrobust"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unknown inference modes: {unknown}. Allowed values are {sorted(allowed)}.")
    return modes


def parse_formulations(s):
    formulations = [x.strip() for x in s.split(",") if x.strip()]
    allowed = {"opt", "tree"}
    unknown = sorted(set(formulations) - allowed)
    if unknown:
        raise ValueError(f"Unknown formulations: {unknown}. Allowed values are {sorted(allowed)}.")
    return formulations


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
    return abs(float(ub) - float(lb)) / max(1.0, abs(float(ub)))


def load_signal(max_readings):
    data_root = os.path.join(project_root, "data")
    path = os.path.join(data_root, "accelerometer.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, header=None, skiprows=1)
    y = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)
    if max_readings is not None:
        y = y[:max_readings]

    if len(y) == 0:
        raise ValueError("No raw accelerometer readings found.")

    meta = {
        "raw_readings": len(y),
        "block_size": ACCELEROMETER_BLOCK_SIZE,
    }
    return y, meta


def build_ghmm_instance(y, gamma, lambda_outlier, sigma2, nu):
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    if nu <= 0:
        raise ValueError("nu must be positive.")

    block_size = ACCELEROMETER_BLOCK_SIZE
    raw_count = len(y)
    n_obs = (raw_count // block_size) * block_size
    if n_obs == 0:
        raise ValueError("Not enough raw readings to form one complete accelerometer block.")
    y = np.asarray(y[:n_obs], dtype=float)
    T = n_obs // block_size

    n = T + n_obs
    c = np.zeros(n)
    lam = np.empty(n)
    lam[:T] = gamma
    lam[T:] = lambda_outlier

    rows = []
    cols = []
    data = []
    const = 0.0

    def add_diag(i, value):
        rows.append(i)
        cols.append(i)
        data.append(value)

    def add_edge(i, j, value):
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([value, value])

    obs_weight = 1.0 / (nu * nu)
    state_weight = 1.0 / sigma2

    for t in range(T):
        x_idx = t
        for k in range(block_size):
            obs_idx = t * block_size + k
            r_idx = T + obs_idx
            y_val = y[obs_idx]

            add_diag(x_idx, 2.0 * obs_weight)
            add_diag(r_idx, 2.0 * obs_weight)
            add_edge(x_idx, r_idx, -2.0 * obs_weight)

            c[x_idx] += -2.0 * obs_weight * y_val
            c[r_idx] += 2.0 * obs_weight * y_val
            const += obs_weight * y_val * y_val

    for t in range(T):
        add_diag(t, 4.0 * state_weight)
    for t in range(1, T):
        add_edge(t - 1, t, -2.0 * state_weight)

    Q = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    meta = {
        "T": T,
        "n_obs": n_obs,
        "n_model": n,
        "truncated_readings": n_obs,
        "block_size": block_size,
    }
    return Q, c, lam, const, y, meta


def build_projected_robust_instance(y, gamma, sigma2, nu):
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    if nu <= 0:
        raise ValueError("nu must be positive.")

    block_size = ACCELEROMETER_BLOCK_SIZE
    raw_count = len(y)
    n_obs = (raw_count // block_size) * block_size
    if n_obs == 0:
        raise ValueError("Not enough raw readings to form one complete accelerometer block.")
    y = np.asarray(y[:n_obs], dtype=float)
    T = n_obs // block_size

    c = np.zeros(T)
    lam = gamma * np.ones(T)
    rows = []
    cols = []
    data = []

    def add_diag(i, value):
        rows.append(i)
        cols.append(i)
        data.append(value)

    def add_edge(i, j, value):
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([value, value])

    state_weight = 1.0 / sigma2
    for t in range(T):
        add_diag(t, 4.0 * state_weight)
    for t in range(1, T):
        add_edge(t - 1, t, -2.0 * state_weight)

    Q = sp.csr_matrix((data, (rows, cols)), shape=(T, T))
    meta = {
        "T": T,
        "n_obs": n_obs,
        "n_model": T,
        "truncated_readings": n_obs,
        "block_size": block_size,
    }
    return Q, c, lam, y, meta


def build_nonrobust_ghmm_instance(y, gamma, sigma2, nu):
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    if nu <= 0:
        raise ValueError("nu must be positive.")

    block_size = ACCELEROMETER_BLOCK_SIZE
    raw_count = len(y)
    n_obs = (raw_count // block_size) * block_size
    if n_obs == 0:
        raise ValueError("Not enough raw readings to form one complete accelerometer block.")
    y = np.asarray(y[:n_obs], dtype=float)
    T = n_obs // block_size

    c = np.zeros(T)
    lam = gamma * np.ones(T)
    rows = []
    cols = []
    data = []
    const = 0.0

    def add_diag(i, value):
        rows.append(i)
        cols.append(i)
        data.append(value)

    def add_edge(i, j, value):
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([value, value])

    obs_weight = 1.0 / (nu * nu)
    state_weight = 1.0 / sigma2

    for t in range(T):
        for k in range(block_size):
            y_val = y[t * block_size + k]
            add_diag(t, 2.0 * obs_weight)
            c[t] += -2.0 * obs_weight * y_val
            const += obs_weight * y_val * y_val

    if T == 1:
        add_diag(0, 2.0 * state_weight)
    else:
        for t in range(1, T):
            add_diag(t - 1, 2.0 * state_weight)
            add_diag(t, 2.0 * state_weight)
            add_edge(t - 1, t, -2.0 * state_weight)

    Q = sp.csr_matrix((data, (rows, cols)), shape=(T, T))
    meta = {
        "T": T,
        "n_obs": n_obs,
        "n_model": T,
        "truncated_readings": n_obs,
        "block_size": block_size,
    }
    return Q, c, lam, const, y, meta


def cor_reform(Q, c, const, lam, M=1000.0, timelimit=None, threads=None, verbose=True):
    """
    Build the sparse g-based three-way convex-hull CORe formulation.
    """
    Q = Q.tocsr()
    c = np.asarray(c, dtype=float).reshape(-1)
    lam = np.asarray(lam, dtype=float).reshape(-1)
    n = c.size
    xbar = M * np.ones(n)
    q_diag = Q.diagonal()
    tau = np.sqrt(2.0 * lam * q_diag)

    m = gp.Model("g_hull_miqp_accelerometer")
    m.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)
    if threads:
        m.Params.Threads = threads

    x = m.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    g = m.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g")
    m._x = x
    m._g = g

    xp = m.addMVar(n, lb=-GRB.INFINITY, ub=0, name="xp")
    xm = m.addMVar(n, lb=0, ub=GRB.INFINITY, name="xm")

    g0 = m.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g0")
    gpv = m.addMVar(n, lb=0, ub=GRB.INFINITY, name="gplus")
    gmv = m.addMVar(n, lb=0, ub=GRB.INFINITY, name="gminus")

    z0 = m.addMVar(n, vtype=GRB.BINARY, name="z0")
    zp = m.addMVar(n, vtype=GRB.BINARY, name="zplus")
    zm = m.addMVar(n, vtype=GRB.BINARY, name="zminus")
    m._z0 = z0
    m._zp = zp
    m._zm = zm

    for i in range(n):
        expr = c[i]
        row_start, row_end = Q.indptr[i], Q.indptr[i + 1]
        for idx in range(row_start, row_end):
            j = Q.indices[idx]
            qij = Q.data[idx]
            if j != i and qij != 0.0:
                expr += qij * x[j]
        m.addConstr(g[i] == expr, name=f"g_def[{i}]")

    for i in range(n):
        m.addConstr(x[i] == xp[i] + xm[i], name=f"disagg_x[{i}]")
        m.addConstr(g[i] == g0[i] + gpv[i] - gmv[i], name=f"disagg_g[{i}]")
        m.addConstr(z0[i] + zp[i] + zm[i] == 1, name=f"regime_sum[{i}]")

        m.addConstr(g0[i] <= tau[i] * z0[i], name=f"off_g0_ub[{i}]")
        m.addConstr(g0[i] >= -tau[i] * z0[i], name=f"off_g0_lb[{i}]")

        m.addConstr(q_diag[i] * xp[i] + gpv[i] == 0, name=f"onp_stat[{i}]")
        m.addConstr(gpv[i] >= tau[i] * zp[i], name=f"onp_g_lb[{i}]")

        m.addConstr(q_diag[i] * xm[i] - gmv[i] == 0, name=f"onm_stat[{i}]")
        m.addConstr(-gmv[i] <= -tau[i] * zm[i], name=f"onm_g_ub[{i}]")

        m.addConstr(xp[i] >= -xbar[i] * zp[i], name=f"xp_lb[{i}]")
        m.addConstr(xm[i] <= xbar[i] * zm[i], name=f"xm_ub[{i}]")

    m.setObjective(x.T @ Q @ x / 2 + c.T @ x + const + lam.T @ (zp + zm), GRB.MINIMIZE)
    return m


def cor_reform_projected_robust(
    Q,
    c,
    lam,
    y,
    lambda_outlier,
    nu,
    block_size,
    M=1000.0,
    timelimit=None,
    threads=None,
    verbose=True,
):
    """
    Projected robust formulation with epigraph observation losses.

    Projecting w gives min{obs_weight * (y_i - x_t)^2, lambda_outlier}.
    The split residual a0 + a1 mirrors the robust binary classification runner:
    a0 carries the non-outlier residual and a1 is only available when z_i = 1.
    """
    Q = Q.tocsr()
    c = np.asarray(c, dtype=float).reshape(-1)
    lam = np.asarray(lam, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    T = c.size
    n_obs = y.size
    if lambda_outlier < 0:
        raise ValueError("lambda_outlier must be nonnegative.")
    obs_weight = 1.0 / (nu * nu)
    residual_radius = float(np.sqrt(lambda_outlier / obs_weight))

    m = gp.Model("projected_robust_epigraph_accelerometer")
    m.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)
    if threads:
        m.Params.Threads = threads

    x = m.addMVar(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    s = m.addMVar(T, vtype=GRB.BINARY, name="s")
    m._x = x
    m._z_hidden = s
    for i in range(T):
        m.addConstr(x[i] <= M * s[i], name=f"x_ub[{i}]")
        m.addConstr(x[i] >= -M * s[i], name=f"x_lb[{i}]")

    u = m.addMVar(n_obs, lb=0.0, name="obs_loss")
    a0 = m.addMVar(n_obs, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="resid_clean")
    a1 = m.addMVar(n_obs, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="resid_outlier")
    z_out = m.addMVar(n_obs, vtype=GRB.BINARY, name="z_outlier")
    m._u = u
    m._z_out = z_out
    m._a0 = a0
    m._a1 = a1

    for obs_idx, y_val in enumerate(y):
        t = obs_idx // block_size
        residual_bound = float(abs(y_val) + M)
        m.addConstr(float(y_val) - x[t] == a0[obs_idx] + a1[obs_idx], name=f"resid_split[{obs_idx}]")
        m.addConstr(a0[obs_idx] <= residual_radius * (1 - z_out[obs_idx]), name=f"clean_resid_ub[{obs_idx}]")
        m.addConstr(a0[obs_idx] >= -residual_radius * (1 - z_out[obs_idx]), name=f"clean_resid_lb[{obs_idx}]")
        m.addConstr(a1[obs_idx] <= residual_bound * z_out[obs_idx], name=f"outlier_resid_ub[{obs_idx}]")
        m.addConstr(a1[obs_idx] >= -residual_bound * z_out[obs_idx], name=f"outlier_resid_lb[{obs_idx}]")
        m.addConstr(obs_weight * a0[obs_idx] * a0[obs_idx] <= u[obs_idx], name=f"obs_epigraph[{obs_idx}]")

    m.setObjective(
        x.T @ Q @ x / 2
        + c.T @ x
        + lam.T @ s
        + gp.quicksum(u[i] for i in range(n_obs))
        + lambda_outlier * gp.quicksum(z_out[i] for i in range(n_obs)),
        GRB.MINIMIZE,
    )
    return m


def tighten_big_m(Q, c, const, lam, big_m_init, threads):
    n = c.size
    model_relax = gp.Model()
    z_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z")
    x_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")

    model_relax.addConstrs(x_relax[i] <= big_m_init * z_relax[i] for i in range(n))
    model_relax.addConstrs(x_relax[i] >= -big_m_init * z_relax[i] for i in range(n))
    model_relax.setObjective(x_relax.T @ Q @ x_relax / 2 + c.T @ x_relax + const + lam.T @ z_relax, GRB.MINIMIZE)
    model_relax.params.OutputFlag = 0
    model_relax.params.Threads = threads
    model_relax.optimize()

    if model_relax.SolCount == 0:
        print(f"Could not tighten Big-M; using initial Big-M {big_m_init}.")
        return float(big_m_init)

    x_relax_vals = np.asarray(x_relax.X, dtype=float)
    max_abs_x = float(np.max(np.abs(x_relax_vals)))

    if max_abs_x <= 1e-8:
        tightened = float(big_m_init)
    else:
        tightened = min(float(big_m_init), 2.0 * max_abs_x)

    print(f"The relaxed obj is {model_relax.ObjVal}.")
    print(f"The largest absolute value of x is {max_abs_x}.")
    print(f"Use new Big-M {tightened}.")
    return tightened


def append_solution_rows(
    hidden_rows,
    outlier_rows,
    inference_mode,
    formulation,
    v,
    z,
    y,
    T,
    has_outliers,
    tol,
    block_size,
):
    x_vals = v[:T]
    z_vals = np.asarray(z)

    for t in range(T):
        start_reading = t * block_size
        end_reading = (t + 1) * block_size
        hidden_rows.append([
            inference_mode,
            formulation,
            t + 1,
            start_reading,
            end_reading,
            x_vals[t],
            int(z_vals[t] > 0.5 or abs(x_vals[t]) > tol),
        ])

    if not has_outliers:
        for obs_idx, y_val in enumerate(y):
            t = obs_idx // block_size
            k = obs_idx % block_size
            outlier_rows.append([
                inference_mode,
                formulation,
                obs_idx + 1,
                obs_idx,
                obs_idx,
                t + 1,
                k + 1,
                y_val,
                0,
                0.0,
                y_val,
            ])
        return

    r_vals = v[T:]
    for obs_idx, y_val in enumerate(y):
        t = obs_idx // block_size
        k = obs_idx % block_size
        r_val = r_vals[obs_idx]
        w_val = -r_val
        outlier_rows.append([
            inference_mode,
            formulation,
            obs_idx + 1,
            obs_idx,
            obs_idx,
            t + 1,
            k + 1,
            y_val,
            int(z_vals[T + obs_idx] > 0.5 or abs(r_val) > tol),
            w_val,
            y_val - w_val,
        ])


def append_projected_solution_rows(
    hidden_rows,
    outlier_rows,
    inference_mode,
    formulation,
    x_vals,
    hidden_z,
    outlier_z,
    y,
    T,
    tol,
    block_size,
):
    x_vals = np.asarray(x_vals, dtype=float)
    hidden_z = np.asarray(hidden_z, dtype=float)
    outlier_z = np.asarray(outlier_z, dtype=float)

    for t in range(T):
        start_reading = t * block_size
        end_reading = (t + 1) * block_size
        hidden_rows.append([
            inference_mode,
            formulation,
            t + 1,
            start_reading,
            end_reading,
            x_vals[t],
            int(hidden_z[t] > 0.5 or abs(x_vals[t]) > tol),
        ])

    for obs_idx, y_val in enumerate(y):
        t = obs_idx // block_size
        k = obs_idx % block_size
        flag = int(outlier_z[obs_idx] > 0.5)
        x_t = x_vals[t]
        w_val = y_val - x_t if flag else 0.0
        outlier_rows.append([
            inference_mode,
            formulation,
            obs_idx + 1,
            obs_idx,
            obs_idx,
            t + 1,
            k + 1,
            y_val,
            flag,
            w_val,
            y_val - w_val,
        ])


def main():
    args = parse_args()
    inference_modes = parse_inference_modes(args.inference_modes)
    formulations = parse_formulations(args.formulations)
    solution_tol = 1e-7

    raw_y, data_meta = load_signal(args.max_readings)
    results = []
    hidden_rows = []
    outlier_rows = []

    columns = [
        "dataset", "inference_mode", "formulation", "T", "n_model", "gamma", "lambda_outlier",
        "sigma2", "nu", "big_m", "root_ub", "root_lb", "root_gap", "end_ub", "end_lb",
        "end_gap", "hidden_active_count", "outlier_active_count", "node_count",
        "time", "status",
    ]

    for inference_mode in inference_modes:
        if inference_mode == "robust":
            Q_tree, c_tree, lam_tree, const_tree, y, meta_tree = build_ghmm_instance(
                raw_y,
                gamma=args.gamma,
                lambda_outlier=args.lambda_outlier,
                sigma2=args.sigma2,
                nu=args.nu,
            )
            Q_opt, c_opt, lam_opt, y_opt, meta_opt = build_projected_robust_instance(
                raw_y,
                gamma=args.gamma,
                sigma2=args.sigma2,
                nu=args.nu,
            )
            has_outliers = True
            meta_tree.update(data_meta)
            meta_opt.update(data_meta)
        else:
            Q_tree, c_tree, lam_tree, const_tree, y, meta_tree = build_nonrobust_ghmm_instance(
                raw_y,
                gamma=args.gamma,
                sigma2=args.sigma2,
                nu=args.nu,
            )
            Q_opt, c_opt, lam_opt, y_opt, meta_opt = Q_tree, c_tree, lam_tree, y, meta_tree
            has_outliers = False
            meta_tree.update(data_meta)
            meta_opt = meta_tree

        print([
            "accelerometer",
            inference_mode,
            meta_tree["T"],
            meta_tree["n_model"],
            meta_tree["block_size"],
            args.gamma,
            args.lambda_outlier,
            args.sigma2,
            args.nu,
        ])
        print(
            f"Read {meta_tree['raw_readings']} raw values; using {meta_tree['truncated_readings']} "
            f"raw observations in {meta_tree['T']} blocks of {meta_tree['block_size']}."
        )
        if inference_mode == "robust":
            print(
                f"Projected robust opt has {meta_opt['n_model']} continuous hidden variables; "
                f"tree uses explicit dimension {meta_tree['n_model']}."
            )
            BIG_M_TREE = tighten_big_m(
                Q_tree,
                c_tree,
                const_tree,
                lam_tree,
                float(args.big_m_init),
                args.threads,
            )
            BIG_M_OPT = BIG_M_TREE
        else:
            BIG_M_TREE = tighten_big_m(Q_tree, c_tree, const_tree, lam_tree, float(args.big_m_init), args.threads)
            BIG_M_OPT = BIG_M_TREE

        for formulation in formulations:
            try:
                global root_bound
                root_bound = [np.inf, -np.inf]

                if formulation == "opt":
                    if inference_mode == "robust":
                        model_opt = cor_reform_projected_robust(
                            Q=Q_opt,
                            c=c_opt,
                            lam=lam_opt,
                            y=y_opt,
                            lambda_outlier=args.lambda_outlier,
                            nu=args.nu,
                            block_size=meta_opt["block_size"],
                            M=BIG_M_OPT,
                            timelimit=args.timelimit,
                            threads=args.threads,
                            verbose=True,
                        )
                    else:
                        model_opt = cor_reform(
                            Q=Q_opt,
                            c=c_opt,
                            const=const_tree,
                            lam=lam_opt,
                            M=BIG_M_OPT,
                            timelimit=args.timelimit,
                            threads=args.threads,
                            verbose=True,
                        )
                    model_opt.optimize(record_root_lb)

                    if model_opt.SolCount > 0:
                        end_ub = float(model_opt.ObjVal)
                        if inference_mode == "robust":
                            v_vals = np.asarray(model_opt._x.X, dtype=float)
                            hidden_z = np.asarray(model_opt._z_hidden.X, dtype=float)
                            outlier_z = np.asarray(model_opt._z_out.X, dtype=float)
                            z_vals = np.concatenate([hidden_z, outlier_z])
                            append_projected_solution_rows(
                                hidden_rows, outlier_rows, inference_mode, "opt",
                                v_vals, hidden_z, outlier_z, y_opt,
                                meta_opt["T"], solution_tol, meta_opt["block_size"],
                            )
                        else:
                            v_vals = np.asarray(model_opt._x.X, dtype=float)
                            z_vals = np.asarray(1.0 - model_opt._z0.X, dtype=float)
                            append_solution_rows(
                                hidden_rows, outlier_rows, inference_mode, "opt", v_vals, z_vals, y,
                                meta_opt["T"], has_outliers, solution_tol,
                                meta_opt["block_size"],
                            )
                    else:
                        z_vals = np.array([])
                        end_ub = np.nan

                    end_lb = float(model_opt.ObjBound) if hasattr(model_opt, "ObjBound") else np.nan
                    root_ub, root_lb = root_bound[0], root_bound[1]
                    root_gap = safe_gap(root_ub, root_lb)
                    if not np.isfinite(root_gap) and model_opt.NodeCount <= 1 and model_opt.SolCount > 0:
                        root_ub, root_lb = end_ub, end_lb
                        root_gap = safe_gap(root_ub, root_lb)

                    hidden_active = int(np.count_nonzero(z_vals[:meta_opt["T"]] > 0.5)) if z_vals.size else 0
                    outlier_active = int(np.count_nonzero(z_vals[meta_opt["T"]:] > 0.5)) if has_outliers and z_vals.size else 0
                    result = [
                        "accelerometer", inference_mode, "opt", meta_opt["T"], meta_opt["n_model"],
                        args.gamma, args.lambda_outlier, args.sigma2, args.nu, BIG_M_OPT,
                        root_ub, root_lb, root_gap, end_ub, end_lb, safe_gap(end_ub, end_lb),
                        hidden_active, outlier_active, model_opt.NodeCount, model_opt.Runtime,
                        int(model_opt.Status),
                    ]
                    results.append(result)
                    print("--------------------------------------------------")
                    print(f"Solve the {inference_mode} problem in the proposed formulation")
                    print(f"The obj is {end_ub}.")
                    print(f"Runtime: {model_opt.Runtime}.")
                    print(f"Hidden active count: {hidden_active}; outlier active count: {outlier_active}")
                    print("--------------------------------------------------")

                elif formulation == "tree":
                    start = time.time()
                    tree_obj, v_tree = Para_Algo(Q_tree.toarray(), c_tree, lam_tree, BIG_M_TREE)
                    tree_obj += const_tree
                    tree_time = time.time() - start
                    z_tree = (np.abs(v_tree) > solution_tol).astype(float)

                    append_solution_rows(
                        hidden_rows, outlier_rows, inference_mode, "tree", v_tree, z_tree, y,
                        meta_tree["T"], has_outliers, solution_tol,
                        meta_tree["block_size"],
                    )

                    hidden_active = int(np.count_nonzero(z_tree[:meta_tree["T"]]))
                    outlier_active = int(np.count_nonzero(z_tree[meta_tree["T"]:])) if has_outliers else 0
                    result = [
                        "accelerometer", inference_mode, "tree", meta_tree["T"], meta_tree["n_model"],
                        args.gamma, args.lambda_outlier, args.sigma2, args.nu, BIG_M_TREE,
                        np.nan, np.nan, np.nan, float(tree_obj), np.nan, np.nan,
                        hidden_active, outlier_active, 0, tree_time, "tree_completed",
                    ]
                    results.append(result)
                    print("--------------------------------")
                    print(f"solve the {inference_mode} problem in tree algorithm")
                    print(f"The obj is {tree_obj}.")
                    print(f"Runtime: {tree_time}.")
                    print(f"Hidden active count: {hidden_active}; outlier active count: {outlier_active}")
                    print("--------------------------------")

                results_df = pd.DataFrame(results, columns=columns)
                print(results_df)
                out_csv = os.path.join(
                    current_dir,
                    "..",
                    "experiments_results",
                    f"accelerometer_GHMM_{args.job_name}.csv",
                )
                os.makedirs(os.path.dirname(out_csv), exist_ok=True)
                results_df.to_csv(out_csv, index=False)
            except Exception as e:
                print(f"Error: {e}")
                continue

    hidden_df = pd.DataFrame(
        hidden_rows,
        columns=["inference_mode", "formulation", "t", "start_reading", "end_reading", "hidden_state", "hidden_active"],
    )
    outlier_df = pd.DataFrame(
        outlier_rows,
        columns=[
            "inference_mode", "formulation", "reading_index", "start_reading", "end_reading",
            "t", "k", "y", "outlier_flag", "outlier_adjustment_w", "cleaned_y",
        ],
    )

    solution_prefix = os.path.join(current_dir, "..", "experiments_results", f"accelerometer_GHMM_{args.job_name}")
    hidden_df.to_csv(f"{solution_prefix}_hidden_signal.csv", index=False)
    outlier_df.to_csv(f"{solution_prefix}_outlier_flags.csv", index=False)


if __name__ == "__main__":
    main()
