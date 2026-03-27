#!/usr/bin/env python3
"""
Run experiments for Rank-One Augmented Indicator QP:

    min_{x,z}  1/2 x^T Q x + (a^T x)^2 + mu^T z
    s.t.       x_i (1 - z_i) = 0,  z_i in {0,1}

Compare:
- original: Big-M formulation (x_i <= M z_i, x_i >= -M z_i)
- core:     CORe 3-way ideal hull formulation using the rank-one-aware field h_i

Dataset loader expected:
- load_rankone_instance(n, delta, rep, data_dir=...) -> (Q (csr), a (ndarray), meta (dict))

Your dataset script (no mu stored) produces .npz containing Q (CSR) and a.
We generate mu in this runner from tau (like your other script).

Notes:
- The CORe derivation uses:
    g_i = sum_{j!=i} Q_ij x_j
    s_i = sum_{j!=i} a_j x_j
    h_i = g_i + 2 a_i s_i
    d_i = Q_ii + 2 a_i^2
    tau_i = sqrt(2 mu_i d_i)
  Here we pick mu_i = tau^2 / (2 d_i), making tau_i == tau for all i.

- We bound |g_i| and |s_i| using |x|<=M, then bound |h_i| <= H_i = G_i + 2|a_i| S_i.

Requires: gurobipy, numpy, scipy, pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import gurobipy as gp
import cvxpy as cp
import random
from gurobipy import GRB
from src.utils import decomposition, get_data, get_data_offline, load_instance_Q_sparsity
import os
import argparse
import json


current_dir = os.path.dirname(os.path.abspath(__file__))

np.set_printoptions(linewidth=200)

def parse_args():
    p = argparse.ArgumentParser()

    # allow scalar or list-like inputs
    p.add_argument("--n_list", type=str, default="500",
                   help='e.g. "500" or "50,60,70" or "[50,60,70]"')
    p.add_argument("--delta_list", type=str, default="0.01",
                   help='e.g. "0.01" or "0.01,0.05" or "[0.01,0.05]"')
    p.add_argument("--rep_list", type=str, default="1",
                   help='e.g. "0" or "0,1,2" or "[0,1,2]"')
    p.add_argument("--tau_list", type=str, default="0.1",
                   help='e.g. "0.05,0.1,0.2"')
    p.add_argument("--timelimit", type=float, default=300.0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--big_m_init", type=float, default=1000.0)

    p.add_argument("--job_name", type=str, default="temp",
                   help="Name of this job (used as output csv filename)")

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
        return [cast(x) for x in s.split(",") if x.strip() != ""]
    return [cast(s)]

from itertools import combinations
from src.rank2 import fast_dp_general

root_bound = [np.inf, -np.inf]
def record_root_lb(model, where):
    if where == GRB.Callback.MIPNODE:
        # check if this is the root node
        nodecnt = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
        if nodecnt == 0:
            # get the relaxation bound at this node
            lb = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
            ub = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
            # store it if not yet recorded
            if lb >= root_bound[1]:
                root_bound[1] = lb
            if ub <= root_bound[0]:
                root_bound[0] = ub


# ============================================================
# CORe reformulation for SHIFTED + RANK-ONE objective:
#
#   min  0.5 x'Qx - c'x + (a'x)^2 + lam'z
#   s.t. x_i(1-z_i)=0  (modeled via |x_i| <= xbar_i, and the CORe disjunction)
#
# Coordinate field:
#   g_i = sum_{j!=i} Q_ij x_j
#   s_i = sum_{j!=i} a_j x_j
#   H_i = g_i - c_i + 2 a_i s_i
#   D_i = Q_ii + 2 a_i^2
#   tau_i = sqrt(2 lam_i D_i)
#
# Disjunction on (x_i, H_i):
#   D0: x_i=0, |H_i| <= tau_i
#   D+: D_i x_i + H_i = 0,  H_i >= tau_i
#   D-: D_i x_i + H_i = 0,  H_i <= -tau_i
#
# Need bound |H_i| <= Hbar_i from |x|<=xbar:
#   |g_i| <= sum_{j!=i} |Q_ij| xbar_j
#   |s_i| <= sum_{j!=i} |a_j|  xbar_j
#   => |H_i| <= G_i + |c_i| + 2|a_i| S_i
# ============================================================
def cor_reform_rankone_shift(Q, d, a, lam, xbar=100.0, timelimit=None, mip_gap=None, threads=None, verbose=False):
    """
    CORe 3-way ideal hull formulation for the shifted + rank-one augmented indicator QP.

    Parameters
    ----------
    Q : sparse matrix or ndarray (n,n), symmetric PSD recommended, Q_ii>0
    d : ndarray (n,)
    a : ndarray (n,)
    lam : ndarray (n,) >= 0 (indicator penalties)
    xbar : float or ndarray (n,), bound |x_i|<=xbar_i (required)
    timelimit, mip_gap, threads, verbose : gurobi params

    Returns
    -------
    model : gurobipy.Model
        with model._z0 storing the "off" regime binaries (so z = 1 - z0)
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ImportError("This function requires gurobipy (Gurobi Python API).") from e

    # Convert Q to dense array for simplicity in quadratic objective assembly
    Q = Q.toarray() if hasattr(Q, "toarray") else np.array(Q)
    n = d.size

    # shift: original objective had 0.5(d-x)'Q(d-x) = 0.5 x'Qx - (Qd)'x + const
    # so c = Qd
    c = Q @ d

    # ensure xbar vector
    if np.isscalar(xbar):
        xbar = float(xbar) * np.ones(n)
    else:
        xbar = np.array(xbar, dtype=float).reshape(-1)


    # constants
    D = np.diag(Q) + 2.0 * (a ** 2)                 # D_i
    tau = np.sqrt(2.0 * lam * D)                    # tau_i

    # bounds for H_i
    Qabs = np.abs(Q).copy()
    np.fill_diagonal(Qabs, 0.0)
    # G_i = sum_{j!=i} |Q_ij| xbar_j
    G = Qabs @ xbar
    # S_i = sum_{j!=i} |a_j| xbar_j
    S_total = np.abs(a)@xbar
    S = np.full(n, S_total) - np.abs(a) * xbar
    Hbar = G + np.abs(c) + 2.0 * np.abs(a) * S

    m = gp.Model("core_rankone_shift_hull")
    m.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)
    if threads is not None:
        m.Params.Threads = int(threads)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    # Aggregate variables
    x = m.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    g = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g")  # off-diagonal Q-field
    s = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="s")  # off-diagonal a-field
    H = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="H")  # effective field

    # Disaggregated variables per regime
    x0 = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x0")
    xp = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xp")
    xm = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xm")

    H0 = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="H0")
    Hp = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Hp")
    Hm = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Hm")

    # Regime binaries
    z0 = m.addVars(n, vtype=GRB.BINARY, name="z0")     # off
    zp = m.addVars(n, vtype=GRB.BINARY, name="zplus")  # on+
    zm = m.addVars(n, vtype=GRB.BINARY, name="zminus") # on-
    m._z0 = z0

    # Bounds on aggregate x (required)
    for i in range(n):
        m.addConstr(x[i] <= xbar[i], name=f"x_ub[{i}]")
        m.addConstr(x[i] >= -xbar[i], name=f"x_lb[{i}]")

    # Definitions of g_i, s_i, H_i
    for i in range(n):
        # g_i = sum_{j!=i} Q_ij x_j  (use sparse neighbor set)
        m.addConstr(g[i] == gp.quicksum(Q[i, j] * x[j] for j in range(n) if j != i), name=f"g_def[{i}]")
        # m.addConstr(g[i] <= G[i], name=f"g_ub[{i}]")
        # m.addConstr(g[i] >= -G[i], name=f"g_lb[{i}]")

        # s_i = sum_{j!=i} a_j x_j
        m.addConstr(s[i] == gp.quicksum(a[j] * x[j] for j in range(n) if j != i), name=f"s_def[{i}]")
        # m.addConstr(s[i] <= S[i], name=f"s_ub[{i}]")
        # m.addConstr(s[i] >= -S[i], name=f"s_lb[{i}]")

        # H_i = g_i - c_i + 2 a_i s_i
        m.addConstr(H[i] == g[i] - c[i] + 2.0 * a[i] * s[i], name=f"H_def[{i}]")
        # m.addConstr(H[i] == g[i] - c[i], name=f"H_def[{i}]")
        # m.addConstr(H[i] <= Hbar[i], name=f"H_ub[{i}]")
        # m.addConstr(H[i] >= -Hbar[i], name=f"H_lb[{i}]")

    # Hull coupling and regime constraints
    for i in range(n):
        m.addConstr(x[i] == x0[i] + xp[i] + xm[i], name=f"disagg_x[{i}]")
        m.addConstr(H[i] == H0[i] + Hp[i] + Hm[i], name=f"disagg_H[{i}]")
        m.addConstr(z0[i] + zp[i] + zm[i] == 1, name=f"regime_sum[{i}]")

        # Off: x0=0, |H0| <= tau z0
        m.addConstr(x0[i] == 0.0, name=f"off_x0[{i}]")
        m.addConstr(H0[i] <= float(tau[i]) * z0[i], name=f"off_H0_ub[{i}]")
        m.addConstr(H0[i] >= -float(tau[i]) * z0[i], name=f"off_H0_lb[{i}]")

        # On+: D_i xp + Hp = 0,  tau z+ <= Hp <= Hbar z+
        m.addConstr(float(D[i]) * xp[i] + Hp[i] == 0.0, name=f"onp_stat[{i}]")
        m.addConstr(Hp[i] >= float(tau[i]) * zp[i], name=f"onp_Hp_lb[{i}]")
        m.addConstr(Hp[i] <= float(Hbar[i]) * zp[i], name=f"onp_Hp_ub[{i}]")

        # On-: D_i xm + Hm = 0,  -Hbar z- <= Hm <= -tau z-
        m.addConstr(float(D[i]) * xm[i] + Hm[i] == 0.0, name=f"onm_stat[{i}]")
        m.addConstr(Hm[i] >= -float(Hbar[i]) * zm[i], name=f"onm_Hm_lb[{i}]")
        m.addConstr(Hm[i] <= -float(tau[i]) * zm[i], name=f"onm_Hm_ub[{i}]")

        # Optional disagg bounds (helps numerics)
        m.addConstr(xp[i] <= xbar[i] * zp[i], name=f"xp_ub[{i}]")
        m.addConstr(xp[i] >= -xbar[i] * zp[i], name=f"xp_lb[{i}]")
        m.addConstr(xm[i] <= xbar[i] * zm[i], name=f"xm_ub[{i}]")
        m.addConstr(xm[i] >= -xbar[i] * zm[i], name=f"xm_lb[{i}]")

    # Objective: 0.5 x'Qx - c'x + (a'x)^2 + sum lam_i (zplus+zminus) + const 0.5 d'Qd
    obj = 0.5 * x.T@Q@x - c.T@x + (a.T@x)*(a.T@x) + gp.quicksum(float(lam[i]) * (zp[i] + zm[i]) for i in range(n)) + 0.5 * float(d.T @ (Q @ d))
    # obj = 0.5 * x.T @ Q @ x - c.T @ x + gp.quicksum(
    #     float(lam[i]) * (zp[i] + zm[i]) for i in range(n)) + 0.5 * float(d.T @ (Q @ d))
    m.setObjective(obj, GRB.MINIMIZE)
    return m


args = parse_args()

n_list = parse_list(args.n_list, int)
delta_list = parse_list(args.delta_list, float)
rep_list = parse_list(args.rep_list, int)
tau_list = parse_list(args.tau_list, float)
timelimit = args.timelimit
THREADS = args.threads
BIG_M_INIT = float(args.big_m_init)

job_name = args.job_name
results = []

for n in n_list:
    for delta in delta_list:
        for rep in rep_list:
            # Load Q, d from your generator (Q presumably sparse)
            Q, d, meta = load_instance_Q_sparsity(n, delta, rep)

            # Build rank-one vector a (reproducible per rep)
            rng = np.random.default_rng(100000 * rep + 17 * n)
            a = rng.standard_normal(n)/n  # the rank-one vector should not dominate Q
            # a = np.zeros(n)

            # Shift term: c = Q d
            Q_dense_for_c = Q.toarray() if hasattr(Q, "toarray") else np.array(Q)
            c = Q_dense_for_c @ d

            BIG_M = BIG_M_INIT

            for tau in tau_list:
                # try:
                    print([n, delta, tau, rep])

                    # indicator penalty vector (same shape as before)
                    lam = tau * np.ones(n) / np.sqrt(n)

                    # ---------------------------------------------------------
                    # Get a tighter big-M by solving the continuous relaxation
                    # (relax z in [0,1]) for the ORIGINAL problem
                    # ---------------------------------------------------------
                    root_bound = [np.inf, -np.inf]

                    model_relax = gp.Model()
                    z_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z')
                    x_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')

                    model_relax.addConstrs(x_relax[i] <= BIG_M * z_relax[i] for i in range(n))
                    model_relax.addConstrs(x_relax[i] >= -BIG_M * z_relax[i] for i in range(n))


                    # ---- IMPORTANT ----
                    # We cannot mix CVXPY inside Gurobi objective.
                    # Build (a'x)^2 directly in Gurobi:
                    # -------------------
                    # Use Gurobi QuadExpr for (a'x)^2:
                    model_relax.setObjective(
                        (x_relax.T @ Q @ x_relax) / 2.0
                        - c.T @ x_relax
                        + (a @ x_relax) * (a @ x_relax)
                        + lam.T @ z_relax
                        + float(d.T @ (Q_dense_for_c @ d)) / 2.0,
                        GRB.MINIMIZE
                    )

                    model_relax.params.OutputFlag = 0
                    model_relax.params.Threads = THREADS
                    model_relax.optimize()

                    print(f"The relaxed obj is {model_relax.objVal}.")
                    x_relax_vals = np.array([x_relax[i].X for i in range(n)])
                    print(f"The largest |x| in relaxation is {max(abs(x_relax_vals))}")

                    BIG_M = min(BIG_M, 2 * max(abs(x_relax_vals)))
                    if not np.isfinite(BIG_M) or BIG_M <= 0:
                        BIG_M = BIG_M_INIT
                    print(f'Use new Big-M {BIG_M}.')

                    # ---------------------------------------------------------
                    # Solve ORIGINAL MIQP: big-M indicator + rank-one term
                    # ---------------------------------------------------------
                    root_bound = [np.inf, -np.inf]

                    model_ori = gp.Model()
                    z_ori = model_ori.addMVar(n, vtype=GRB.BINARY, name="z")
                    x_ori = model_ori.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")

                    model_ori.setObjective(
                        x_ori.T @ Q @ x_ori / 2.0
                        - c.T @ x_ori
                        + (a.T @ x_ori) * (a.T @ x_ori)
                        # + (a @ x_ori) **2
                        + float(d.T @ (Q_dense_for_c @ d)) / 2.0
                        + lam.T @ z_ori,
                        GRB.MINIMIZE
                    )

                    model_ori.addConstrs(x_ori[i] <= BIG_M * z_ori[i] for i in range(n))
                    model_ori.addConstrs(x_ori[i] >= -BIG_M * z_ori[i] for i in range(n))

                    model_ori.params.OutputFlag = 1
                    model_ori.params.Threads = THREADS
                    model_ori.params.TimeLimit = timelimit/10
                    model_ori.optimize(record_root_lb)

                    z_opt_vals = np.array([z_ori[i].X for i in range(n)])
                    result_opt = [
                        n, delta, tau, rep, 'original',
                        root_bound[0], root_bound[1],
                        (root_bound[0] - root_bound[1]) / root_bound[0] if root_bound[0] < np.inf else np.nan,
                        model_ori.ObjVal, model_ori.ObjBound,
                        (model_ori.ObjVal - model_ori.ObjBound) / model_ori.ObjVal if model_ori.ObjVal != 0 else np.nan,
                        np.count_nonzero(z_opt_vals),
                        model_ori.NodeCount,
                        model_ori.runtime
                    ]
                    results.append(result_opt)

                    print('--------------------------------')
                    print('solve the problem in original formulation')
                    print(f"The obj is {model_ori.objVal}.")
                    if root_bound[0] < np.inf:
                        print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. "
                              f"The root gap is: {np.round(100*(root_bound[0]-root_bound[1])/root_bound[0],4)}%. "
                              f"Runtime: {model_ori.runtime}.")
                    print(np.where(z_opt_vals == 1)[0])
                    print(f"Number of nonzeros: {np.count_nonzero(z_opt_vals)}")
                    print('--------------------------------')

                    # ---------------------------------------------------------
                    # Solve CORe formulation (ideal hull): shifted + rank-one
                    # ---------------------------------------------------------
                    root_bound = [np.inf, -np.inf]

                    model_opt = cor_reform_rankone_shift(Q, d, a, lam, xbar=BIG_M,
                                                         timelimit=timelimit, threads=THREADS,
                                                         mip_gap=None, verbose=True)
                    model_opt.params.OutputFlag = 1
                    model_opt.params.Threads = THREADS
                    model_opt.params.TimeLimit = timelimit
                    model_opt.optimize(record_root_lb)

                    z_opt_vals = np.array([1 - model_opt._z0[i].X for i in range(n)])  # inferred on/off
                    result_opt = [
                        n, delta, tau, rep, 'core',
                        root_bound[0], root_bound[1],
                        (root_bound[0] - root_bound[1]) / root_bound[0] if root_bound[0] < np.inf else np.nan,
                        model_opt.ObjVal, model_opt.ObjBound,
                        (model_opt.ObjVal - model_opt.ObjBound) / model_opt.ObjVal if model_opt.ObjVal != 0 else np.nan,
                        np.count_nonzero(z_opt_vals),
                        model_opt.NodeCount,
                        model_opt.runtime
                    ]
                    results.append(result_opt)

                    print('--------------------------------------------------')
                    print("Solve the optimal solution in the CORe formulation")
                    print(f"The obj is {model_opt.objVal}.")
                    if root_bound[0] < np.inf:
                        print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. "
                              f"The root gap is: {np.round(100*(root_bound[0]-root_bound[1])/root_bound[0],4)}%. "
                              f"Runtime: {model_opt.runtime}.")
                    print('--------------------------------------------------')

                    # Save incremental results
                    results_df = pd.DataFrame(
                        results,
                        columns=['n', 'delta', 'tau', 'rep', 'formulation',
                                 'root_ub', 'root_lb', 'root_gap',
                                 'end_ub', 'end_lb', 'end_gap',
                                 'nnz', 'node_count', 'time']
                    )
                    print(results_df)

                    out_dir = os.path.join(current_dir, "../experiments_results")
                    os.makedirs(out_dir, exist_ok=True)
                    results_df.to_csv(f"{out_dir}/Q_rankone_{job_name}.csv", index=False)

                # except Exception as e:
                #     print(f"Error: {e}")
                #     continue

