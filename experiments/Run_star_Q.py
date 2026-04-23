import argparse
import json
import os
import time

import gurobipy as gp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from gurobipy import GRB


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

np.set_printoptions(linewidth=200)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--n_list", type=str, default="1000",
                   help='e.g. "500" or "50,60,70" or "[50,60,70]"')
    p.add_argument("--rep_list", type=str, default="0",
                   help='e.g. "0" or "0,1,2" or "[0,1,2]"')
    p.add_argument("--tau_list", type=str, default="0.2",
                   help='e.g. "0.05,0.1,0.2"')
    p.add_argument("--timelimit", type=float, default=10.0)
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


def load_instance_Q_star(n: int, rep: int):
    """
    Load one star-structured dataset specified by (n, rep).

    Returns:
        Q   : scipy.sparse.csr_matrix
        d   : numpy array
        meta: dict
    """
    data_root = os.path.join(project_root, "data", "Q_star")
    fname = f"inst_n{n}_star_rep{rep:02d}.npz"
    path = os.path.join(data_root, f"n={n}", fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Instance not found: {path}")

    obj = np.load(path, allow_pickle=True)

    Q = sp.csr_matrix(
        (obj["Q_data"], obj["Q_indices"], obj["Q_indptr"]),
        shape=tuple(obj["Q_shape"])
    )

    d = obj["d"]
    meta = json.loads(str(obj["meta_json"]))
    return Q, d, meta


root_bound = [np.inf, -np.inf]


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


def cor_reform(Q, d, lam, M=100, timelimit=None, mip_gap=None, threads=None, verbose=False):
    """
    Solve the MIQP using the g-based 3-way convex-hull (Balas) reformulation.
    """
    Q = Q.toarray()
    n = d.size
    c = -Q.T @ d
    xbar = M * np.ones(n)

    N = [[] for _ in range(n)]
    for i in range(n):
        nz = np.nonzero(Q[i, :])[0]
        for j in nz:
            if j != i and Q[i, j] != 0.0:
                N[i].append(j)

    tau = np.sqrt(2.0 * lam * np.diag(Q))
    Qabs = np.abs(Q)
    np.fill_diagonal(Qabs, 0)
    G = np.abs(c) + Qabs @ xbar

    m = gp.Model("g_hull_miqp")
    m.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)
    if threads:
        m.Params.Threads = threads
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

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

    obj = 0.5 * x @ Q @ x + c@x + lam@(zp+zm) + 0.5*d@Q@d

    m.setObjective(obj, GRB.MINIMIZE)
    return m


args = parse_args()

n_list = parse_list(args.n_list, int)
rep_list = parse_list(args.rep_list, int)
tau_list = parse_list(args.tau_list, float)
timelimit = args.timelimit
THREADS = args.threads
BIG_M_INIT = float(args.big_m_init)

job_name = args.job_name
results = []

for n in n_list:
    for rep in rep_list:
        Q, d, meta = load_instance_Q_star(n, rep)

        c = -Q.T @ d
        BIG_M = BIG_M_INIT
        for tau in tau_list:
            try:
                print([n, tau, rep])
                lam = tau * np.ones(n) * np.log(n) / n
                root_bound = [np.inf, -np.inf]

                model_relax = gp.Model()
                z_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z")
                x_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")

                model_relax.addConstrs(x_relax[i] <= BIG_M * z_relax[i] for i in range(n))
                model_relax.addConstrs(x_relax[i] >= -BIG_M * z_relax[i] for i in range(n))

                eqn = x_relax.T @ Q @ x_relax / 2 + c.T @ x_relax + d.T @ Q @ d / 2 + lam.T @ z_relax
                model_relax.setObjective(eqn, GRB.MINIMIZE)
                model_relax.params.OutputFlag = 0
                model_relax.params.Threads = THREADS
                model_relax.optimize()

                print(f"The relaxed obj is {model_relax.objVal}.")
                x_relax_vals = np.array([x_relax[i].X for i in range(n)])
                print(f"The larges value of x is {max(x_relax_vals)}")
                BIG_M = min(BIG_M, 2 * max(abs(x_relax_vals)))
                print(f"Use new Big-M {BIG_M}.")

                model_ori = gp.Model()
                z_ori = model_ori.addMVar(n, vtype=GRB.BINARY)
                x_ori = model_ori.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
                model_ori.setObjective(x_ori.T @ Q @ x_ori / 2 + c.T @ x_ori + d.T @ Q @ d / 2 + lam.T @ z_ori, GRB.MINIMIZE)
                model_ori.addConstrs(x_ori[i] <= BIG_M * z_ori[i] for i in range(n))
                model_ori.addConstrs(x_ori[i] >= -BIG_M * z_ori[i] for i in range(n))
                model_ori.params.OutputFlag = 1
                model_ori.params.Threads = THREADS
                model_ori.params.TimeLimit = timelimit
                model_ori.optimize(record_root_lb)

                z_ori_vals = np.array([z_ori[i].X for i in range(n)])
                result_opt = [
                    n, tau, rep, "original", root_bound[0], root_bound[1],
                    (root_bound[0] - root_bound[1]) / root_bound[0], model_ori.ObjVal, model_ori.ObjBound,
                    (model_ori.ObjVal - model_ori.ObjBound) / model_ori.ObjVal, np.count_nonzero(z_ori_vals),
                    model_ori.NodeCount, model_ori.runtime
                ]
                results.append(result_opt)
                print("--------------------------------")
                print("solve the problem in original formulation")
                print(f"The obj is {model_ori.objVal}.")
                print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100 * (root_bound[0] - root_bound[1]) / root_bound[0], 4)}%. Runtime: {model_ori.runtime}.")
                print(np.where(z_ori_vals == 1)[0])
                print(f"Number of outliers: {np.count_nonzero(z_ori_vals)}")
                print("--------------------------------")

                root_bound = [np.inf, -np.inf]

                model_opt = cor_reform(Q, d, lam, BIG_M)
                model_opt.params.OutputFlag = 1
                model_opt.params.Threads = THREADS
                model_opt.params.TimeLimit = timelimit
                model_opt.optimize(record_root_lb)

                z_opt_vals = np.array([1 - model_opt._z0[i].X for i in range(n)])
                result_opt = [
                    n, tau, rep, "opt", root_bound[0], root_bound[1],
                    (root_bound[0] - root_bound[1]) / root_bound[0], model_opt.ObjVal, model_opt.ObjBound,
                    (model_opt.ObjVal - model_opt.ObjBound) / model_opt.ObjVal, np.count_nonzero(z_opt_vals),
                    model_opt.NodeCount, model_opt.runtime
                ]
                results.append(result_opt)
                print("--------------------------------------------------")
                print("Solve the optimal solution in the proposed formulation")
                print(f"The obj is {model_opt.objVal}.")
                print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100 * (root_bound[0] - root_bound[1]) / root_bound[0], 4)}%. Runtime: {model_opt.runtime}.")
                print("--------------------------------------------------")

                results_df = pd.DataFrame(
                    results,
                    columns=["n", "tau", "rep", "formulation", "root_ub", "root_lb", "root_gap",
                             "end_ub", "end_lb", "end_gap", "nnz", "node_count", "time"]
                )
                print(results_df)
                results_df.to_csv(f"{current_dir}/../experiments_results/star_Q_{job_name}.csv", index=False)
            except Exception as e:
                print(f"Error: {e}")
                continue
