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
    p.add_argument("--rep_list", type=str, default="0",
                   help='e.g. "0" or "0,1,2" or "[0,1,2]"')
    p.add_argument("--tau_list", type=str, default="0.1",
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


def cor_reform(Q, d, lam, M=100, timelimit=None, mip_gap=None, threads=None, verbose=False):
    """
    Solve the MIQP using the g-based 3-way convex-hull (Balas) reformulation:

      For each i, enforce (x_i, g_i) ∈ D_i^0 ∪ D_i^+ ∪ D_i^- via an ideal hull formulation:
        D_i^0: x_i = 0, |g_i| ≤ tau_i
        D_i^+: Q_ii x_i + g_i = 0, g_i ≥ tau_i
        D_i^-: Q_ii x_i + g_i = 0, g_i ≤ -tau_i
      with g_i = c_i + sum_{j!=i} Q_ij x_j  (off-diagonal field)

    Requires: gurobipy

    Parameters
    ----------
    Q : (n,n) array_like, symmetric PSD recommended. Needs Q_ii > 0 for all i.
    c : (n,) array_like
    lam : (n,) array_like, assumed >= 0
    xbar : float or (n,) array_like, bound |x_i| <= xbar_i (required to bound g and make hull closed)
    time_limit : float, optional (seconds)
    mip_gap : float, optional (relative MIP gap)
    verbose : bool

    Returns
    -------
    result : dict with keys:
        status, obj, x, z_plus, z_minus, z0
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ImportError("This function requires gurobipy (Gurobi Python API).") from e
    Q = Q.toarray()
    n = d.size
    c = -Q.T@d
    xbar = M*np.ones(n)

    # neighbor sets from off-diagonal nonzeros
    # (tree structure just makes these small; code works for any sparsity)
    N = [[] for _ in range(n)]
    for i in range(n):
        # include j != i with Q_ij != 0
        nz = np.nonzero(Q[i, :])[0]
        for j in nz:
            if j != i and Q[i, j] != 0.0:
                N[i].append(j)

    # constants for hull
    tau = np.sqrt(2.0 * lam * np.diag(Q))
    # valid |g_i| bound from |x_j|<=xbar_j
    Qabs = np.abs(Q)
    np.fill_diagonal(Qabs, 0)  # optional
    G = np.abs(c) + Qabs @ xbar

    m = gp.Model("g_hull_miqp")
    m.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)
    if threads:
        m.Params.Threads = threads
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    # Aggregate variables
    x = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    g = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g")

    # Disaggregated per-regime variables
    x0 = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x0")
    xp = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xp")
    xm = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xm")

    g0 = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g0")
    gpv = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="gplus")
    gmv = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="gminus")

    # Regime binaries
    z0 = m.addVars(n, vtype=GRB.BINARY, name="z0")
    zp = m.addVars(n, vtype=GRB.BINARY, name="zplus")
    zm = m.addVars(n, vtype=GRB.BINARY, name="zminus")

    # Bounds on aggregate x (required)
    for i in range(n):
        m.addConstr(x[i] <= xbar[i], name=f"x_ub[{i}]")
        m.addConstr(x[i] >= -xbar[i], name=f"x_lb[{i}]")

    # g definition and optional explicit g bounds
    for i in range(n):
        expr = c[i]
        if N[i]:
            expr += gp.quicksum(Q[i, j] * x[j] for j in N[i])
        # off-diagonal sum: we excluded j=i already
        m.addConstr(g[i] == expr, name=f"g_def[{i}]")
        m.addConstr(g[i] <= G[i], name=f"g_ub[{i}]")
        m.addConstr(g[i] >= -G[i], name=f"g_lb[{i}]")

    # Hull coupling: x = x0+xp+xm, g = g0+gplus+gminus, select regime
    for i in range(n):
        m.addConstr(x[i] == x0[i] + xp[i] + xm[i], name=f"disagg_x[{i}]")
        m.addConstr(g[i] == g0[i] + gpv[i] + gmv[i], name=f"disagg_g[{i}]")
        m.addConstr(z0[i] + zp[i] + zm[i] == 1, name=f"regime_sum[{i}]")

        # Off piece: x0 = 0, |g0| <= tau_i z0
        m.addConstr(x0[i] == 0, name=f"off_x0[{i}]")
        m.addConstr(g0[i] <= tau[i] * z0[i], name=f"off_g0_ub[{i}]")
        m.addConstr(g0[i] >= -tau[i] * z0[i], name=f"off_g0_lb[{i}]")

        # On+ piece: Q_ii xp + gplus = 0, tau z+ <= gplus <= G z+
        Qii = Q[i, i]
        m.addConstr(Qii * xp[i] + gpv[i] == 0, name=f"onp_stat[{i}]")
        m.addConstr(gpv[i] >= tau[i] * zp[i], name=f"onp_g_lb[{i}]")
        m.addConstr(gpv[i] <= G[i] * zp[i], name=f"onp_g_ub[{i}]")

        # On- piece: Q_ii xm + gminus = 0, -G z- <= gminus <= -tau z-
        m.addConstr(Qii * xm[i] + gmv[i] == 0, name=f"onm_stat[{i}]")
        m.addConstr(gmv[i] >= -G[i] * zm[i], name=f"onm_g_lb[{i}]")
        m.addConstr(gmv[i] <= -tau[i] * zm[i], name=f"onm_g_ub[{i}]")

        # (Optional but often helps) also bound disaggregated vars tightly
        # These follow from x bounds and disaggregation; keeps numerics stable.
        m.addConstr(xp[i] <= xbar[i] * zp[i], name=f"xp_ub[{i}]")
        m.addConstr(xp[i] >= -xbar[i] * zp[i], name=f"xp_lb[{i}]")
        m.addConstr(xm[i] <= xbar[i] * zm[i], name=f"xm_ub[{i}]")
        m.addConstr(xm[i] >= -xbar[i] * zm[i], name=f"xm_lb[{i}]")
        # x0 is fixed 0 so no need for bounds there.

    # Objective: 1/2 x^T Q x + c^T x + sum lambda_i (zplus+zminus)
    # Build quadratic form explicitly
    obj = gp.QuadExpr()
    # 0.5 * sum_{i,j} Q_ij x_i x_j
    for i in range(n):
        # diagonal
        obj.add(0.5 * Q[i, i] * x[i] * x[i])
        # off-diagonal (i<j)
        for j in range(i + 1, n):
            if Q[i, j] != 0.0:
                obj.add(Q[i, j] * x[i] * x[j])
    # linear term c^T x
    obj.add(gp.LinExpr(c.tolist(), [x[i] for i in range(n)]))
    # penalty
    obj.add(gp.quicksum(lam[i] * (zp[i] + zm[i]) for i in range(n)))
    # constant
    obj.add(0.5*d.T@Q@d)

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
# data_list = ['diabetes', 'autompg']
results = []

for n in n_list:
    for delta in delta_list:
        for rep in rep_list:
            Q, d, meta = load_instance_Q_sparsity(n, delta, rep)
            c = -Q.T @ d
            BIG_M = BIG_M_INIT
            for tau in tau_list:
                print([n, delta, tau, rep])
                lam = tau * np.ones(n) / np.sqrt(n)
                # define a container to store the root node lower bound
                root_bound = [np.inf, -np.inf]

                # get tight big-M
                model_relax = gp.Model()
                z_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z')
                x_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
                # add constraints
                model_relax.addConstrs(x_relax[i] <= BIG_M*z_relax[i] for i in range(n))
                model_relax.addConstrs(x_relax[i] >= -BIG_M*z_relax[i] for i in range(n))

                # set objective
                eqn = x_relax.T@Q@x_relax/2 + c.T@x_relax + d.T@Q@d/2 + lam.T@z_relax
                model_relax.setObjective(eqn, GRB.MINIMIZE)
                # model_relax.params.QCPDual = 1
                model_relax.params.OutputFlag = 0
                model_relax.params.Threads = THREADS
                model_relax.optimize()
                print(f"The relaxed obj is {model_relax.objVal}.")
                x_relax_vals = np.array([x_relax[i].X for i in range(n)])
                print(f"The larges value of x is {max(x_relax_vals)}")
                BIG_M = min(BIG_M, 2*max(abs(x_relax_vals)))
                print(f'Use new Big-M {BIG_M}.')

                ## solve the problem in original formulation
                model_ori = gp.Model()
                z_ori = model_ori.addMVar(n, vtype=GRB.BINARY)
                x_ori = model_ori.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
                model_ori.setObjective(x_ori.T@Q@x_ori/2 + c.T@x_ori + d.T@Q@d/2 + lam.T@z_ori, GRB.MINIMIZE)
                model_ori.addConstrs(x_ori[i] <= BIG_M*z_ori[i] for i in range(n))
                model_ori.addConstrs(x_ori[i] >= -BIG_M*z_ori[i] for i in range(n))
                model_ori.params.OutputFlag = 1
                model_ori.params.Threads = THREADS
                model_ori.params.TimeLimit = timelimit
                model_ori.optimize(record_root_lb)
                z_opt_vals = np.array([z_ori[i].X for i in range(n)])
                result_opt = [n, delta, tau, rep, 'original', root_bound[0], root_bound[1],
                          (root_bound[0] - root_bound[1]) / root_bound[0], model_ori.ObjVal, model_ori.ObjBound,
                          (model_ori.ObjVal - model_ori.ObjBound) / model_ori.ObjVal, model_ori.NodeCount,
                          model_ori.runtime]
                results.append(result_opt)
                print('--------------------------------')
                print('solve the problem in original formulation')
                print(f"The obj is {model_ori.objVal}.")
                print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100*(root_bound[0]-root_bound[1])/root_bound[0],4)}%. Runtime: {model_ori.runtime}.")
                print(np.where(z_opt_vals == 1)[0])
                print(f"Number of outliers: {np.count_nonzero(z_opt_vals)}")
                print('--------------------------------')

                ## solve the optimal solution in the proposed formulation
                root_bound = [np.inf, -np.inf]

                model_opt = cor_reform(Q, d, lam, BIG_M)
                model_opt.params.OutputFlag = 1
                # model_opt.params.PreMIQCPForm = 1
                model_opt.params.Threads = THREADS
                model_opt.params.TimeLimit = timelimit
                model_opt.optimize(record_root_lb)
                result_opt = [n, delta, tau, rep, 'opt', root_bound[0], root_bound[1],
                              (root_bound[0] - root_bound[1]) / root_bound[0], model_opt.ObjVal, model_opt.ObjBound,
                              (model_opt.ObjVal - model_opt.ObjBound) / model_opt.ObjVal, model_opt.NodeCount,
                              model_opt.runtime]
                results.append(result_opt)
                print('--------------------------------------------------')
                print("Solve the optimal solution in the proposed formulation")
                print(f"The obj is {model_opt.objVal}.")
                print(
                    f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100 * (root_bound[0] - root_bound[1]) / root_bound[0], 4)}%. Runtime: {model_opt.runtime}.")
                print('--------------------------------------------------')


                results_df = pd.DataFrame(results, columns=['n', 'delta', 'tau', 'rep' ,'formulation','root_ub','root_lb','root_gap','end_ub','end_lb','end_gap','node_count','time'])
                print(results_df)
                results_df.to_csv(f"{current_dir}/../experiments_results/{job_name}.csv", index=False)


