
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import gurobipy as gp
import cvxpy as cp
import random
from gurobipy import GRB
from src.utils import decomposition, get_data, get_data_offline
import os
import argparse
import json


current_dir = os.path.dirname(os.path.abspath(__file__))

np.set_printoptions(linewidth=200)


from itertools import combinations
from src.rank2 import fast_dp_general

def parse_args():
    p = argparse.ArgumentParser()

    # allow scalar or list-like inputs
    p.add_argument("--dataset", type=str, default="diabetes",
                   help='e.g. "diabetes" or "diabetes,crime"')
    p.add_argument("--sample_size", type=int, default=0)
    p.add_argument("--dimension", type=int, default=0)
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


def cor_reform(Q, c, lam, M=100, timelimit=None, mip_gap=None, threads=None, verbose=False):
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
    n = len(c)
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
    m._x = x
    m._g = g  # useful to check the interaction level

    # Disaggregated per-regime variables
    # x0 = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x0")
    xp = m.addVars(n, lb=-GRB.INFINITY, ub=0, name="xp")
    xm = m.addVars(n, lb=0, ub=GRB.INFINITY, name="xm")

    g0 = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g0")
    gpv = m.addVars(n, lb=0, ub=GRB.INFINITY, name="gplus")
    gmv = m.addVars(n, lb=0, ub=GRB.INFINITY, name="gminus")

    # Regime binaries
    z0 = m.addVars(n, vtype=GRB.BINARY, name="z0")
    m._z0 = z0

    zp = m.addVars(n, vtype=GRB.BINARY, name="zplus")
    zm = m.addVars(n, vtype=GRB.BINARY, name="zminus")
    m._zp = zp
    m._zm = zm
    # # Bounds on aggregate x (required)
    # for i in range(n):
    #     m.addConstr(x[i] <= xbar[i], name=f"x_ub[{i}]")
    #     m.addConstr(x[i] >= -xbar[i], name=f"x_lb[{i}]")

    # g definition and optional explicit g bounds
    for i in range(n):
        expr = c[i]
        if N[i]:
            expr += gp.quicksum(Q[i, j] * x[j] for j in N[i])
        # off-diagonal sum: we excluded j=i already
        m.addConstr(g[i] == expr, name=f"g_def[{i}]")
        # m.addConstr(g[i] <= G[i], name=f"g_ub[{i}]")
        # m.addConstr(g[i] >= -G[i], name=f"g_lb[{i}]")

    # Hull coupling: x = x0+xp+xm, g = g0+gplus+gminus, select regime
    for i in range(n):
        m.addConstr(x[i] == xp[i] + xm[i], name=f"disagg_x[{i}]")
        m.addConstr(g[i] == g0[i] + gpv[i] - gmv[i], name=f"disagg_g[{i}]")
        m.addConstr(z0[i] + zp[i] + zm[i] == 1, name=f"regime_sum[{i}]")

        # Off piece: x0 = 0, |g0| <= tau_i z0
        # m.addConstr(x0[i] == 0, name=f"off_x0[{i}]")
        m.addConstr(g0[i] <= tau[i] * z0[i], name=f"off_g0_ub[{i}]")
        m.addConstr(g0[i] >= -tau[i] * z0[i], name=f"off_g0_lb[{i}]")

        # On+ piece: Q_ii xp + gplus = 0, tau z+ <= gplus <= G z+
        Qii = Q[i, i]
        m.addConstr(Qii * xp[i] + gpv[i] == 0, name=f"onp_stat[{i}]")
        m.addConstr(gpv[i] >= tau[i] * zp[i], name=f"onp_g_lb[{i}]")
        # m.addConstr(gpv[i] <= G[i] * zp[i], name=f"onp_g_ub[{i}]")

        # On- piece: Q_ii xm + gminus = 0, -G z- <= gminus <= -tau z-
        m.addConstr(Qii * xm[i] - gmv[i] == 0, name=f"onm_stat[{i}]")
        # m.addConstr(-gmv[i] >= -G[i] * zm[i], name=f"onm_g_lb[{i}]")
        m.addConstr(-gmv[i] <= -tau[i] * zm[i], name=f"onm_g_ub[{i}]")

        # # (Optional but often helps) also bound disaggregated vars tightly
        # # These follow from x bounds and disaggregation; keeps numerics stable.
        # m.addConstr(xp[i] <= xbar[i] * zp[i], name=f"xp_ub[{i}]")
        m.addConstr(xp[i] >= -xbar[i] * zp[i], name=f"xp_lb[{i}]")
        m.addConstr(xm[i] <= xbar[i] * zm[i], name=f"xm_ub[{i}]")
        # m.addConstr(xm[i] >= -xbar[i] * zm[i], name=f"xm_lb[{i}]")
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
    # turn off if needed
    # m.setParam("Presolve", 0)
    # m.setParam("Cuts", 0)
    # m.setParam("Heuristics", 0)
    # m.setParam("Aggregate", 0)
    # m.setParam("PrePasses", 0)
    # m.setParam("MIPFocus", 0)

    return m

args = parse_args()

timelimit = args.timelimit
THREADS = args.threads
BIG_M = float(args.big_m_init)
sample_size = int(args.sample_size)
dimension = int(args.dimension)
job_name = args.job_name
data_list = parse_list(args.dataset, str)
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

# Access features and target
# data_list = ['diabetes', 'autompg']

results = []
for dataset in data_list:
    X, y = get_data_offline(dataset)
    if dimension:
        X = X[:, :dimension]
    if sample_size:
        X = X[:sample_size,:]
        y = y[:sample_size]
    n, m = X.shape
# try:
    print([m, n, dataset])
    X = (X - np.mean(X, axis=0)) / X.std(axis=0)
    y = (y - np.mean(y)) / np.std(y)
    mu_g = 0.5
    G = X.T@X + 2*mu_g*np.eye(m)  # regularization
    D = np.eye(n)
    F = X

    c = -y
    d = - X.T@y

    mu = 1
    lam = mu*np.ones((n, 1))
    print(np.linalg.eigvalsh(np.bmat([[D, F], [F.T, G]]))[0:5])

    # define a container to store the root node lower bound
    root_bound = [np.inf, -np.inf]


    # get tight big-M
    model_relax = gp.Model()
    z_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z')
    x_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
    y_relax = model_relax.addMVar(m, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
    # add constraints
    model_relax.addConstrs(x_relax[i] <= BIG_M*z_relax[i] for i in range(n))
    model_relax.addConstrs(x_relax[i] >= -BIG_M*z_relax[i] for i in range(n))

    # set objective
    eqn = y.T@y/2 + y_relax.T@G@y_relax/2 + x_relax.T@D@x_relax/2 + x_relax.T@F@y_relax + c.T@x_relax + d.T@y_relax + lam.T@z_relax
    model_relax.setObjective(eqn[0], GRB.MINIMIZE)
    # model_relax.params.QCPDual = 1
    model_relax.params.OutputFlag = 0
    model_relax.params.Threads = 8
    model_relax.optimize()
    print(f"The relaxed obj is {model_relax.objVal}.")
    x_relax_vals = np.array([x_relax[i].X for i in range(n)])
    y_relax_vals = np.array([y_relax[i].X for i in range(m)])
    print(f"The larges value of x is {max(x_relax_vals)}")
    print(f"The larges value of y is {max(y_relax_vals)}")
    BIG_M = min(BIG_M, 2*max(abs(x_relax_vals)))
    print(f'Use new Big-M {BIG_M}.')
    continous_bound = 1.5*np.abs(F@y_relax_vals + c).max()


    ## solve the problem in original formulation
    model_ori = gp.Model()
    z_ori = model_ori.addMVar(n, vtype=GRB.BINARY)
    beta_ori = model_ori.addMVar(m, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta")
    w_ori = model_ori.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w")
    res = y - X@beta_ori - w_ori
    model_ori.setObjective(res@res/2 + mu_g*beta_ori.T@beta_ori + lam.T@z_ori, GRB.MINIMIZE)
    model_ori.addConstrs(w_ori[i] <= BIG_M*z_ori[i] for i in range(n))
    model_ori.addConstrs(w_ori[i] >= -BIG_M*z_ori[i] for i in range(n))
    model_ori.params.OutputFlag = 1
    model_ori.params.Threads = 8
    model_ori.params.TimeLimit = timelimit
    model_ori.params.NodefileStart = 1
    model_ori.optimize(record_root_lb)
    z_ori_vals = np.array([z_ori[i].X for i in range(n)])
    result_opt = [m, n, dataset, 'original', root_bound[0], root_bound[1],
              (root_bound[0] - root_bound[1]) / root_bound[0], model_ori.ObjVal, model_ori.ObjBound,
              (model_ori.ObjVal - model_ori.ObjBound) / model_ori.ObjVal, , np.count_nonzero(z_ori_vals), model_ori.NodeCount,
              model_ori.runtime]
    results.append(result_opt)
    print('--------------------------------')
    print('solve the problem in original formulation')
    print(f"The obj is {model_ori.objVal}.")
    print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100*(root_bound[0]-root_bound[1])/root_bound[0],4)}%. Runtime: {model_ori.runtime}.")
    print(np.where(z_ori_vals == 1)[0])
    print(f"Number of outliers: {np.count_nonzero(z_ori_vals)}")
    print('--------------------------------')



    root_bound = [np.inf, -np.inf]
    # convenience: diagonal entries
    Dii = np.diag(D) if D.ndim == 2 else np.asarray(D)

    # tau_i = sqrt(2 * lam[i] * Dii[i])  (matches your thresholds)
    tau = np.sqrt(2.0 * lam * Dii)


    # model_opt = cor_reform(Q, combined_c, lam, BIG_M)
    model_opt = gp.Model()

    # binaries
    z = model_opt.addMVar(n, vtype=GRB.BINARY, name="z")
    zp = model_opt.addMVar(n, vtype=GRB.BINARY, name="zp")
    zm = model_opt.addMVar(n, vtype=GRB.BINARY, name="zm")
    model_opt._z = z

    # continuous
    y_opt = model_opt.addMVar(m, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")

    t0 = model_opt.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t0")
    tp = model_opt.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tp")
    tm = model_opt.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tm")

    x = model_opt.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    xp = model_opt.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xp")
    xm = model_opt.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xm")

    L = -continous_bound * np.ones(n)
    U = continous_bound * np.ones(n)

    for i in range(n):
        model_opt.addConstr(x[i] <= BIG_M * z[i])
        model_opt.addConstr(x[i] >= -BIG_M * z[i])
        model_opt.addConstr(xp[i] <= 0)
        model_opt.addConstr(xp[i] >= -BIG_M * zp[i])
        model_opt.addConstr(xm[i] <= BIG_M * zm[i])
        model_opt.addConstr(xm[i] >= 0)

        model_opt.addConstr(z[i] == zp[i] + zm[i])
        model_opt.addConstr(x[i] == xp[i] + xm[i])
        model_opt.addConstr(t0[i] <= tau[i] * (1 - z[i]))
        model_opt.addConstr(t0[i] >= -tau[i] * (1 - z[i]))

        model_opt.addConstr(F[i,:]@y_opt + c[i] == t0[i] + tp[i] + tm[i])
        model_opt.addConstr(tp[i] >= tau[i] * zp[i])
        model_opt.addConstr(tp[i] <= U[i] * zp[i])
        model_opt.addConstr(tm[i] <= -tau[i] * zm[i])
        model_opt.addConstr(tm[i] >= L[i] * zm[i])
        model_opt.addConstr(xp[i] == -tp[i] / D[i,i])
        model_opt.addConstr(xm[i] == -tm[i] / D[i, i])


    # -------------------
    # objective (match your style; note: x^T F y is quadratic)
    # -------------------
    obj = (
            y.T@y/2
            + x.T @ D @ x / 2
            + y_opt.T @ G @ y_opt / 2
            + x.T @ F @ y_opt
            + c.T @ x
            + d.T @ y_opt
            + lam.T @ z
    )

    model_opt.setObjective(obj[0], GRB.MINIMIZE)

    # params (same style)
    model_opt.params.OutputFlag = 1
    model_opt.params.Threads = 8
    model_opt.params.TimeLimit = timelimit
    model_opt.params.NodefileStart = 1

    # # check presolved model
    # # p = model_opt.presolve()
    # # p.write('presolved_model.mps')
    # model_opt.params.OutputFlag = 1
    # # model_opt.params.PreMIQCPForm = 1
    # model_opt.params.Threads = THREADS
    # model_opt.params.TimeLimit = timelimit
    model_opt.optimize(record_root_lb)
    z_opt_vals = np.array([1-model_opt._z[i].X for i in range(n)])
    result_hull = [m, n, dataset, 'opt', root_bound[0], root_bound[1],
                  (root_bound[0] - root_bound[1]) / root_bound[0], model_opt.ObjVal, model_opt.ObjBound,
                  (model_opt.ObjVal - model_opt.ObjBound) / model_opt.ObjVal, np.count_nonzero(z_opt_vals), model_opt.NodeCount,
                  model_opt.runtime]
    results.append(result_hull)
    print('--------------------------------------------------')
    print("Solve the optimal solution in the convex-hull formulation")
    print(f"The obj is {model_opt.objVal}.")
    print(
        f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100 * (root_bound[0] - root_bound[1]) / root_bound[0], 4)}%. Runtime: {model_opt.runtime}.")
    print('--------------------------------------------------')

    results_df = pd.DataFrame(results, columns=['m','n','dataset','formulation','root_ub','root_lb','root_gap','end_ub','end_lb','end_gap','nnz','node_count','time'])
    print(results_df)
    results_df.to_csv(f"{current_dir}/../experiments_results/diagonal_minus_low_rank_{job_name}.csv", index=False)
# except:
#     continue


