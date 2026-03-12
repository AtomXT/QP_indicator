import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import GRB
from src.utils import load_instance_socp_sparsity
import os
import argparse
import json
import scipy


current_dir = os.path.dirname(os.path.abspath(__file__))

np.set_printoptions(linewidth=200)


def parse_args():
    p = argparse.ArgumentParser()

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
    s = s.strip()
    if s.startswith("["):
        arr = json.loads(s)
        return [cast(x) for x in arr]
    if "," in s:
        return [cast(x) for x in s.split(",") if x.strip() != ""]
    return [cast(s)]


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


def build_original_socp(Q, c, sigma, mu, M=100.0, timelimit=None, threads=None, verbose=False):
    """
    Original formulation:

        min  c^T x + x0 + mu^T z
        s.t. x_i(1-z_i)=0
             x0 >= sqrt(sigma^2 + x^T Q x)

    modeled as
        x^T Q x + sigma^2 <= x0^2, x0 >= 0
        -Mz_i <= x_i <= Mz_i
    """
    n = c.size
    model = gp.Model("original_socp")

    model.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        model.Params.TimeLimit = float(timelimit)
    if threads is not None:
        model.Params.Threads = int(threads)

    x = model.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    z = model.addMVar(n, vtype=GRB.BINARY, name="z")
    x0 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x0")

    model.addConstrs(x[i] <= M * z[i] for i in range(n))
    model.addConstrs(x[i] >= -M * z[i] for i in range(n))

    model.addConstr(x @ Q @ x + float(sigma) ** 2 <= x0 * x0, name="soc")

    model.setObjective(c @ x + x0 + mu @ z, GRB.MINIMIZE)

    model._z = z
    model._x = x
    model._x0 = x0
    return model


def build_relaxed_socp(Q, c, sigma, mu, M=100.0, threads=None):
    """
    Continuous relaxation used to tighten big-M:
        z in [0,1]
    """
    n = c.size
    model = gp.Model("relaxed_socp")

    x = model.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    z = model.addMVar(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z")
    x0 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x0")

    model.addConstrs(x[i] <= M * z[i] for i in range(n))
    model.addConstrs(x[i] >= -M * z[i] for i in range(n))

    model.addConstr(x @ Q @ x + float(sigma) ** 2 <= x0 * x0, name="soc")

    model.setObjective(c @ x + x0 + mu @ z, GRB.MINIMIZE)

    model.Params.OutputFlag = 0
    if threads is not None:
        model.Params.Threads = int(threads)

    model._x = x
    model._z = z
    model._x0 = x0
    return model


def build_core_socp(Q, c, sigma, mu, M=100.0, timelimit=None, threads=None, verbose=False):
    """
    CORe-enhanced formulation based on the necessary optimality condition:
        if z_i = 1, then
            Q_ii x_i + g_i + c_i x0 = 0
        where
            g_i = sum_{j != i} Q_ij x_j

    We keep:
        -Mz_i <= x_i <= Mz_i
        x^T Q x + sigma^2 <= x0^2
    and use indicator constraints for the on-regime condition.
    """
    n = c.size
    Q_dense = Q.toarray()

    model = gp.Model("core_socp")

    model.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        model.Params.TimeLimit = float(timelimit)
    if threads is not None:
        model.Params.Threads = int(threads)

    x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    g = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g")
    x0 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x0")

    for i in range(n):
        model.addConstr(x[i] <= M * z[i], name=f"x_ub[{i}]")
        model.addConstr(x[i] >= -M * z[i], name=f"x_lb[{i}]")
    # for i in range(n):
    #     model.addGenConstrIndicator(z[i], 0, x[i] == 0.0)

    for i in range(n):
        expr = gp.LinExpr()
        for j in range(n):
            if j != i and Q_dense[i, j] != 0.0:
                expr += Q_dense[i, j] * x[j]
        model.addConstr(g[i] == expr, name=f"g_def[{i}]")

    x_vec = gp.MVar.fromlist([x[i] for i in range(n)])
    model.addConstr(x_vec @ Q @ x_vec + float(sigma) ** 2 <= x0 * x0, name="soc")

    for i in range(n):
        Qii = float(Q_dense[i, i])
        model.addGenConstrIndicator(
            z[i], 1,
            Qii * x[i] + g[i] + float(c[i]) * x0 == 0.0,
            name=f"core_foc[{i}]"
        )

    model.setObjective(
        gp.quicksum(float(c[i]) * x[i] for i in range(n))
        + x0
        + gp.quicksum(float(mu[i]) * z[i] for i in range(n)),
        GRB.MINIMIZE
    )

    model._z = z
    model._x = x
    model._g = g
    model._x0 = x0
    return model


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
            try:
                Q, c, sigma, meta = load_instance_socp_sparsity(n, delta, rep)
                sigma = 0
                # Q += scipy.sparse.eye(n)*2
            except Exception as e:
                print(f"Failed to load instance (n={n}, delta={delta}, rep={rep}): {e}")
                continue

            for tau in tau_list:
                # try:
                    print([n, delta, tau, rep])

                    mu = tau * np.ones(n) / np.sqrt(n)

                    root_bound = [np.inf, -np.inf]

                    # --------------------------------------------------
                    # Tighten big-M using continuous relaxation
                    # --------------------------------------------------
                    BIG_M = BIG_M_INIT

                    model_relax = build_relaxed_socp(Q, c, sigma, mu, M=BIG_M, threads=THREADS)
                    model_relax.optimize()

                    if model_relax.Status == GRB.OPTIMAL:
                        x_relax_vals = np.array(model_relax._x.X)
                        max_abs_x = np.max(np.abs(x_relax_vals)) if x_relax_vals.size > 0 else 0.0
                        BIG_M = min(BIG_M, max(1e-6, 2.0 * max_abs_x))
                        print(f"Relaxed obj = {model_relax.ObjVal}")
                        print(f"max |x_relax| = {max_abs_x}")
                        print(f"Use tightened Big-M = {BIG_M}")
                    else:
                        print("Continuous relaxation not optimal, keep initial Big-M.")

                    # --------------------------------------------------
                    # Original formulation
                    # --------------------------------------------------
                    root_bound = [np.inf, -np.inf]

                    model_ori = build_original_socp(
                        Q, c, sigma, mu,
                        M=BIG_M,
                        timelimit=timelimit,
                        threads=THREADS,
                        verbose=True
                    )
                    model_ori.optimize(record_root_lb)

                    if model_ori.SolCount > 0:
                        z_ori_vals = np.array(model_ori._z.X)
                        end_ub = model_ori.ObjVal
                    else:
                        z_ori_vals = np.zeros(n)
                        end_ub = np.nan

                    end_lb = model_ori.ObjBound if model_ori.Status not in [GRB.INF_OR_UNBD, GRB.INFEASIBLE] else np.nan
                    root_gap = np.nan
                    end_gap = np.nan

                    if np.isfinite(root_bound[0]) and np.isfinite(root_bound[1]) and abs(root_bound[0]) > 1e-12:
                        root_gap = (root_bound[0] - root_bound[1]) / abs(root_bound[0])

                    if model_ori.SolCount > 0 and np.isfinite(end_lb) and abs(end_ub) > 1e-12:
                        end_gap = (end_ub - end_lb) / abs(end_ub)

                    results.append([
                        n, delta, tau, rep, "original",
                        root_bound[0], root_bound[1], root_gap,
                        end_ub, end_lb, end_gap,
                        int(np.count_nonzero(z_ori_vals > 0.5)),
                        model_ori.NodeCount,
                        model_ori.Runtime,
                    ])
                    print(max([abs(model_ori._x[i].X) for i in range(n)]))

                    print('--------------------------------')
                    print('solve the problem in original formulation')
                    if model_ori.SolCount > 0:
                        print(f"Obj = {model_ori.ObjVal}")
                    print(f"Root UB = {root_bound[0]}, Root LB = {root_bound[1]}, Root gap = {root_gap}")
                    print(f"Node count = {model_ori.NodeCount}, Runtime = {model_ori.Runtime}")
                    print(f"nnz(z) = {np.count_nonzero(z_ori_vals > 0.5)}")
                    print('--------------------------------')

                    # --------------------------------------------------
                    # CORe formulation
                    # --------------------------------------------------
                    root_bound = [np.inf, -np.inf]

                    model_core = build_core_socp(
                        Q, c, sigma, mu,
                        M=BIG_M,
                        timelimit=timelimit,
                        threads=THREADS,
                        verbose=True
                    )
                    model_core.optimize(record_root_lb)

                    if model_core.SolCount > 0:
                        z_core_vals = np.array([model_core._z[i].X for i in range(n)])
                        end_ub = model_core.ObjVal
                    else:
                        z_core_vals = np.zeros(n)
                        end_ub = np.nan

                    end_lb = model_core.ObjBound if model_core.Status not in [GRB.INF_OR_UNBD, GRB.INFEASIBLE] else np.nan
                    root_gap = np.nan
                    end_gap = np.nan

                    if np.isfinite(root_bound[0]) and np.isfinite(root_bound[1]) and abs(root_bound[0]) > 1e-12:
                        root_gap = (root_bound[0] - root_bound[1]) / abs(root_bound[0])

                    if model_core.SolCount > 0 and np.isfinite(end_lb) and abs(end_ub) > 1e-12:
                        end_gap = (end_ub - end_lb) / abs(end_ub)

                    results.append([
                        n, delta, tau, rep, "core",
                        root_bound[0], root_bound[1], root_gap,
                        end_ub, end_lb, end_gap,
                        int(np.count_nonzero(z_core_vals > 0.5)),
                        model_core.NodeCount,
                        model_core.Runtime,
                    ])

                    print('--------------------------------')
                    print('solve the problem in CORe formulation')
                    if model_core.SolCount > 0:
                        print(f"Obj = {model_core.ObjVal}")
                    print(f"Root UB = {root_bound[0]}, Root LB = {root_bound[1]}, Root gap = {root_gap}")
                    print(f"Node count = {model_core.NodeCount}, Runtime = {model_core.Runtime}")
                    print(f"nnz(z) = {np.count_nonzero(z_core_vals > 0.5)}")
                    print('--------------------------------')

                    results_df = pd.DataFrame(
                        results,
                        columns=[
                            'n', 'delta', 'tau', 'rep', 'formulation',
                            'root_ub', 'root_lb', 'root_gap',
                            'end_ub', 'end_lb', 'end_gap',
                            'nnz', 'node_count', 'time'
                        ]
                    )

                    out_dir = f"{current_dir}/../experiments_results"
                    os.makedirs(out_dir, exist_ok=True)
                    results_df.to_csv(f"{out_dir}/SOCP_sparsity_{job_name}.csv", index=False)
                    print(results_df)

                # except Exception as e:
                #     print(f"Error on (n={n}, delta={delta}, tau={tau}, rep={rep}): {e}")
                #     continue