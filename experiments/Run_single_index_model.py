import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import gurobipy as gp
from gurobipy import nlfunc
import cvxpy as cp
import random
from gurobipy import GRB
from src.utils import decomposition, get_data, get_data_offline, load_instance_Q_sparsity
from sklearn.datasets import load_breast_cancer
import os
import argparse
import json

import networkx as nx
from networkx.algorithms.approximation.treewidth import treewidth_min_fill_in, treewidth_min_degree


current_dir = os.path.dirname(os.path.abspath(__file__))

np.set_printoptions(linewidth=200)

def parse_args():
    p = argparse.ArgumentParser()

    # allow scalar or list-like inputs
    p.add_argument("--n_list", type=str, default="50",
                   help='e.g. "500" or "50,60,70" or "[50,60,70]"')
    p.add_argument("--tau_list", type=str, default="20",
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


def cor_reform(
    X,
    y,
    lam,
    timelimit=None,
    threads=None,
    verbose=False,
):
    """
    Build the exact CORe formulation for robust binary classification:

        min_{theta,t0,t1,u,z}  sum_i u_i + sum_i lam_i z_i
        s.t.
            s_i <x_i, theta> = t0_i + t1_i
            tau_i (1-z_i) <= t0_i <= U_i (1-z_i)
            L_i z_i <= t1_i <= tau_i z_i
            u_i >= log(1 + exp(-t0_i)) - B_i z_i
            0 <= u_i <= B_i (1-z_i)
            z_i in {0,1}

    where
        s_i = 2 y_i - 1  if y in {0,1},
        s_i = y_i        if y in {-1,1},
        B_i = max{lam_i, log(2)}.

    Parameters
    ----------
    X : array_like, shape (n, p)
        Feature matrix.
    y : array_like, shape (n,)
        Labels, either in {0,1} or {-1,1}.
    lam : array_like, shape (n,)
        Contamination penalties lambda_i. Must be strictly positive.
    timelimit : float, optional
    threads : int, optional
    verbose : bool, optional

    Returns
    -------
    m : gurobipy.Model
        Built model. Variables are attached as:
            m._theta, m._t0, m._t1, m._u, m._z
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ImportError("This function requires gurobipy (Gurobi Python API).") from e

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    lam = np.asarray(lam, dtype=float).reshape(-1)

    n, p = X.shape
    s = 2 * y - 1

    # tau from log(1+exp(-tau)) = lambda
    tau = -np.log(np.expm1(lam))
    B = np.maximum(lam, np.log(2.0))
    U = np.linalg.norm(X, axis=1)*np.sqrt(n*p*np.log(2))
    L = -U

    m = gp.Model("core_logistic")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.FuncNonlinear = 1
    if threads is not None:
        m.Params.Threads = threads
    if timelimit is not None:
        m.Params.TimeLimit = timelimit

    theta = m.addVars(p, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")
    t0 = m.addVars(n, lb=-GRB.INFINITY, name="t0")
    t1 = m.addVars(n, lb=-GRB.INFINITY, name="t1")
    u = m.addVars(n, lb=0.0, name="u")
    z = m.addVars(n, vtype=GRB.BINARY, name="z")
    m._z = z
    m._u = u
    m._t0 = t0
    m._t1 = t1
    m._theta = theta
    q = m.addVars(n, lb=0.0, name="q")

    for i in range(n):
        m.addConstr(
            s[i]*gp.quicksum(X[i, j] * theta[j] for j in range(p)) == t0[i] + t1[i]
        )
        m.addConstr(t0[i] >= tau[i] * (1 - z[i]))
        m.addConstr(t0[i] <= U[i] * (1 - z[i]))
        m.addConstr(t1[i] <= tau[i] * z[i])
        m.addConstr(t1[i] >= L[i] * z[i])
        m.addConstr(q[i] == nlfunc.log(1 + nlfunc.exp(-t0[i])))
        m.addConstr(u[i] >= q[i] - B[i] * z[i])

    m.setObjective(
        gp.quicksum(u[i] + lam[i] * z[i] for i in range(n)) + gp.quicksum(theta[i]*theta[i] for i in range(p))/p,
        GRB.MINIMIZE
    )

    return m


args = parse_args()

n_list = parse_list(args.n_list, int)
tau_list = parse_list(args.tau_list, float)
timelimit = args.timelimit
THREADS = args.threads
BIG_M_INIT = float(args.big_m_init)

job_name = args.job_name
results = []

# Load dataset
data = load_breast_cancer()


BIG_M = 1
for n in n_list:
    # Features and target
    X = data.data[:n, :]
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    X = (X - X_mean) / X_std
    y = data.target[:n]
    for tau in tau_list:
        try:
            print([n, tau])
            lam = tau * np.ones(n) * np.log(n) / n
            # define a container to store the root node lower bound
            root_bound = [np.inf, -np.inf]

            # get tight big-M
            n, p = X.shape

            ## solve the problem in original formulation
            m = gp.Model()
            theta = m.addVars(p, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")
            w = m.addVars(n, lb=-1.0, ub=1.0, name="w")
            m._w = w
            z = m.addVars(n, vtype=GRB.BINARY, name="z")
            m._z = z
            # z = m.addVars(n, lb=0, ub=1, name="z")
            q = m.addVars(n, lb=0, ub=1, name="q")
            loss = m.addVars(n, lb=0.0, name="loss")

            for i in range(n):
                m.addConstr(-BIG_M * z[i] <= w[i])
                m.addConstr(w[i] <= BIG_M * z[i])
                m.addConstr(q[i] == nlfunc.logistic(gp.quicksum(X[i, j] * theta[j] for j in range(p))) + w[i])
                if y[i] == 1:
                    m.addConstr(loss[i] == -nlfunc.log(q[i]))
                else:
                    m.addConstr(loss[i] == -nlfunc.log(1 - q[i]))

            m.setObjective(
                gp.quicksum(loss[i] + lam[i] * z[i] for i in range(n)) + gp.quicksum(theta[i]*theta[i] for i in range(p))/p,
                GRB.MINIMIZE
            )

            m.params.OutputFlag = 1
            m.params.Threads = THREADS
            m.params.TimeLimit = timelimit
            m.Params.NodefileStart = 1
            m.optimize(record_root_lb)
            z_ori_vals = np.array([z[i].X for i in range(n)])
            result_opt = [n, tau, 'original', root_bound[0], root_bound[1],
                    (root_bound[0] - root_bound[1]) / root_bound[0], m.ObjVal, m.ObjBound,
                    (m.ObjVal - m.ObjBound) / (m.ObjVal+1e-6), np.count_nonzero(z_ori_vals), m.NodeCount,
                    m.runtime]
            results.append(result_opt)
            print('--------------------------------')
            print('solve the problem in original formulation')
            print(f"The obj is {m.objVal}.")
            print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100*(root_bound[0]-root_bound[1])/root_bound[0],4)}%. Runtime: {m.runtime}.")
            print(np.where(z_ori_vals == 1)[0])
            print(f"Number of outliers: {np.count_nonzero(z_ori_vals)}")
            print('--------------------------------')

            ## solve the optimal solution in the proposed formulation
            root_bound = [np.inf, -np.inf]

            model_opt = cor_reform(X, y, lam)
            # check presolved model
            # p = model_opt.presolve()
            # p.write('presolved_model.mps')
            model_opt.params.OutputFlag = 1
            # model_opt.params.PreMIQCPForm = 1
            model_opt.params.Threads = THREADS
            model_opt.params.TimeLimit = timelimit
            model_opt.optimize(record_root_lb)
            z_opt_vals = np.array([model_opt._z[i].X for i in range(n)])
            result_opt = [n, tau, 'opt', root_bound[0], root_bound[1],
                        (root_bound[0] - root_bound[1]) / root_bound[0], model_opt.ObjVal, model_opt.ObjBound,
                        (model_opt.ObjVal - model_opt.ObjBound) / (model_opt.ObjVal+1e-6), np.count_nonzero(z_opt_vals), model_opt.NodeCount,
                        model_opt.runtime]
            results.append(result_opt)
            print('--------------------------------------------------')
            print("Solve the optimal solution in the proposed formulation")
            print(f"The obj is {model_opt.objVal}.")
            print(
                f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100 * (root_bound[0] - root_bound[1]) / root_bound[0], 4)}%. Runtime: {model_opt.runtime}.")
            print(np.where(z_opt_vals == 1)[0])
            print(f"Number of outliers: {np.count_nonzero(z_opt_vals)}")
            print('--------------------------------------------------')

            results_df = pd.DataFrame(results, columns=['n', 'tau' ,'formulation','root_ub','root_lb','root_gap','end_ub','end_lb','end_gap','nnz','node_count','time'])
            print(results_df)
            results_df.to_csv(f"{current_dir}/../experiments_results/single_index_model_{job_name}.csv", index=False)
        except Exception as e:
            print(f"Error: {e}")
            continue

