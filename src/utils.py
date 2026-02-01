import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn import datasets
from gurobipy import GRB
from ucimlrepo import fetch_ucirepo
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def decomposition(D, G, F, pairs, s=1):
    n, m = F.shape
    Di = [cp.diag(cp.Variable(n, nonneg=True)) for i in range(s)]
    Fi = [cp.Variable((n, m)) for i in range(s)]
    Gi = [cp.Variable((m, m), PSD=True) for i in range(s)]
    Fi_mask = []
    Gi_mask = []
    for i in range(s):
        ii, jj = pairs[i]

        fi_mask = np.ones((n, m))
        fi_mask[:, [ii, jj]] = 0
        Fi_mask.append(fi_mask)

        gi_mask = np.ones((m, m))
        gi_mask[ii, [ii, jj]] = 0
        gi_mask[jj, [ii, jj]] = 0
        Gi_mask.append(gi_mask)

    # constraints
    constraint_0 = [cp.bmat([[D - cp.sum(Di), F - cp.sum(Fi)], [(F - cp.sum(Fi)).T, G - cp.sum(Gi)]]) >> 0]
    for i in range(s):
        constraint_0.append(cp.bmat([[Di[i], Fi[i]], [Fi[i].T, Gi[i]]]) >> 0)
        constraint_0.append(cp.multiply(Gi[i], Gi_mask[i]) == 0)
        constraint_0.append(cp.multiply(Fi[i], Fi_mask[i]) == 0)
    obj_expr = 0
    for ii in range(s):
        obj_expr += cp.sum(cp.diag(Di[ii]))
    objective = cp.Maximize(obj_expr)

    # Formulate the optimization problem
    problem = cp.Problem(objective, constraint_0)

    # Solve the problem using MOSEK
    problem.solve(solver=cp.MOSEK)
    print(f"The decomposition time is {problem.solver_stats.solve_time}.")

    # Check the results
    if problem.status == cp.OPTIMAL:
        print("Optimal value:", problem.value)
    else:
        print("Problem not solved to optimality. Status:", problem.status)

    Gi_ = [np.where(np.abs(Gi[ii].value) < 1e-8, 0, Gi[ii].value) for ii in range(len(pairs))]
    Fi_ = [np.where(np.abs(Fi[ii].value) < 1e-8, 0, Fi[ii].value) for ii in range(len(pairs))]
    Di_ = [np.where(Di[ii].value < 1e-8, 0, Di[ii].value) for ii in range(len(pairs))]

    Gi_sum_diff_, Di_sum_diff_, Fi_sum_diff_ = G - cp.sum(Gi_), D - cp.sum(Di_), F - cp.sum(Fi_)
    return Di_, Gi_, Fi_, Di_sum_diff_, Gi_sum_diff_, Fi_sum_diff_


def get_data(name):
    if name == 'diabetes':
        data = datasets.load_diabetes()
        X = data.data
        y = data.target
    elif name == 'autompg':
        auto_mpg = fetch_ucirepo(id=9)
        # 9 is auto_mpg 87 is servo
        # data (as pandas dataframes)
        X = auto_mpg.data.features.dropna().values
        X = X[:, [4, 5, 6, 0, 1, 2, 3]]
        y = auto_mpg.data.targets.values.reshape(-1)
    else:
        X, y = None, None
        print('Unknown dataset')
    return X, y

def get_data_offline(name):
    if name == 'diabetes':
        data = np.load(f"{project_root}/data/diabetes.npz")
        X = data["X"]
        y = data["y"]
    elif name == 'autompg':
        data = np.load(f"{project_root}/data/autompg.npz")
        X = data["X"]
        y = data["y"]
    elif name == 'crime':
        data = pd.read_csv(f"{project_root}/data/Crime.csv")
        X = data.iloc[:, :-1].values
        y = data.iloc[:,-1].values
    else:
        X, y = None, None
        print('Unknown dataset')
    return X, y


def pairwise_infeasible(F, a, D, lam, i, j, s, t, tol=1e-9):
    """
    Check infeasibility of (z_i^s, z_j^t) = (1,1).

    Parameters
    ----------
    F : ndarray (n, m)
    a : ndarray (n,)
    D : ndarray (n,)     # diagonal entries D_ii
    lam : ndarray (n,)
    i, j : int           # indices, i != j
    s, t : str           # '+' or '-'
    tol : float          # numerical tolerance

    Returns
    -------
    infeasible : bool
    """

    theta_i = np.sqrt(2 * D[i] * lam[i])
    theta_j = np.sqrt(2 * D[j] * lam[j])

    A = []
    b = []

    # constraint for i
    if s == '+':
        # F_i y >= theta_i - a_i  ->  -F_i y <= -(theta_i - a_i)
        A.append(-F[i])
        b.append(-(theta_i - a[i]))
    elif s == '-':
        # F_i y <= -theta_i - a_i
        A.append(F[i])
        b.append(-theta_i - a[i])
    else:
        raise ValueError("s must be '+' or '-'")

    # constraint for j
    if t == '+':
        A.append(-F[j])
        b.append(-(theta_j - a[j]))
    elif t == '-':
        A.append(F[j])
        b.append(-theta_j - a[j])
    else:
        raise ValueError("t must be '+' or '-'")

    A = np.vstack(A)
    b = np.array(b)

    m = F.shape[1]

    # dummy objective (we only care about feasibility)
    res = linprog(
        c=np.zeros(m),
        A_ub=A,
        b_ub=b,
        bounds=[(-100, 100)] * m,
        method='highs'
    )

    return not res.success
