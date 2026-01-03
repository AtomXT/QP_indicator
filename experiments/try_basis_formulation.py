
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import gurobipy as gp
import cvxpy as cp
import random
from gurobipy import GRB
from src.utils import decomposition, get_data, pairwise_infeasible
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

np.set_printoptions(linewidth=200)


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

def solve_miqp_basis_encoding(n, m, D_diag, G, F, a, b, lam, M=1e3):
    """
    Solve the MIQP using basis encoding formulation.

    Parameters:
    -----------
    n : int
        Number of x variables
    m : int
        Number of y variables (small)
    D_diag : np.array (n,)
        Diagonal elements of D
    G : np.array (m, m)
        Matrix G
    F : np.array (n, m)
        Matrix F
    a : np.array (n,)
        Vector a
    b : np.array (m,)
        Vector b
    lam : np.array (n,)
        Vector λ
    M : float
        Big-M constant

    Returns:
    --------
    dict or None
        Solution if found, None otherwise
    """

    # Create model
    model = gp.Model("MIQP_Basis_Encoding")

    # ====================
    # Variables
    # ====================

    # Basis variables: t_j for j=1..2n
    t = model.addVars(2*n, vtype=gp.GRB.BINARY, name="t")
    # Sign variables: s_j^+, s_j^- for j=1..2n
    s_plus = model.addVars(2*n, vtype=gp.GRB.BINARY, name="s_plus")
    s_minus = model.addVars(2*n, vtype=gp.GRB.BINARY, name="s_minus")
    # Continuous variables
    y_opt = model.addMVar(m, lb=-gp.GRB.INFINITY, name="y")
    x_opt = model.addMVar(n, lb=-gp.GRB.INFINITY, name="x")
    # Original z variables
    z_opt = model.addMVar(n, vtype=gp.GRB.BINARY, name="z")
    # z_plus = model.addMVar(n, vtype=GRB.BINARY, name='z+')
    # z_minus = model.addMVar(n, vtype=GRB.BINARY, name='z-')


    # ====================
    # Precompute hyperplane parameters
    # ====================

    # Precompute h_j (normals) and c_j (RHS)
    h = np.zeros((2*n, m))
    c = np.zeros(2*n)

    for i in range(n):
        # Positive hyperplane (j = 2i): F_i y + a_i = sqrt(2D_ii λ_i)
        h[2*i, :] = F[i, :]
        c[2*i] = np.sqrt(2 * D_diag[i] * lam[i]) - a[i]

        # Negative hyperplane (j = 2i+1): F_i y + a_i = -sqrt(2D_ii λ_i)
        h[2* i + 1, :] = F[i, :]
        c[2* i + 1] = -np.sqrt(2 * D_diag[i] * lam[i]) - a[i]

    # ====================
    # Constraints
    # ====================

    # 1. Basis size: at most m hyperplanes active
    model.addConstr(gp.quicksum(t[j] for j in range(2*n)) <= m, "basis_size")
    # 2. Sign consistency constraints
    # s_j^+ + s_j^- ≤ 1
    # model.addConstrs(s_plus[j] + s_minus[j] <= 1 for j in range(2*n))
    # If t_j=1, then s_j^+ = s_j^- = 0
    # model.addConstrs(s_plus[j] <= 1 - t[j] for j in range(2*n))
    # model.addConstrs(s_minus[j] <= 1 - t[j] for j in range(2*n))
    # Exactly one of s_j^+ or s_j^- is 1 if t_j=0
    model.addConstrs(s_plus[j] + s_minus[j] == 1 - t[j] for j in range(2*n))
    # 3. Hyperplane constraints
    # If s_j^+ = 1, then h_j^T y ≥ c_j
    model.addConstrs(h[j, :] @ y_opt >= c[j] - M * (1 - s_plus[j]) for j in range(2*n))
    # If s_j^- = 1, then h_j^T y ≤ c_j
    model.addConstrs(h[j, :]@y_opt <= c[j] + M * (1 - s_minus[j]) for j in range(2*n))
    # If t_j = 1, then h_j^T y = c_j
    model.addConstrs(h[j, :] @ y_opt <= c[j] + M * (1 - t[j]) for j in range(2 * n))
    model.addConstrs(h[j, :] @ y_opt >= c[j] - M * (1 - t[j]) for j in range(2 * n))

    # 4. Complementarity and optimality for x
    for i in range(n):
        # z_i = s_{2i}^+ + s_{2i+1}^- (positive and negative hyperplanes for index i)
        model.addConstr(z_opt[i] == s_plus[ 2 *i] + s_minus[ 2 * i +1], f"z_def_{i}")

        # D_ii x_i + F_i y + a_i = 0 if z_i = 1, otherwise bounded by M
        expr = D_diag[i] * x_opt[i] + F[i, :] @ y_opt + a[i]
        model.addConstr(expr <= M * (1 - z_opt[i]), f"compl_upper_{i}")
        model.addConstr(expr >= -M * (1 - z_opt[i]), f"compl_lower_{i}")

        # Bound x_i when z_i = 0
        model.addConstr(x_opt[i] <= M * z_opt[i], f"x_upper_{i}")
        model.addConstr(x_opt[i] >= -M * z_opt[i], f"x_lower_{i}")

    # 5. Gradient condition: G y + b + F^T x = 0
        model.addConstr(G@y_opt + F.T@x_opt == -b, f"gradient")

        # # add constraints
        # model.addConstrs(z_opt[i] == z_plus[i] + z_minus[i] for i in range(n))
        # model.addConstrs(-np.sqrt(2 * lam[i] * D[i, i]) - 1000 * z_opt[i] <= gp.quicksum(
        #     F[i, j] * y_opt[j] for j in range(m)) + c[i] for i in range(n))
        # model.addConstrs(np.sqrt(2 * lam[i] * D[i, i]) + 1000 * z_opt[i] >= gp.quicksum(
        #     F[i, j] * y_opt[j] for j in range(m)) + c[i] for i in range(n))
        # model.addConstrs(-np.sqrt(2 * lam[i] * D[i, i]) + 1000 * (1 - z_minus[i]) >= gp.quicksum(
        #     F[i, j] * y_opt[j] for j in range(m)) + c[i] for i in range(n))
        # model.addConstrs(np.sqrt(2 * lam[i] * D[i, i]) - 1000 * (1 - z_plus[i]) <= gp.quicksum(
        #     F[i, j] * y_opt[j] for j in range(m)) + c[i] for i in range(n))
    # ====================
    # Objective
    # ====================

    model.setObjective(y.T @ y / 2 + y_opt.T @ G @ y_opt / 2 + x_opt.T @ D @ x_opt / 2 + x_opt.T @ F @ y_opt + a.T @ x_opt + b.T @ y_opt + lam.T @ z_opt,
                       gp.GRB.MINIMIZE)

    # ====================
    # Solver parameters
    # ====================

    # Enable non-convex quadratic solving
    # model.Params.NonConvex = 2

    # Time limit (optional)
    model.Params.TimeLimit = 20  # 10 minutes

    # Display progress
    model.Params.OutputFlag = 1

    # ====================
    # Solve and return
    # ====================

    model.optimize()

    if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
        # Extract solution
        sol_x = np.array([x_opt[i].X for i in range(n)])
        sol_y = np.array([y_opt[k].X for k in range(m)])
        sol_z = np.array([z_opt[i].X for i in range(n)])
        sol_obj = model.ObjVal

        # Print some statistics
        active_basis = [j for j in range(2*n) if t[j].X > 0.5]
        print(f"\nSolution found with objective: {sol_obj:.6f}")
        print(f"Number of active hyperplanes: {len(active_basis)}")
        print(f"Active hyperplanes: {active_basis}")

        # Return solution
        return {
            'x': sol_x,
            'y': sol_y,
            'z': sol_z,
            'obj': sol_obj,
            'model': model,
            'status': model.status
        }
    else:
        print(f"Optimization failed with status: {model.status}")
        return None



# n_list = [50, 60, 70, 80, 90, 100, 120, 150, 200]
n_list = [200]
# m_list = [2, 3, 4]
m_list = [3]
# m = 3
# n = 90
# Access features and target
timelimit = 30
# data_list = ['diabetes', 'autompg']
data_list = ['diabetes']
results = []
for dataset in data_list:
    X_, y_ = get_data(dataset)
    for m in m_list:
        for n in n_list:
        # try:
            print([m, n, dataset])
            X, y = X_[:n, :m], y_[:n]
            X = (X - np.mean(X, axis=0)) / X.std(axis=0)
            y = (y - np.mean(y)) / np.std(y)
            mu_g = 0.2
            G = X.T@X + 2*mu_g*np.eye(m)  # regularization
            D = np.eye(n)
            F = X

            c = -y
            d = - X.T@y

            BIG_M = 1000
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

            print(f"The larges value of x is {max(x_relax_vals)}")
            BIG_M = min(BIG_M, 2*max(abs(x_relax_vals)))
            print(f'Use new Big-M {BIG_M}.')

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
            model_ori.optimize(record_root_lb)
            z_opt_vals = np.array([z_ori[i].X for i in range(n)])
            result_opt = [m, n, dataset, 'original', root_bound[0], root_bound[1],
                      (root_bound[0] - root_bound[1]) / root_bound[0], model_ori.ObjVal, model_ori.ObjBound,
                      (model_ori.ObjVal - model_ori.ObjBound) / model_ori.ObjVal, model_ori.NodeCount,
                      model_ori.runtime, 0]
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

            model_opt = gp.Model()
            z_opt = model_opt.addMVar(n, vtype=GRB.BINARY, name='z')
            z_plus = model_opt.addMVar(n, vtype=GRB.BINARY, name='z+')
            z_minus = model_opt.addMVar(n, vtype=GRB.BINARY, name='z-')
            y_opt = model_opt.addMVar(m, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
            x_opt = model_opt.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')

            # add constraints
            model_opt.addConstrs(x_opt[i] <= BIG_M * z_opt[i] for i in range(n))
            model_opt.addConstrs(x_opt[i] >= -BIG_M * z_opt[i] for i in range(n))
            model_opt.addConstrs(z_opt[i] == z_plus[i] + z_minus[i] for i in range(n))
            model_opt.addConstrs(-np.sqrt(2 * lam[i] * D[i, i]) - 1000 * z_opt[i] <= gp.quicksum(
                F[i, j] * y_opt[j] for j in range(m)) + c[i] for i in range(n))
            model_opt.addConstrs(np.sqrt(2 * lam[i] * D[i, i]) + 1000 * z_opt[i] >= gp.quicksum(
                F[i, j] * y_opt[j] for j in range(m)) + c[i] for i in range(n))
            model_opt.addConstrs(-np.sqrt(2 * lam[i] * D[i, i]) + 1000 * (1 - z_minus[i]) >= gp.quicksum(
                F[i, j] * y_opt[j] for j in range(m)) + c[i] for i in range(n))
            model_opt.addConstrs(np.sqrt(2 * lam[i] * D[i, i]) - 1000 * (1 - z_plus[i]) <= gp.quicksum(
                F[i, j] * y_opt[j] for j in range(m)) + c[i] for i in range(n))


            obj = y.T @ y / 2 + y_opt.T @ G @ y_opt / 2 + x_opt.T @ D @ x_opt / 2 + x_opt.T @ F @ y_opt + c.T @ x_opt + d.T @ y_opt + lam.T @ z_opt
            model_opt.setObjective(obj[0], GRB.MINIMIZE)
            model_opt.params.OutputFlag = 1
            # model_opt.params.PreMIQCPForm = 1
            model_opt.params.Threads = 8
            model_opt.params.TimeLimit = timelimit
            model_opt.optimize(record_root_lb)
            result_opt = [m, n, dataset, 'opt', root_bound[0], root_bound[1],
                          (root_bound[0] - root_bound[1]) / root_bound[0], model_opt.ObjVal, model_opt.ObjBound,
                          (model_opt.ObjVal - model_opt.ObjBound) / model_opt.ObjVal, model_opt.NodeCount,
                          model_opt.runtime, 0]
            results.append(result_opt)
            print('--------------------------------------------------')
            print("Solve the optimal solution in the proposed formulation")
            print(f"The obj is {model_opt.objVal}.")
            print(
                f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100 * (root_bound[0] - root_bound[1]) / root_bound[0], 4)}%. Runtime: {model_opt.runtime}.")
            print('--------------------------------------------------')

            # solve the optimal solution in the proposed formulation with infeasible pairs
            results_basis = solve_miqp_basis_encoding(n, m, np.diag(D), G, F, c, d, lam.squeeze(), M=10)
            model_opt = results_basis['model']

            result_opt = [m, n, dataset, 'opt_feasibility', 'NA', 'NA',
                             'NA', model_opt.ObjVal, model_opt.ObjBound,
                             (model_opt.ObjVal - model_opt.ObjBound) / model_opt.ObjVal, model_opt.NodeCount,
                             model_opt.runtime, 0]
            results.append(result_opt)

            print('--------------------------------------------------')
            print("Solve the optimal solution in the proposed formulation")
            print(f"The obj is {model_opt.objVal}.")
            print(
                f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100 * (root_bound[0] - root_bound[1]) / root_bound[0], 4)}%. Runtime: {model_opt.runtime}.")
            print('--------------------------------------------------')

            results_df = pd.DataFrame(results, columns=['m','n','dataset','formulation','root_ub','root_lb','root_gap','end_ub','end_lb','end_gap','node_count','time','find_time'])
            print(results_df)
            results_df.to_csv(f"{current_dir}/../experiments_results/basis_results.csv")
        # except:
        #     continue


