
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

# n_list = [50, 60, 70, 80, 90, 100, 120, 150, 200]
n_list = [50, 80, 100, 120, 150, 200]
# m_list = [2, 3, 4]
m_list = [10]
# m = 3
# n = 90
# Access features and target
timelimit = 600
# data_list = ['diabetes', 'autompg']
data_list = ['autompg']
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
                          model_opt.runtime]
            results.append(result_opt)
            print('--------------------------------------------------')
            print("Solve the optimal solution in the proposed formulation")
            print(f"The obj is {model_opt.objVal}.")
            print(
                f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100 * (root_bound[0] - root_bound[1]) / root_bound[0], 4)}%. Runtime: {model_opt.runtime}.")
            print('--------------------------------------------------')

            # solve the optimal solution in the proposed formulation with closed form x
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

            model_opt.addConstrs(x_opt[i] + (F[i, :]@y_opt + c[i])/D[i, i] <= 1000*(1 - z_opt[i]) for i in range(n))
            model_opt.addConstrs(-x_opt[i] - (F[i, :] @ y_opt + c[i]) / D[i, i] <= 1000 * (1 - z_opt[i]) for i in range(n))

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
                          model_opt.runtime]
            results.append(result_opt)


            print('--------------------------------------------------')
            print("Solve the optimal solution in the new proposed formulation")
            print(f"The obj is {model_opt.objVal}.")
            print(
                f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100 * (root_bound[0] - root_bound[1]) / root_bound[0], 4)}%. Runtime: {model_opt.runtime}.")
            print('--------------------------------------------------')

            results_df = pd.DataFrame(results, columns=['m','n','dataset','formulation','root_ub','root_lb','root_gap','end_ub','end_lb','end_gap','node_count','time'])
            print(results_df)
            results_df.to_csv(f"{current_dir}/../experiments_results/new_opt_formulation_large_m_results_job2.csv")
        # except:
        #     continue


