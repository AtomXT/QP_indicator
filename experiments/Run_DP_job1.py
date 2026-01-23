
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
m_list = [2]
# m = 3
# n = 90
# Access features and target
timelimit = 300
# data_list = ['diabetes', 'autompg']
data_list = ['diabetes']
results = []
for dataset in data_list:
    X_, y_ = get_data(dataset)
    for m in m_list:
        for n in n_list:
            try:
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
                extra_obj = y.T @ y / 2

                BIG_M = 1000
                mu = 1
                lam = mu*np.ones((n, 1))
                print(np.linalg.eigvalsh(np.bmat([[D, F], [F.T, G]]))[0:5])
                start = time.time()
                x, y, z, f = fast_dp_general(G/2, D/2, F, c, d, lam)
                end = time.time()
                f = f[0][0] + extra_obj
                result_opt = [m, n, dataset, 'dp', '-', '-', '-', f, f, 0, '-',
                              end - start]
                results.append(result_opt)


                print('--------------------------------------------------')
                print("Solve the optimal solution in DP")
                print(f"The obj is {f}.")
                print('--------------------------------------------------')

                results_df = pd.DataFrame(results, columns=['m','n','dataset','formulation','root_ub','root_lb','root_gap','end_ub','end_lb','end_gap','node_count','time'])
                print(results_df)
                results_df.to_csv(f"{current_dir}/../experiments_results/DP_results_job1.csv")
            except:
                continue


