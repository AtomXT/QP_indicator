import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import gurobipy as gp
import cvxpy as cp
import random
from gurobipy import GRB
import os

np.set_printoptions(linewidth=200)

from sklearn import datasets
from itertools import combinations
from src.rank2 import fast_dp_general


data = datasets.load_diabetes()
k = 3
m = k
n = 100
# Access features and target
X = data.data[0:n, 0:m]
X = (X - np.mean(X, axis=0))/X.std(axis=0)
y = data.target[:n]
y = (y-np.mean(y))/np.std(y)

G = X.T@X/2
D = np.eye(n)/2 + 0.5*np.eye(n)  # regularization
F = X/2

c = -y
d = - X.T@y

BIG_M = 1000
mu = 0.5
lam = mu*np.ones((n, 1))
np.linalg.eigvalsh(np.bmat([[D, F], [F.T, G]]))

## get the optimal solution
model_opt = gp.Model()
z_opt = model_opt.addMVar(n, vtype=GRB.BINARY, lb=0, ub=1, name='z')
x_opt = model_opt.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
y_opt = model_opt.addMVar(k, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
# add constraints
model_opt.addConstrs(x_opt[i]*(1-z_opt[i]) == 0 for i in range(n))

# set objective
eqn = y.T@y/2 + y_opt.T@G@y_opt + x_opt.T@D@x_opt + x_opt.T@F@y_opt + c.T@x_opt + d.T@y_opt + lam.T@z_opt
model_opt.setObjective(eqn[0], GRB.MINIMIZE)
# model_opt.params.QCPDual = 1
model_opt.params.OutputFlag = 1
model_opt.params.TimeLimit = 3
model_opt.optimize()
print(f"The obj is {model_opt.objVal}.")


# extract solutions
z_opt_vals = np.array([z_opt[i].X for i in range(n)])
y_opt_vals = np.array([y_opt[i].X for i in range(k)])
x_opt_vals = np.array([x_opt[i].X for i in range(n)])

# get tight big-M
model_relax = gp.Model()
z_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z')
x_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
y_relax = model_relax.addMVar(k, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
# add constraints
model_relax.addConstrs(x_relax[i] <= BIG_M*z_relax[i] for i in range(n))
model_relax.addConstrs(x_relax[i] >= -BIG_M*z_relax[i] for i in range(n))

# set objective
eqn = y.T@y/2 + y_relax.T@G@y_relax + x_relax.T@D@x_relax + x_relax.T@F@y_relax + c.T@x_relax + d.T@y_relax + lam.T@z_relax
model_relax.setObjective(eqn[0], GRB.MINIMIZE)
# model_relax.params.QCPDual = 1
model_relax.params.OutputFlag = 0
model_relax.optimize()
print(f"The relaxed obj is {model_relax.objVal}.")
x_relax_vals = np.array([x_relax[i].X for i in range(n)])

print(f"The larges value of x is {max(x_relax_vals)}")
BIG_M = min(BIG_M, 2*max(abs(x_relax_vals)))
print(f'Use new Big-M {BIG_M}.')

# get dual variables
model_dul = gp.Model()
z_dul = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z')
# z_dul = model_dul.addMVar(n, vtype=GRB.BINARY, name='z')
y_dul = model_dul.addMVar(k, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
x_dul = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
z_dul_bar = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z_bar')
# z_dul_bar = model_dul.addMVar(n, vtype=GRB.BINARY, name='z_bar')
y_dul_bar = model_dul.addMVar(k, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y_bar')
x_dul_bar = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x_bar')
# w_dul_tilde = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='w_tilde')
# t_dul = model_dul.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")

# add constraints
model_dul.addConstrs(x_dul[i] <= BIG_M*z_dul[i] for i in range(n))
model_dul.addConstrs(x_dul[i] >= -BIG_M*z_dul[i] for i in range(n))
z_equal = model_dul.addConstrs(z_dul[i] == z_dul_bar[i] for i in range(n))
y_equal = model_dul.addConstrs(y_dul[i] == y_dul_bar[i] for i in range(k))
x_equal = model_dul.addConstrs(x_dul[i] == x_dul_bar[i] for i in range(n))

# set objective
# model_dul.setObjective(t_dul, GRB.MINIMIZE)
eqn = y.T@y/2 + y_dul_bar.T@G@y_dul_bar + x_dul_bar.T@D@x_dul_bar + x_dul_bar.T@F@y_dul_bar + c.T@x_dul_bar + d.T@y_dul_bar + lam.T@z_dul_bar
model_dul.setObjective(eqn[0], GRB.MINIMIZE)
model_dul.params.OutputFlag = 0
model_dul.params.QCPDual = 1
model_dul.optimize()

alpha = np.array([z_equal[i].Pi for i in range(n)])
beta = np.array([x_equal[i].Pi for i in range(n)])
gamma = np.array([y_equal[i].Pi for i in range(k)])
combined = []
combined.append(np.concatenate([alpha, beta, gamma]))
print(alpha, beta, gamma)

import random
s = 1
index_pair = [list(t) for t in combinations(range(m), 2)]
pairs = random.sample(index_pair, s)
# pairs = [[3, 4]]

Di = [cp.diag(cp.Variable(n)) for i in range(s)]
Fi = [cp.Variable((n, m)) for i in range(s)]
Gi = [cp.Variable((m, m)) for i in range(s)]
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


# Q = [[cp.Variable((m, m)) for ii in range(n)] for i in range(s)]
# Qj = [[cp.Variable((n, m)) for ii in range(n)] for i in range(s)]
# I = np.eye(n)

# constraints
constraint_0 = [cp.bmat([[D - cp.sum(Di), F - cp.sum(Fi)], [(F - cp.sum(Fi)).T, G - cp.sum(Gi)]]) >> 0]
# constraint_0 = []
for i in range(s):
    constraint_0.append(cp.bmat([[Di[i], Fi[i]], [Fi[i].T, Gi[i]]]) >> 0)
    constraint_0.append(cp.multiply(Gi[i], Gi_mask[i]) == 0)
    constraint_0.append(cp.multiply(Fi[i], Fi_mask[i]) == 0)

# objective = cp.Minimize(cp.norm_inf(D - cp.sum(Di))+cp.norm_inf(G - cp.sum(Gi)))
# objective = cp.Minimize(cp.norm_inf(G - cp.sum(Gi)))
# objective = cp.Minimize(cp.norm(D - cp.sum(Di), 'nuc')+cp.norm(G - cp.sum(Gi), 'nuc'))
# objective = cp.Minimize(cp.sum(cp.diag(-cp.sum(Di))) + cp.sum(cp.diag(-cp.sum(Gi))))
objective = cp.Minimize(0.4*cp.norm(Fi[0], 1)+cp.lambda_max(cp.bmat([[D - cp.sum(Di), F - cp.sum(Fi)], [(F - cp.sum(Fi)).T, G - cp.sum(Gi)]])))

# Formulate the optimization problem
problem = cp.Problem(objective, constraint_0)

# Solve the problem using MOSEK
problem.solve(solver=cp.MOSEK)
print(f"The solving time is {problem.solver_stats.solve_time}.")

# Check the results
if problem.status == cp.OPTIMAL:
    print("Optimal value:", problem.value)
    print(f"Maximum eigenvalue of D: {np.max(np.linalg.eigvals(D))}")
    print(f"Maximum eigenvalue of G: {np.max(np.linalg.eigvals(G))}")
    print(f"Maximum eigenvalue of D-\sum_i D_i: {np.max(np.linalg.eigvals(D - cp.sum(Di).value))}")
    print(f"Maximum eigenvalue of G-\sum_i G_i: {np.max(np.linalg.eigvals(G - cp.sum(Gi).value))}")
    print(f"Minimum eigenvalue of G-\sum_i G_i: {np.min(np.linalg.eigvals(G - cp.sum(Gi).value))}")
    print(f"Maximum eigenvalue of unstructured problem: {np.max(np.linalg.eigvalsh(np.block([[D - cp.sum(Di).value, F - cp.sum(Fi).value], [(F - cp.sum(Fi).value).T, G - cp.sum(Gi).value]])))}")
    print(f"Minimum eigenvalue of unstructured problem: {np.min(np.linalg.eigvalsh(np.block([[D - cp.sum(Di).value, F - cp.sum(Fi).value], [(F - cp.sum(Fi).value).T, G - cp.sum(Gi).value]])))}")
    print(np.linalg.eigvalsh(np.block([[D - cp.sum(Di).value, F - cp.sum(Fi).value], [(F - cp.sum(Fi).value).T, G - cp.sum(Gi).value]])))
else:
    print("Problem not solved to optimality. Status:", problem.status)

Gi_ = [np.where(np.abs(Gi[ii].value) < 1e-6, 0, Gi[ii].value) for ii in range(len(pairs))]
Fi_ = [np.where(np.abs(Fi[ii].value) < 1e-6, 0, Fi[ii].value) for ii in range(len(pairs))]
Di_ = [np.where(np.abs(Di[ii].value) < 1e-6, 0, Di[ii].value) for ii in range(len(pairs))]
# Di_ = [Di[ii].value for ii in range(len(pairs))]
Gi_sum_diff_, Di_sum_diff_, Fi_sum_diff_ = G - cp.sum(Gi).value, D - cp.sum(Di).value, F - cp.sum(Fi).value
Gi_sum_diff_[np.abs(Gi_sum_diff_) < 1e-6] = 0
Fi_sum_diff_[np.abs(Fi_sum_diff_) < 1e-6] = 0
Di_sum_diff_[np.abs(Di_sum_diff_) < 1e-6] = 0



for iii in range(2):
    print(f"adding the {iii + 1}th cut.")

    # get dual variables
    model_dul = gp.Model()
    z_dul = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z')
    y_dul = model_dul.addMVar(k, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
    x_dul = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
    z_dul_bar = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z_bar')
    y_dul_bar = model_dul.addMVar(k, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y_bar')
    x_dul_bar = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x_bar')
    t_dul = model_dul.addMVar(len(pairs), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")

    # add constraints
    model_dul.addConstrs(x_dul[i] <= BIG_M * z_dul[i] for i in range(n))
    model_dul.addConstrs(x_dul[i] >= -BIG_M * z_dul[i] for i in range(n))
    z_equal = model_dul.addConstrs(z_dul[i] == z_dul_bar[i] for i in range(n))
    y_equal = model_dul.addConstrs(y_dul[i] == y_dul_bar[i] for i in range(k))
    x_equal = model_dul.addConstrs(x_dul[i] == x_dul_bar[i] for i in range(n))

    psi_values = []
    for ii, pair in enumerate(pairs):
        print(f"- adding {ii + 1}th pair.")
        _, _, _, f_dp = fast_dp_general(Gi_[ii][np.ix_(pair, pair)], Di_[ii], Fi_[ii][:, pair], -beta,
                                        -gamma[pair], -alpha.reshape(-1, 1))
        psi_v = f_dp
        psi_values.append(psi_v)
        # print(psi_v)
        model_dul.addConstr(
            t_dul[ii] >= y_dul_bar[pair].T @ Gi_[ii][np.ix_(pair, pair)] @ y_dul_bar[pair] + x_dul_bar.T @ Di_[
                ii] @ x_dul_bar + x_dul_bar.T @ Fi_[ii][:, pair] @ y_dul_bar[pair])

        model_dul.addConstr(t_dul[ii] >= alpha.T @ z_dul + beta.T @ x_dul + gamma[pair].T @ y_dul[pair] + psi_v)

    extra_term = y.T @ y / 2 + y_dul_bar.T @ Gi_sum_diff_ @ y_dul_bar + x_dul_bar.T @ Di_sum_diff_ @ x_dul_bar + x_dul_bar.T @ Fi_sum_diff_ @ y_dul_bar + c.T @ x_dul_bar + d.T @ y_dul_bar + lam.T @ z_dul_bar
    # extra_term = y.T@y/2 + c.T@x_dul_bar + d.T@y_dul_bar + lam.T@z_dul_bar
    # # set objective
    model_dul.setObjective(gp.quicksum(t_dul) + extra_term[0], GRB.MINIMIZE)
    model_dul.params.OutputFlag = 0
    model_dul.params.QCPDual = 1
    model_dul.optimize()

    alpha = np.array([z_equal[i].Pi for i in range(n)])
    beta = np.array([x_equal[i].Pi for i in range(n)])
    gamma = np.array([y_equal[i].Pi for i in range(k)])
    combined.append(np.concatenate([alpha, beta, gamma]))
    # model_dul.setObjective(t_dul+extra_term[0], GRB.MINIMIZE)
    # model_dul.params.OutputFlag = 0
    # model_dul.params.QCPDual = 1
    # model_dul.update()
    # model_dul.optimize()
    print(model_dul.objVal)

z_dul_val = np.squeeze([zi.X for zi in z_dul_bar])
thr = np.quantile(z_dul_val,0.9)
print(np.array([1.0 if v>thr else 0.0 for v in z_dul_val]))
print(np.abs(z_opt_vals))


## cut and branch

# define a container to store the root node lower bound
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

# get dual variables
model_dul = gp.Model()
z_dul = model_dul.addMVar(n, vtype=GRB.BINARY, name='z')
y_dul = model_dul.addMVar(k, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
x_dul = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
t_dul = model_dul.addMVar(len(pairs), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")

# add constraints
model_dul.addConstrs(x_dul[i] <= BIG_M * z_dul[i] for i in range(n))
model_dul.addConstrs(x_dul[i] >= -BIG_M * z_dul[i] for i in range(n))

for ii, pair in enumerate(pairs):
    print(f"- adding {ii + 1}th pair.")
    model_dul.addConstr(
        t_dul[ii] >= y_dul[pair].T @ Gi_[ii][np.ix_(pair, pair)] @ y_dul[pair] + x_dul.T @ Di_[
            ii] @ x_dul + x_dul.T @ Fi_[ii][:, pair] @ y_dul[pair])

    model_dul.addConstr(t_dul[ii] >= alpha.T @ z_dul + beta.T @ x_dul + gamma[pair].T @ y_dul[pair] + psi_values[ii])

extra_term = y.T @ y / 2 + y_dul.T @ Gi_sum_diff_ @ y_dul + x_dul.T @ Di_sum_diff_ @ x_dul + x_dul.T @ Fi_sum_diff_ @ y_dul + c.T @ x_dul + d.T @ y_dul + lam.T @ z_dul
# # set objective
model_dul.setObjective(gp.quicksum(t_dul) + extra_term[0], GRB.MINIMIZE)
model_dul.params.OutputFlag = 1
model_dul.params.TimeLimit = 30
# model_dul.setParam("NodeLimit", 2)
model_dul.optimize(record_root_lb)
print(model_dul.objVal)
print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100*(root_bound[0]-root_bound[1])/root_bound[0],4)}%")

z_dul_val = np.squeeze([zi.X for zi in z_dul])
thr = np.quantile(z_dul_val,0.9)
print(np.array([1.0 if v>thr else 0.0 for v in z_dul_val]))
print(np.abs(z_opt_vals))