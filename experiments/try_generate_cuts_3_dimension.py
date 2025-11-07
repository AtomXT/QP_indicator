import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import os
from sklearn import datasets
from itertools import combinations
from src.rank2 import fast_dp_general



np.set_printoptions(linewidth=200)

os.chdir('//')

data = datasets.load_diabetes()
k = 3
m = k
n = 100
# Access features and target
X = data.data[0:n, 0:m]
X = (X - np.mean(X, axis=0))/X.std(axis=0)
y = data.target[:n]
y = (y-np.mean(y))/np.std(y)

G = X.T@X/2 + 0.5*np.eye(n) # regularization
D = np.eye(n)/2
F = X/2

c = -y
d = - X.T@y

BIG_M = 1000
mu = 0.5
lam = mu*np.ones((n, 1))

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
model_opt.params.TimeLimit = 10
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
# print(alpha, beta, gamma)

s = 1
chosen_pairs = (0, 1)

D0 = cp.diag(cp.Variable(n))
F0 = cp.Variable((n, m))
G0 = cp.Variable((m, m))

D = np.eye(n)/2 + np.eye(n)*0.5  # regularization
G = X.T@X
F = X/2

ii, jj = chosen_pairs

fi_mask = np.ones((n, m))
fi_mask[:, [ii, jj]] = 0

gi_mask = np.ones((m, m))
gi_mask[ii, [ii, jj]] = 0
gi_mask[jj, [ii, jj]] = 0

## constraints
constraints = []

constraints.append(cp.bmat([[D - D0, F - F0], [(F - F0).T, G - G0]]) >> 0)
constraints.append(cp.bmat([[D0, F0], [F0.T, G0]]) >> 0)

# mask
constraints.append(cp.multiply(G0, gi_mask) == 0)
constraints.append(cp.multiply(F0, fi_mask) == 0)

objective = cp.Minimize(cp.lambda_max(G-G0))

# Formulate the optimization problem
problem = cp.Problem(objective, constraints)

# Solve the problem using MOSEK
problem.solve(solver=cp.MOSEK)
print(f"The solving time is {problem.solver_stats.solve_time}.")

# Check the results
if problem.status == cp.OPTIMAL:
    print("Optimal value:", problem.value)
    print("diagonal of D0 matrix:", np.diag(D0.value))
    # print("F0 matrix:", F0.value)
    # print("G0 matrix:", G0.value)
    print(f"Maximum eigenvalue of D: {np.max(np.linalg.eigvals(D))}")
    print(f"Maximum eigenvalue of G: {np.max(np.linalg.eigvals(G))}")
    print(f"Maximum eigenvalue of G-G_0: {np.max(np.linalg.eigvals(G - G0.value))}")
else:
    print("Problem not solved to optimality. Status:", problem.status)

F0_ = F0.value
F0_[np.abs(F0_)<1e-6] = 0

for ii in range(3):
    print(f"adding the {ii + 1}th cut.")
    _, _, _, f_dp = fast_dp_general(G0.value[0:2, 0:2], D0.value, F0_[:, 0:2], -beta, -gamma[:2],
                                    -alpha.reshape(-1, 1))
    # psi_v = f_dp + y.T@y/2
    print(f_dp)
    psi_v = f_dp

    # get dual variables
    model_dul = gp.Model()
    z_dul = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z')
    y_dul = model_dul.addMVar(k, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
    x_dul = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
    z_dul_bar = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z_bar')
    y_dul_bar = model_dul.addMVar(k, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y_bar')
    x_dul_bar = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x_bar')
    t_dul = model_dul.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")

    # add constraints
    model_dul.addConstrs(x_dul[i] <= BIG_M * z_dul[i] for i in range(n))
    model_dul.addConstrs(x_dul[i] >= -BIG_M * z_dul[i] for i in range(n))
    z_equal = model_dul.addConstrs(z_dul[i] == z_dul_bar[i] for i in range(n))
    y_equal = model_dul.addConstrs(y_dul[i] == y_dul_bar[i] for i in range(k))
    x_equal = model_dul.addConstrs(x_dul[i] == x_dul_bar[i] for i in range(n))

    eqn = y_dul_bar[:2].T @ G0.value[0:2, 0:2] @ y_dul_bar[
                                                 :2] + x_dul_bar.T @ D0.value @ x_dul_bar + x_dul_bar.T @ F0_[:,
                                                                                                          0:2] @ y_dul_bar[
                                                                                                                 :2]
    model_dul.addConstr(t_dul >= eqn)

    model_dul.addConstr(t_dul >= alpha.T @ z_dul + beta.T @ x_dul + gamma[:2].T @ y_dul[:2] + psi_v)

    extra_term = y.T @ y / 2 + y_dul_bar.T @ (G - G0.value) @ y_dul_bar + x_dul_bar.T @ (
                D - D0.value) @ x_dul_bar + x_dul_bar.T @ (
                             F - F0_) @ y_dul_bar + c.T @ x_dul_bar + d.T @ y_dul_bar + lam.T @ z_dul_bar
    # set objective
    model_dul.setObjective(t_dul + extra_term[0], GRB.MINIMIZE)
    model_dul.params.OutputFlag = 0
    model_dul.params.QCPDual = 1
    model_dul.optimize()

    alpha = np.array([z_equal[i].Pi for i in range(n)])
    beta = np.array([x_equal[i].Pi for i in range(n)])
    gamma = np.array([y_equal[i].Pi for i in range(k)])
    combined.append(np.concatenate([alpha, beta, gamma]))
    # BIG_M = np.max([xi.X for xi in x_dul])
    # print(f"The new BIG M is: {BIG_M}")
    print(model_dul.objVal)
print(np.abs(np.round(z_opt_vals)))
z_dul_val = np.squeeze([zi.X for zi in z_dul])
thr = np.quantile(z_dul_val,0.9)
print(np.array([1.0 if v>thr else 0.0 for v in z_dul_val]))