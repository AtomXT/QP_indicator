import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import gurobipy as gp
from gurobipy import GRB
import os
from src.rank2 import fast_dp_general

np.set_printoptions(linewidth=200)

data = np.loadtxt("../../data/cal_housing.data", delimiter=',')
data_df = pd.DataFrame(data, columns=["longitude", "latitude", "housingMedianAge", "totalRooms", "totalBedrooms", "population", "households", "medianIncome", "medianHouseValue"]
)

k = 2
n = 100
# Access features and target
# X = (data.data[0:n,[0, 6]] - np.mean(data.data[0:n, [0, 6]], axis=0)) / data.data[0:n,[0, 6]].std(axis=0)
X = data[0:n,[4, 7]]
X = (X - np.mean(X, axis=0))/X.std(axis=0)
y = data[:n, 8]
y = (y-np.mean(y))/np.std(y)


C = X.T@X/2
D = np.eye(n)/2 + 0.5*np.eye(n)
Q = X/2
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
eqn = y.T@y/2 + y_opt.T@C@y_opt + x_opt.T@D@x_opt + x_opt.T@Q@y_opt + c.T@x_opt + d.T@y_opt + lam.T@z_opt
model_opt.setObjective(eqn[0], GRB.MINIMIZE)
# model_opt.params.QCPDual = 1
model_opt.params.OutputFlag = 1
model_opt.params.TimeLimit = 20
model_opt.optimize()
print(f"The obj is {model_opt.objVal}.")
x_opt_vals = np.array([x_opt[i].X for i in range(n)])


# extract solutions
z_opt_vals = np.array([z_opt[i].X for i in range(n)])
y_opt_vals = np.array([y_opt[i].X for i in range(k)])
x_opt_vals = np.array([x_opt[i].X for i in range(n)])
print(f"The value of y_opt: {y_opt_vals}")


## solve it using general dp
start = time.time()
x_dp, y_dp, z_dp, f_dp = fast_dp_general(C, D, Q, c, d, lam)
end = time.time()
print(y.T@y/2+f_dp, f"DP algorithm use {np.round(end-start, 2)} seconds.")
print(f"DP solution {np.round(y_dp, 2)}.")

## Solve relax problem in general form
# get tight big-M
model_relax = gp.Model()
z_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z')
x_relax = model_relax.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
y_relax = model_relax.addMVar(k, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
# add constraints
model_relax.addConstrs(x_relax[i] <= BIG_M*z_relax[i] for i in range(n))
model_relax.addConstrs(x_relax[i] >= -BIG_M*z_relax[i] for i in range(n))

# set objective
eqn = y.T@y/2 + y_relax.T@C@y_relax + x_relax.T@D@x_relax + x_relax.T@Q@y_relax + c.T@x_relax + d.T@y_relax + lam.T@z_relax
model_relax.setObjective(eqn[0], GRB.MINIMIZE)
# model_relax.params.QCPDual = 1
model_relax.params.OutputFlag = 0
model_relax.optimize()
print(f"The relaxed obj is {model_relax.objVal}.")
x_relax_vals = np.array([x_relax[i].X for i in range(n)])

print(f"The larges value of x is {max(x_relax_vals)}")
BIG_M = min(BIG_M, 2*max(abs(x_relax_vals)))
print(f'Use new Big-M {BIG_M}.')

## get dual variables
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
eqn = y.T@y/2 + y_dul_bar.T@C@y_dul_bar + x_dul_bar.T@D@x_dul_bar + x_dul_bar.T@Q@y_dul_bar + c.T@x_dul_bar + d.T@y_dul_bar + lam.T@z_dul_bar
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

for ii in range(10):
    print(f"adding the {ii + 1}th cut.")
    _, _, _, f_dp = fast_dp_general(C, D, Q, c - beta, d - gamma, lam - alpha.reshape(-1, 1))
    psi_v = f_dp + y.T @ y / 2

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
    t_dul = model_dul.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")

    # add constraints
    model_dul.addConstrs(x_dul[i] <= BIG_M * z_dul[i] for i in range(n))
    model_dul.addConstrs(x_dul[i] >= -BIG_M * z_dul[i] for i in range(n))
    z_equal = model_dul.addConstrs(z_dul[i] == z_dul_bar[i] for i in range(n))
    y_equal = model_dul.addConstrs(y_dul[i] == y_dul_bar[i] for i in range(k))
    x_equal = model_dul.addConstrs(x_dul[i] == x_dul_bar[i] for i in range(n))

    eqn = y.T @ y / 2 + y_dul_bar.T @ C @ y_dul_bar + x_dul_bar.T @ D @ x_dul_bar + x_dul_bar.T @ Q @ y_dul_bar + c.T @ x_dul_bar + d.T @ y_dul_bar + lam.T @ z_dul_bar
    model_dul.addConstr(t_dul >= eqn[0])

    model_dul.addConstr(t_dul >= alpha.T @ z_dul + beta.T @ x_dul + gamma.T @ y_dul + psi_v)

    # set objective
    # model_dul.setObjective(t_dul, GRB.MINIMIZE)
    model_dul.setObjective(t_dul, GRB.MINIMIZE)
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


## compare
print(np.abs(np.round(z_opt_vals)))
z_dul_val = np.squeeze([zi.X for zi in z_dul])
thr = np.quantile(z_dul_val,0.9)
print(np.array([1.0 if v>thr else 0.0 for v in z_dul_val]))