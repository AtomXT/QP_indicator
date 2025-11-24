import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import gurobipy as gp
import cvxpy as cp
import random
from gurobipy import GRB
import os

np.set_printoptions(linewidth=200)

from sklearn import datasets
from itertools import combinations
from src.rank2 import fast_dp_general

def generate_sparse_data(n_samples=100, n_features=20, true_sparsity=5, noise_level=0.1, random_state=42):
    """Generate synthetic data with sparse true coefficients"""
    np.random.seed(random_state)

    # Generate true coefficients with sparsity
    true_beta = np.zeros(n_features)
    important_features = np.random.choice(n_features, true_sparsity, replace=False)
    true_beta[important_features] = np.random.normal(0, 2, true_sparsity)

    # Generate features with correlation
    X = np.random.normal(0, 1, (n_samples, n_features))

    # Add some correlation between features
    for i in range(1, n_features):
        X[:, i] = 0.7 * X[:, i - 1] + 0.3 * X[:, i]

    # Generate target with noise
    y = X @ true_beta + np.random.normal(0, noise_level, n_samples)

    return X, y, true_beta, important_features


print("Generating synthetic data...")
p = 20
n = 100
s = 5
X, y, true_beta, important_features = generate_sparse_data(
    n_samples=n,
    n_features=p,  # Smaller problem for reasonable computation time
    true_sparsity=s,
    noise_level=0.5
)

print(f"True important features: {important_features}")
print(f"True coefficients (non-zero): {true_beta[important_features]}")
# X = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,15,17]]
# important_features = [0,1,8,18,19]
# true_beta = true_beta[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,15,17]]

# Access features and target
# X = (X - np.mean(X, axis=0))/X.std(axis=0)
# y = (y-np.mean(y))/np.std(y)



Q = X.T@X
c_ori = -X.T@y

u, v = 15, 17
keep_indices = [k for k in range(p) if k != u and k != v]
submatrix = X[keep_indices, :][:, keep_indices]


c = c_ori[keep_indices]
d = c_ori[[u, v]]



D = Q[keep_indices, :][:, keep_indices]
F = Q[keep_indices, :][:, [u, v]]
G = Q[[u, v], :][:, [u, v]]


BIG_M = 10
mu = 1.4
lam = mu*np.ones((p, 1))
# print(np.linalg.eigvalsh(np.bmat([[D, F/2], [F.T/2, G]]))[0:5])

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


## solve the problem in original formulation
model_ori = gp.Model()
z_ori = model_ori.addMVar(p, vtype=GRB.BINARY)
beta_ori = model_ori.addMVar(p, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta")
res = y - X@beta_ori
model_ori.setObjective(res@res/2 + lam.T@z_ori, GRB.MINIMIZE)
model_ori.addConstrs(beta_ori[i] <= BIG_M*z_ori[i] for i in range(p))
model_ori.addConstrs(beta_ori[i] >= -BIG_M*z_ori[i] for i in range(p))
model_ori.params.OutputFlag = 1
model_ori.params.TimeLimit = 30
model_ori.optimize(record_root_lb)
z_opt_vals = np.array([z_ori[i].X for i in range(p)])
beta_opt_vals = np.array([beta_ori[i].X for i in range(p)])
print('--------------------------------')
print('solve the problem in original formulation')
print(f"The obj is {model_ori.objVal}.")
print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100*(root_bound[0]-root_bound[1])/root_bound[0],4)}%. Runtime: {model_ori.runtime}.")
print(np.where(z_opt_vals == 1)[0])
print(f"Estimated beta: {beta_opt_vals}")
print(f"True beta: {true_beta}")
print('--------------------------------')

## solve the problem in the quadratic formulation
root_bound = [np.inf, -np.inf]
## get the optimal solution
model_opt = gp.Model()
z_opt = model_opt.addMVar(p, vtype=GRB.BINARY, lb=0, ub=1, name='z')
x_opt = model_opt.addMVar(p, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
# add constraints
model_opt.addConstrs(x_opt[i] <= BIG_M*z_opt[i] for i in range(p))
model_opt.addConstrs(x_opt[i] >= -BIG_M*z_opt[i] for i in range(p))
# set objective
eqn = y.T@y/2 + x_opt.T@Q@x_opt/2 + c_ori.T@x_opt + lam.T@z_opt
model_opt.setObjective(eqn[0], GRB.MINIMIZE)
# model_opt.params.QCPDual = 1
model_opt.params.OutputFlag = 0
model_opt.params.TimeLimit = 30
model_opt.optimize(record_root_lb)
z_opt_vals = np.array([z_opt[i].X for i in range(p)])
beta_opt_vals = np.array([x_opt[i].X for i in range(p)])
print('solve the problem in the quadratic formulation')
print(f"The obj is {model_opt.objVal}.")
print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100*(root_bound[0]-root_bound[1])/root_bound[0],4)}%. Runtime: {model_opt.runtime}.")
print(np.where(z_opt_vals == 1)[0])
print(f"Estimated beta: {beta_opt_vals}")
print(f"True beta: {true_beta}")
print('--------------------------------')
#
# # extract solutions
# z_opt_vals = np.array([z_opt[i].X for i in range(n)])
# y_opt_vals = np.array([y_opt[i].X for i in range(m)])
# x_opt_vals = np.array([x_opt[i].X for i in range(n)])
#
# get dual variables
model_dul = gp.Model()
z_dul = model_dul.addMVar(p-2, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z')
# z_dul = model_dul.addMVar(p, vtype=GRB.BINARY, name='z')
x_dul = model_dul.addMVar(p-2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
y_dul = model_dul.addMVar(2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
z_dul_bar = model_dul.addMVar(p-2, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z_bar')
# z_dul_bar = model_dul.addMVar(p, vtype=GRB.BINARY, name='z_bar')
# z_dul_bar = model_dul.addMVar(n, vtype=GRB.BINARY, name='z_bar')
x_dul_bar = model_dul.addMVar(p-2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x_bar')
y_dul_bar = model_dul.addMVar(2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y_bar')
t_dul = model_dul.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="t")

# add constraints
model_dul.addConstrs(x_dul[i] <= BIG_M*z_dul[i] for i in range(p-2))
model_dul.addConstrs(x_dul[i] >= -BIG_M*z_dul[i] for i in range(p-2))
z_equal = model_dul.addConstrs(z_dul[i] == z_dul_bar[i] for i in range(p-2))
x_equal = model_dul.addConstrs(x_dul[i] == x_dul_bar[i] for i in range(p-2))
y_equal = model_dul.addConstrs(y_dul[i] == y_dul_bar[i] for i in range(2))

# set objective
# model_dul.setObjective(t_dul, GRB.MINIMIZE)
eqn = mu*2 + y.T@y/2 + y_dul.T@G@y_dul/2 + x_dul.T@D@x_dul/2 + x_dul.T@F@y_dul + c.T@x_dul + d.T@y_dul + lam[keep_indices].T@z_dul
model_dul.setObjective(eqn[0], GRB.MINIMIZE)
model_dul.params.OutputFlag = 1
model_dul.params.QCPDual = 0
model_dul.optimize()

alpha = np.array([z_equal[i].Pi for i in range(p-2)])
beta = np.array([x_equal[i].Pi for i in range(p-2)])
gamma = np.array([y_equal[i].Pi for i in range(2)])
print(alpha, beta, gamma)
#
#
s = 1
index_pair = [list(t) for t in combinations(range(p), 2)]
# pairs = random.sample(index_pair, s)
pairs = [[u, v]]

Di = [cp.diag(cp.Variable(p-2)) for i in range(s)]
Fi = [cp.Variable((p-2, 2)) for i in range(s)]
Gi = [cp.Variable((2, 2)) for i in range(s)]


# constraints
constraint_0 = [Q - cp.bmat([[cp.sum(Di), cp.sum(Fi)/2], [cp.sum(Fi).T/2, cp.sum(Gi)]]) >> 0]
# constraint_0 = []
for i in range(s):
    constraint_0.append(cp.bmat([[Di[i], Fi[i]/2], [Fi[i].T/2, Gi[i]]]) >> 0)

# objective = cp.Minimize(cp.norm_inf(D - cp.sum(Di))+cp.norm_inf(G - cp.sum(Gi)))
# objective = cp.Minimize(cp.norm_inf(G - cp.sum(Gi)))
# objective = cp.Minimize(cp.norm(D - cp.sum(Di), 'nuc')+cp.norm(G - cp.sum(Gi), 'nuc'))
# objective = cp.Minimize(cp.sum(cp.diag(-cp.sum(Di))) + cp.sum(cp.diag(-cp.sum(Gi))))
# objective = cp.Maximize(-6.6e-6*cp.norm(Fi[0], 1)+cp.lambda_min(cp.bmat([[Di[0], Fi[0]/2], [Fi[0].T/2, Gi[0]]])))
obj_expr = 0
for ii in range(s):
    obj_expr += cp.lambda_min(cp.bmat([[Di[ii], Fi[ii]/2], [Fi[ii].T/2, Gi[ii]]]))
objective = cp.Maximize(obj_expr)
# objective = cp.Minimize(0*cp.norm(Fi[0], 1)+cp.lambda_max(cp.bmat([[D - cp.sum(Di), F/2 - cp.sum(Fi)/2], [(F/2 - cp.sum(Fi)/2).T, G - cp.sum(Gi)]])))

# Formulate the optimization problem
problem = cp.Problem(objective, constraint_0)

# Solve the problem using MOSEK
problem.solve(solver=cp.MOSEK)
print(f"The decomposition time is {problem.solver_stats.solve_time}.")

# Check the results
if problem.status == cp.OPTIMAL:
    print("Optimal value:", problem.value)
    print(f"Maximum eigenvalue of Q: {np.max(np.linalg.eigvals(Q))}")
    print(f"Minimum eigenvalue of Q: {np.min(np.linalg.eigvals(Q))}")
    print(f"Maximum eigenvalue of unstructured problem: {np.max(np.linalg.eigvalsh(Q - np.block([[cp.sum(Di).value, cp.sum(Fi).value], [cp.sum(Fi).value.T, cp.sum(Gi).value]])))}")
    print(f"Minimum eigenvalue of unstructured problem: {np.min(np.linalg.eigvalsh(Q - np.block([[cp.sum(Di).value, cp.sum(Fi).value], [cp.sum(Fi).value.T, cp.sum(Gi).value]])))}")
    # print(np.linalg.eigvalsh(np.block([[D - cp.sum(Di).value, F/2 - cp.sum(Fi).value/2], [(F/2 - cp.sum(Fi).value/2).T, G - cp.sum(Gi).value]]))[0:3])
else:
    print("Problem not solved to optimality. Status:", problem.status)

Gi_ = [np.where(np.abs(Gi[ii].value) < 1e-8, 0, Gi[ii].value) for ii in range(len(pairs))]
Fi_ = [np.where(np.abs(Fi[ii].value) < 1e-8, 0, Fi[ii].value) for ii in range(len(pairs))]
Di_ = [np.where(Di[ii].value < 1e-8, 0, Di[ii].value) for ii in range(len(pairs))]
Gi_sum_diff_, Di_sum_diff_, Fi_sum_diff_ = G - cp.sum(Gi_), D - cp.sum(Di_), F - cp.sum(Fi_)
# Gi_sum_diff_[np.abs(Gi_sum_diff_) < 1e-6] = 0
# Fi_sum_diff_[np.abs(Fi_sum_diff_) < 1e-6] = 0
# Di_sum_diff_[np.abs(Di_sum_diff_) < 1e-6] = 0

print(f"Number of nonzero rows in F_0: {np.sum(np.count_nonzero(Fi_[0], axis=1) != 0)}")
print("------------------------")
#
for iii in range(1):
    print(f"adding the {iii + 1}th cut.")
    # get dual variables
    model_dul = gp.Model()
    z_dul = model_dul.addMVar(p-2, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z')
    y_dul = model_dul.addMVar(2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
    x_dul = model_dul.addMVar(p-2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
    z_dul_bar = model_dul.addMVar(p-2, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z_bar')
    y_dul_bar = model_dul.addMVar(2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y_bar')
    x_dul_bar = model_dul.addMVar(p-2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x_bar')
    t_dul = model_dul.addMVar(len(pairs), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")

    # add constraints
    model_dul.addConstrs(x_dul[i] <= BIG_M * z_dul[i] for i in range(p-2))
    model_dul.addConstrs(x_dul[i] >= -BIG_M * z_dul[i] for i in range(p-2))
    z_equal = model_dul.addConstrs(z_dul[i] == z_dul_bar[i] for i in range(p-2))
    y_equal = model_dul.addConstrs(y_dul[i] == y_dul_bar[i] for i in range(2))
    x_equal = model_dul.addConstrs(x_dul[i] == x_dul_bar[i] for i in range(p-2))
    psi_values = []
    for ii, pair in enumerate(pairs):
        print(f"- adding {ii + 1}th pair.")
        _, _, _, f_dp = fast_dp_general(Gi_[ii], Di_[ii], Fi_[ii], -beta,
                                        -gamma, -alpha.reshape(-1, 1))
        psi_v = f_dp
        psi_values.append(psi_v)
        # print(psi_v)
        model_dul.addConstr(
            t_dul[ii] >= y_dul_bar.T @ Gi_[ii] @ y_dul_bar/2 + x_dul_bar.T @ Di_[
                ii] @ x_dul_bar/2 + x_dul_bar.T @ Fi_[ii] @ y_dul_bar)

        model_dul.addConstr(t_dul[ii] >= alpha.T @ z_dul + beta.T @ x_dul + gamma.T @ y_dul + psi_v)

    extra_term = mu*2 + y.T @ y / 2 + y_dul_bar.T @ Gi_sum_diff_ @ y_dul_bar/2 + x_dul_bar.T @ Di_sum_diff_ @ x_dul_bar/2 + x_dul_bar.T @ Fi_sum_diff_ @ y_dul_bar + c.T @ x_dul_bar + d.T @ y_dul_bar + lam[keep_indices].T @ z_dul_bar
    # extra_term = y.T@y/2 + c.T@x_dul_bar + d.T@y_dul_bar + lam.T@z_dul_bar
    # # set objective
    model_dul.setObjective(gp.quicksum(t_dul) + extra_term[0], GRB.MINIMIZE)
    model_dul.params.OutputFlag = 0
    model_dul.params.QCPDual = 1
    model_dul.optimize()

    alpha = np.array([z_equal[i].Pi for i in range(p-2)])
    beta = np.array([x_equal[i].Pi for i in range(p-2)])
    gamma = np.array([y_equal[i].Pi for i in range(2)])
    print(model_dul.objVal)

z_dul_val = np.squeeze([zi.X for zi in z_dul_bar])
thr = np.quantile(z_dul_val,0.9)
print(np.array([1.0 if v>thr else 0.0 for v in z_dul_val]))
print(np.abs(z_opt_vals))
#
#
# ## solve the optimal solution in the proposed formulation without cut
# root_bound = [np.inf, -np.inf]
#
# model_opt = gp.Model()
# z_opt = model_opt.addMVar(n, vtype=GRB.BINARY, name='z')
# y_opt = model_opt.addMVar(m, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
# x_opt = model_opt.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
# t_opt = model_opt.addMVar(len(pairs), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")
#
# # add constraints
# model_opt.addConstrs(x_opt[i] <= BIG_M * z_opt[i] for i in range(n))
# model_opt.addConstrs(x_opt[i] >= -BIG_M * z_opt[i] for i in range(n))
#
#
# for ii, pair in enumerate(pairs):
#     model_opt.addConstr(
#         t_opt[ii] >= y_opt[pair].T @ Gi_[ii][np.ix_(pair, pair)] @ y_opt[pair] + x_opt.T @ Di_[
#             ii] @ x_opt + x_opt.T @ Fi_[ii][:, pair] @ y_opt[pair])
#
# extra_term = y.T @ y / 2 + y_opt.T @ Gi_sum_diff_ @ y_opt + x_opt.T @ Di_sum_diff_ @ x_opt + x_opt.T @ Fi_sum_diff_ @ y_opt + c.T @ x_opt + d.T @ y_opt + lam.T @ z_opt
# model_opt.setObjective(gp.quicksum(t_opt) + extra_term[0], GRB.MINIMIZE)
# model_opt.params.OutputFlag = 1
# model_opt.params.TimeLimit = 3
# model_opt.optimize(record_root_lb)
# print('--------------------------------------------------')
# print("Solve the optimal solution in the proposed formulation without cut")
# print(f"The obj is {model_opt.objVal}.")
# print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100*(root_bound[0]-root_bound[1])/root_bound[0],4)}%. Runtime: {model_opt.runtime}.")
# print('--------------------------------------------------')
#
#
# ## cut and branch
#
# # define a container to store the root node lower bound
# root_bound = [np.inf, -np.inf]
# def record_root_lb(model, where):
#     if where == GRB.Callback.MIPNODE:
#         # check if this is the root node
#         nodecnt = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
#         if nodecnt == 0:
#             # get the relaxation bound at this node
#             lb = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
#             ub = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
#             # store it if not yet recorded
#             if lb >= root_bound[1]:
#                 root_bound[1] = lb
#             if ub <= root_bound[0]:
#                 root_bound[0] = ub
#
# # get dual variables
# model_dul = gp.Model()
# z_dul = model_dul.addMVar(n, vtype=GRB.BINARY, name='z')
# y_dul = model_dul.addMVar(m, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
# x_dul = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
# t_dul = model_dul.addMVar(len(pairs), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")
#
# # add constraints
# model_dul.addConstrs(x_dul[i] <= BIG_M * z_dul[i] for i in range(n))
# model_dul.addConstrs(x_dul[i] >= -BIG_M * z_dul[i] for i in range(n))
#
# for ii, pair in enumerate(pairs):
#     print(f"- adding {ii + 1}th pair.")
#     model_dul.addConstr(
#         t_dul[ii] >= y_dul[pair].T @ Gi_[ii][np.ix_(pair, pair)] @ y_dul[pair] + x_dul.T @ Di_[
#             ii] @ x_dul + x_dul.T @ Fi_[ii][:, pair] @ y_dul[pair])
#
#     model_dul.addConstr(t_dul[ii] >= alpha.T @ z_dul + beta.T @ x_dul + gamma[pair].T @ y_dul[pair] + psi_values[ii])
#
# extra_term = y.T @ y / 2 + y_dul.T @ Gi_sum_diff_ @ y_dul + x_dul.T @ Di_sum_diff_ @ x_dul + x_dul.T @ Fi_sum_diff_ @ y_dul + c.T @ x_dul + d.T @ y_dul + lam.T @ z_dul
# # # set objective
# model_dul.setObjective(gp.quicksum(t_dul) + extra_term[0], GRB.MINIMIZE)
# model_dul.params.OutputFlag = 1
# model_dul.params.TimeLimit = 3
# # model_dul.setParam("NodeLimit", 2)
# model_dul.optimize(record_root_lb)
# print('--------------------------------------------------')
# print("Solve the problem in the proposed formulation with cut")
# print(model_dul.objVal)
# print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100*(root_bound[0]-root_bound[1])/root_bound[0],4)}%. Runtime: {model_dul.runtime}.")
# print('--------------------------------------------------')
#
#
# z_dul_val = np.squeeze([zi.X for zi in z_dul])
# thr = np.quantile(z_dul_val,0.8)
# print(np.array([1.0 if v>thr else 0.0 for v in z_dul_val]))
# print(np.abs(z_opt_vals))