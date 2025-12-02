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


def generate_problem(n=5, m=8, K=6, seed=42):
    """
    Generate convex problem with [D F; F^T G] positive semidefinite.

    This ensures the objective quadratic in (x,y) is convex.
    """
    np.random.seed(seed)

    # Helper: create PSD block matrix
    def make_joint_psd(n, m):
        # Create a random (n+m) x (n+m) PSD matrix
        Z = np.random.randn(n+m, 2*(n+m))
        M_full = Z @ Z.T  # This is PSD

        # Extract blocks
        D = M_full[:n, :][:, :n]
        F = M_full[:n, :][:, n:]
        G = M_full[n:, :][:, n:]

        return D, F, G

    # Generate D, F, G with joint PSD property
    D, F, G = make_joint_psd(n, m)

    # Add small positive diagonal to ensure strong convexity
    D = D + np.eye(n) * 0.1
    G = G + np.eye(m) * 0.1

    # Verify joint PSD property
    joint_matrix = np.block([[D, F], [F.T, G]])
    eigvals = np.linalg.eigvalsh(joint_matrix)
    assert np.all(eigvals > -1e-10), "Joint matrix not PSD!"

    # Generate scenarios
    scenarios = []
    for k in range(K):
        i, j = np.random.choice(m, 2, replace=False)

        # Create joint PSD for constraint [D0_k F0_k; F0_k^T G0_k]
        D0, F0, G0 = make_joint_psd(n, 2)
        D0_k_diag = np.diag(np.diag(D0))
        G0_k = np.zeros((m, m))
        G0_k[i, i] = G0[0, 0]
        G0_k[i, j] = G0[0, 1]
        G0_k[j, i] = G0[1, 0]
        G0_k[j, j] = G0[1, 1]
        F0_k = np.zeros((n, m))
        F0_k[:, [i, j]] = F0

        # Ensure constraint matrix is PSD
        constraint_matrix = np.block([[D0_k_diag, F0_k], [F0_k.T, G0_k]])

        # If not PSD, adjust by increasing diagonal
        min_eig = np.min(np.linalg.eigvalsh(constraint_matrix))
        if min_eig < 0:
            D0_k_diag = D0_k_diag + np.eye(n) * (-min_eig + 0.1)

        scenarios.append({
            'D0': D0_k_diag,
            'F0': F0_k,
            'G0': G0_k,
            'indices': [i, j]
        })

    # Other parameters
    a = np.random.uniform(-10, 10, n)
    b = np.random.uniform(-5, 5, m)
    lam = np.abs(np.random.uniform(0.1, 0.2, n))

    return {
        'n': n, 'm': m, 'K': K, 'M': 20,
        'D': D, 'F': F, 'G': G,
        'a': a, 'b': b, 'lam': lam,
        'scenarios': scenarios
    }


# Generate example
problem = generate_problem(n=50, m=2, K=1)
print("Problem generated with:")
print(f"n={problem['n']} warehouses, m={problem['m']} products, K={problem['K']} scenarios")
print(f"Big-M = {problem['M']}")
print(f"λ (fixed costs) = {problem['lam']}")

# Verify structure of first scenario
scenario1 = problem['scenarios'][0]
print(f"\nFirst scenario - correlated y indices: {scenario1['indices']}")
print(f"D0 is diagonal: {np.allclose(scenario1['D0'], np.diag(np.diag(scenario1['D0'])))}")
print(f"G0 non-zero pattern: {np.where(np.abs(scenario1['G0']) > 1e-10)}")


BIG_M = problem['M']
n, m = problem['n'], problem['m']
D, G, F, c, d, lam = problem['D'], problem['G'], problem['F'], problem['a'], problem['b'], problem['lam']
scenarios = problem['scenarios']
## solve the optimal solution in the proposed formulation without cut
root_bound = [np.inf, -np.inf]

model_opt = gp.Model()
z_opt = model_opt.addMVar(n, vtype=GRB.BINARY, name='z')
y_opt = model_opt.addMVar(m, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
x_opt = model_opt.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
t_opt = model_opt.addMVar(len(scenarios), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="t")

# add constraints
model_opt.addConstrs(x_opt[i] <= BIG_M * z_opt[i] for i in range(n))
model_opt.addConstrs(x_opt[i] >= -BIG_M * z_opt[i] for i in range(n))


for ii, scenario in enumerate(scenarios):
    pair = scenario['indices']
    G0, D0, F0 = scenario['G0'], scenario['D0'], scenario['F0']
    model_opt.addConstr(
        t_opt[ii] >= y_opt[pair].T @ G0[np.ix_(pair, pair)] @ y_opt[pair]/2 + x_opt.T @ D0@ x_opt/2 + x_opt.T @ F0[:, pair] @ y_opt[pair])

obj = y_opt.T @ G @ y_opt/2 + x_opt.T @ D @ x_opt/2 + x_opt.T @ F @ y_opt + c.T @ x_opt + d.T @ y_opt + lam.T @ z_opt
model_opt.setObjective(gp.quicksum(t_opt) + obj, GRB.MINIMIZE)
model_opt.params.OutputFlag = 1
model_opt.params.TimeLimit = 10
model_opt.optimize(record_root_lb)
print('--------------------------------------------------')
print("Solve the optimal solution without cut")
print(f"The obj is {model_opt.objVal}.")
print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100*(root_bound[0]-root_bound[1])/root_bound[0],4)}%. Runtime: {model_opt.runtime}.")
print('--------------------------------------------------')


# get dual variables
model_dul = gp.Model()
z_dul = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z')
y_dul = model_dul.addMVar(m, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
x_dul = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
z_dul_bar = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z_bar')
y_dul_bar = model_dul.addMVar(m, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y_bar')
x_dul_bar = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x_bar')
t_dul = model_dul.addMVar(len(scenarios), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="t")

# add constraints
model_dul.addConstrs(x_dul[i] <= BIG_M * z_dul[i] for i in range(n))
model_dul.addConstrs(x_dul[i] >= -BIG_M * z_dul[i] for i in range(n))
z_equal = model_dul.addConstrs(z_dul[i] == z_dul_bar[i] for i in range(n))
y_equal = model_dul.addConstrs(y_dul[i] == y_dul_bar[i] for i in range(m))
x_equal = model_dul.addConstrs(x_dul[i] == x_dul_bar[i] for i in range(n))
for ii, scenario in enumerate(scenarios):
    pair = scenario['indices']
    G0, D0, F0 = scenario['G0'], scenario['D0'], scenario['F0']
    model_dul.addConstr(
        t_dul[ii] >= y_dul[pair].T @ G0[np.ix_(pair, pair)] @ y_dul[pair]/2 + x_dul.T @ D0@ x_dul/2 + x_dul.T @ F0[:, pair] @ y_dul[pair])

# # set objective
obj = y_dul.T @ G @ y_dul/2 + x_dul.T @ D @ x_dul/2 + x_dul.T @ F @ y_dul + c.T @ x_dul + d.T @ y_dul + lam.T @ z_dul
model_dul.setObjective(gp.quicksum(t_dul) + obj, GRB.MINIMIZE)
model_dul.params.OutputFlag = 0
model_dul.params.QCPDual = 1
model_dul.optimize()
alpha = np.array([z_equal[i].Pi for i in range(n)])
beta = np.array([x_equal[i].Pi for i in range(n)])
gamma = np.array([y_equal[i].Pi for i in range(m)])
print(model_dul.objVal)
print(alpha)

psi_values = []
alphas, betas, gammas = [], [], []

for iii in range(2):
    print(f"adding the {iii + 1}th cut.")
    alphas.append(alpha)
    betas.append(beta)
    gammas.append(gamma)
    psi_values_level = []
    for ii, scenario in enumerate(scenarios):
        pair = scenario['indices']
        G0, D0, F0 = scenario['G0'], scenario['D0'], scenario['F0']
        print(f"- adding {ii + 1}th pair.")
        _, _, _, f_dp = fast_dp_general(G0[np.ix_(pair, pair)] / 2, D0/2 , F0[:, pair], -beta,
                                        -gamma[pair], -alpha.reshape(-1, 1))
        psi_v = f_dp
        psi_values_level.append(psi_v)
        # print(psi_v)
        model_dul.addConstr(t_dul[ii] >= alpha.T @ z_dul + beta.T @ x_dul + gamma[pair].T @ y_dul[pair] + psi_v)
    psi_values.append(psi_values_level)
    model_dul.optimize()

    alpha = np.array([z_equal[i].Pi for i in range(n)])
    beta = np.array([x_equal[i].Pi for i in range(n)])
    gamma = np.array([y_equal[i].Pi for i in range(m)])
    print(model_dul.objVal)
    print(alpha)



# define a container to store the root node lower bound
root_bound = [np.inf, -np.inf]

model_dul = gp.Model()
z_dul = model_dul.addMVar(n, vtype=GRB.BINARY, name='z')
y_dul = model_dul.addMVar(m, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y')
x_dul = model_dul.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
t_dul = model_dul.addMVar(len(scenarios), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="t")

# add constraints
model_dul.addConstrs(x_dul[i] <= BIG_M * z_dul[i] for i in range(n))
model_dul.addConstrs(x_dul[i] >= -BIG_M * z_dul[i] for i in range(n))
for ii, scenario in enumerate(scenarios):
    pair = scenario['indices']
    G0, D0, F0 = scenario['G0'], scenario['D0'], scenario['F0']
    model_dul.addConstr(
        t_dul[ii] >= y_dul[pair].T @ G0[np.ix_(pair, pair)] @ y_dul[pair]/2 + x_dul.T @ D0@ x_dul/2 + x_dul.T @ F0[:, pair] @ y_dul[pair])

for iii in range(len(psi_values)):
    alpha, beta, gamma = alphas[iii], betas[iii], gammas[iii]
    psi_value = psi_values[iii]
    for ii, scenario in enumerate(scenarios):
        pair = scenario['indices']
        G0, D0, F0 = scenario['G0'], scenario['D0'], scenario['F0']
        inv_Gi = np.linalg.inv(G0[np.ix_(pair, pair)])
        model_dul.addConstr(t_dul[ii] >= gamma[pair].T @ y_dul[pair] + x_dul.T@(D0 - F0[:,pair]@inv_Gi@F0[:,pair].T)@x_dul/2 + (F0[:,pair]@inv_Gi@gamma[pair]).T@x_dul - gamma[pair].T@inv_Gi@gamma[pair]/2)
        model_dul.addConstr(t_dul[ii] >= alpha.T @ z_dul + beta.T @ x_dul + gamma[pair].T @ y_dul[pair] + psi_value[ii])

# # set objective
obj = y_dul.T @ G @ y_dul/2 + x_dul.T @ D @ x_dul/2 + x_dul.T @ F @ y_dul + c.T @ x_dul + d.T @ y_dul + lam.T @ z_dul
model_dul.setObjective(gp.quicksum(t_dul) + obj, GRB.MINIMIZE)
model_dul.params.OutputFlag = 1
model_dul.params.TimeLimit = 10
# model_dul.setParam("NodeLimit", 2)
model_dul.optimize(record_root_lb)
print('--------------------------------------------------')
print("Solve the problem with cut")
print(model_dul.objVal)
print(f"The root upper bound is: {root_bound[0]}, lower bound is: {root_bound[1]}. The root gap is: {np.round(100*(root_bound[0]-root_bound[1])/root_bound[0],4)}%. Runtime: {model_dul.runtime}.")
print('--------------------------------------------------')


z_dul_val = np.squeeze([zi.X for zi in z_dul])
thr = np.quantile(z_dul_val,0.5)
print(np.array([1.0 if v>thr else 0.0 for v in z_dul_val]))
# print(np.abs(z_opt_vals))