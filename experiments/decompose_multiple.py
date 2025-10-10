# decompose the problem into multiple small problems with m = 2
import random

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from sklearn import datasets
from itertools import combinations


data = datasets.load_diabetes()

n = 50
m = 5
# Access features and target
A = data.data[0:n,0:m]  # [0,1,2,3,4,9]
# A[:,1] = -A[:,1]
r = data.target[:n]
A = (A - np.mean(A, axis=0)) / np.std(A, axis=0)
r = (r-np.mean(r))/np.std(r)

D = np.eye(n)
D = D + 0.5*np.eye(n)  # regularization
G = A.T@A
# G = A.T@A + np.eye(m)*200
# G[~np.eye(m, dtype=bool)] /= 200
F = A/2



# create a list of variables
# s = m*(m-1)//2  # number of subproblems
s = 10
index_pair = list(combinations(range(m), 2))
chosen_pairs = random.sample(index_pair, s)

Di = [cp.diag(cp.Variable(n)) for i in range(s)]
Fi = [cp.Variable((n, m)) for i in range(s)]
Gi = [cp.Variable((m, m)) for i in range(s)]
Fi_mask = []
Gi_mask = []
for i in range(s):
    ii, jj = chosen_pairs[i]

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


    # for j in range(n):
    #     constraint_0.append(Q[i][j] >> 0)
    #     constraint_0.append(D[j,j] - Di[i][j,j] >= 0)
    #     constraint_0.append(Qj[i][j] == F[j,:].reshape(1,-1) - Fi[i][j,:])
    #     constraint_0.append(Q[i][j] * (D[j, j] - Di[i][j, j])>> Qj[i][j].T @ Qj[i][j])
    #     # constraint_0.append(Q[i][j] >> (F[j,:].reshape(1,-1) - Fi[i][j,:]).T @ (F[j,:].reshape(1,-1) - Fi[i][j,:]) / (D[j, j] - Di[i][j, j]))
    #     # constraint_0.append(Q[i][j] >> (F-Fi[i]).T@np.diag(I[:,j])@(F-Fi[i])/(D[j,j] - Di[i][j,j]))
    # constraint_0.append(G-Gi[i] - cp.sum(Q[i]) >> 0)

    '''
    try to find one set of closed form solution
    '''
    # constraint_0.append(Di[i] == D/(m*(m-1)//2))
    # ii, jj = chosen_pairs[i]
    # constraint_0.append(Gi[i][ii, jj] == G[ii, jj])
    # constraint_0.append(Gi[i][jj, ii] == G[jj, ii])
    # constraint_0.append(Gi[i][ii, ii] == G[ii, ii] / (m - 1))
    # constraint_0.append(Gi[i][jj, jj] == G[jj, jj] / (m - 1))
    # constraint_0.append(Fi[i][:, ii] == F[:, ii] / (m - 1))
    # constraint_0.append(Fi[i][:, jj] == F[:, jj] / (m - 1))

# objective = cp.Minimize(cp.norm_inf(D - cp.sum(Di))+cp.norm_inf(G - cp.sum(Gi)))
# objective = cp.Minimize(cp.norm_inf(G - cp.sum(Gi)))
# objective = cp.Minimize(cp.norm(D - cp.sum(Di), 'nuc')+cp.norm(G - cp.sum(Gi), 'nuc'))
# objective = cp.Minimize(cp.sum(cp.diag(-cp.sum(Di))) + cp.sum(cp.diag(-cp.sum(Gi))))
objective = cp.Minimize(cp.lambda_max(cp.bmat([[D - cp.sum(Di), F - cp.sum(Fi)], [(F - cp.sum(Fi)).T, G - cp.sum(Gi)]])))

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



