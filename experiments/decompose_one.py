# decompose the problem into one small problem with m = 2
import random

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from sklearn import datasets
from itertools import combinations

data = datasets.load_diabetes()

n = 50
m = 3
# Access features and target
A = data.data[0:n, 0:m]
r = data.target[:n]
A = (A - np.mean(A, axis=0)) / np.std(A, axis=0)
r = (r-np.mean(r))/np.std(r)

D = np.eye(n)
D = D + np.eye(n)*0.5  # regularization
G = A.T@A
F = A

# create a list of variables
# s = m*(m-1)//2  # number of subproblems
s = 1
chosen_pairs = (0, 1)

D0 = cp.diag(cp.Variable(n))
F0 = cp.Variable((n, m))
G0 = cp.Variable((m, m))


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



