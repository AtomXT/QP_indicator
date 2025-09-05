import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import gurobipy as gp
from gurobipy import GRB
from sklearn import datasets

from src.rank2 import *


data = np.loadtxt("../data/cal_housing.data", delimiter=',')
data_df = pd.DataFrame(data, columns=["longitude", "latitude", "housingMedianAge", "totalRooms", "totalBedrooms", "population", "households", "medianIncome", "medianHouseValue"]
)
k = 2
n = 30
# Access features and target
# X = (data.data[0:n,[0, 6]] - np.mean(data.data[0:n, [0, 6]], axis=0)) / data.data[0:n,[0, 6]].std(axis=0)
X = data[0:n,[4, 7]]
X = (X - np.mean(X, axis=0))/X.std(axis=0)
y = data[:n, 8]
y = (y-np.mean(y))/np.std(y)

BIG_M = 1000
mu = 0.5
lam = mu*np.ones((n, 1))

start = time.time()
x_dp, z_dp, f_dp = fast_dp(X, y, mu*np.ones((n,1)))
end = time.time()
print(f_dp, f"DP algorithm use {np.round(end-start, 2)} seconds.")
print(f"DP solution {np.round(x_dp, 2)}.")

C = X.T@X/2
D = np.eye(n)/2
Q = X
c = -y
d = - X.T@y
lam = mu*np.ones((n,1))
start = time.time()
x_dp, y_dp, z_dp, f_dp = fast_dp_general(C, D, Q, c, d, lam)
end = time.time()
print(15+f_dp, f"DP algorithm use {np.round(end-start, 2)} seconds.")
print(f"DP solution {np.round(y_dp, 2)}.")