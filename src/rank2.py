import numpy as np


def find_candidates_dp(A1, A2, y, lam):
    Z = []
    n = len(y)
    for i in range(n):
        bounds1 = []
        bounds2 = []
        for j in range(n):
            if j != i:
                denominator = A2[j][0] - A1[j][0] * A2[i][0] / A1[i][0]
                if denominator != 0:
                    numerator1 = -np.sqrt(2*lam[j][0]) - y[j][0] + A1[j][0]*y[i][0]/A1[i][0]
                    numerator2 = np.sqrt(2*lam[j][0]) - y[j][0] + A1[j][0]*y[i][0]/A1[i][0]
                    al = A1[j][0]*np.sqrt(2*lam[i][0])/A1[i][0]
                    l1, u1 = (numerator1 + al) / denominator, (numerator2 + al) / denominator
                    if l1 > u1: l1, u1 = u1, l1
                    l2, u2 = (numerator1 - al) / denominator, (numerator2 - al) / denominator
                    if l2 > u2: l2, u2 = u2, l2
                    bounds1.append((l1, j, 0))
                    bounds1.append((u1, j, 1))
                    bounds2.append((l2, j, 0))
                    bounds2.append((u2, j, 1))

        # -------------The flowing method produces 2n(4n-2) candidates. ----------------
        bounds1.sort(key=lambda x: x[0])
        z = np.ones((n,1))
        z_ = np.ones((n,1))
        z_[i] = 0
        Z.append(z.copy())
        Z.append(z_.copy())
        for ii in range(len(bounds1)):
            jj = bounds1[ii][1]
            if z[jj] == 1:
                z[jj], z_[jj] = 0, 0
                Z.append(z.copy())
                Z.append(z_.copy())
            else:
                z[jj], z_[jj] = 1, 1
                Z.append(z.copy())
                Z.append(z_.copy())

        bounds2.sort(key=lambda x: x[0])
        z = np.ones((n, 1))
        z_ = np.ones((n, 1))
        z_[i] = 0
        Z.append(z.copy())
        Z.append(z_.copy())
        # position = [0] * n
        for ii in range(len(bounds2)):
            jj = bounds2[ii][1]
            if z[jj] == 1:
                z[jj], z_[jj] = 0, 0
                Z.append(z.copy())
                Z.append(z_.copy())
            else:
                z[jj], z_[jj] = 1, 1
                Z.append(z.copy())
                Z.append(z_.copy())
    # Z = np.unique(Z, axis=0)
    return Z

def fast_dp(X, y, lam):
    Z = find_candidates_dp(X[:,0].reshape(-1,1), X[:,1].reshape(-1,1), y.reshape(-1,1), lam)
    f_opt = float('inf')
    x_opt = None
    z_opt = None
    for z in Z:
        s = np.squeeze(z==0)  # in this problem, z = 0 means this sample is outlier so not include
        X_s = X[s]
        y_s = y[s]
        if np.linalg.cond(X_s.T@X_s) < 1e4:
            x_z = np.linalg.solve(X_s.T@X_s, X_s.T@y_s)
            # x_z = np.linalg.inv(X_s.T@X_s)@X_s.T@y_s
            f_z = y_s.reshape(-1,1)-X_s@x_z.reshape(-1,1)
            f_z = f_z.T@f_z/2 + lam.T@z
            if f_z < f_opt:
                f_opt = f_z
                x_opt = x_z
                z_opt = z
        else:
            continue
    return x_opt, z_opt, f_opt.item()

def find_candidates_dp_general(C, D, Q, c, d, lam):
    Z = []
    n = len(c)
    for i in range(n):
        bounds1 = []
        bounds2 = []
        for j in range(n):
            if j != i:
                denominator = Q[j, 0]*Q[i, 1]/Q[i, 0] - Q[j, 1] if Q[i, 0] != 0 else - Q[j, 1]
                if denominator != 0:
                    numerator1 = -2*np.sqrt(D[j,j]*lam[j][0]) - c[j] + Q[j, 0]*c[i]/Q[i, 0] if Q[i, 0] != 0 else -2*np.sqrt(D[j,j]*lam[j][0]) - c[j]
                    numerator2 = 2*np.sqrt(D[j,j]*lam[j][0]) - c[j] + Q[j, 0]*c[i]/Q[i, 0] if Q[i, 0] != 0 else 2*np.sqrt(D[j,j]*lam[j][0]) - c[j]
                    al = Q[j, 0]*2*np.sqrt(D[i,i]*lam[i][0])/Q[i, 0] if Q[i, 0] != 0 else 0
                    l1, u1 = (numerator1 + al) / denominator, (numerator2 + al) / denominator
                    if l1 > u1: l1, u1 = u1, l1
                    l2, u2 = (numerator1 - al) / denominator, (numerator2 - al) / denominator
                    if l2 > u2: l2, u2 = u2, l2
                    bounds1.append((l1, j, 0))
                    bounds1.append((u1, j, 1))
                    bounds2.append((l2, j, 0))
                    bounds2.append((u2, j, 1))

        # -------------The flowing method produces 2n(4n-2) candidates. ----------------
        bounds1.sort(key=lambda x: x[0])
        z = np.ones((n,1))
        z_ = np.ones((n,1))
        z_[i] = 0
        Z.append(z.copy())
        Z.append(z_.copy())
        for ii in range(len(bounds1)):
            jj = bounds1[ii][1]
            if z[jj] == 1:
                z[jj], z_[jj] = 0, 0
                Z.append(z.copy())
                Z.append(z_.copy())
            else:
                z[jj], z_[jj] = 1, 1
                Z.append(z.copy())
                Z.append(z_.copy())

        bounds2.sort(key=lambda x: x[0])
        z = np.ones((n, 1))
        z_ = np.ones((n, 1))
        z_[i] = 0
        Z.append(z.copy())
        Z.append(z_.copy())
        # position = [0] * n
        for ii in range(len(bounds2)):
            jj = bounds2[ii][1]
            if z[jj] == 1:
                z[jj], z_[jj] = 0, 0
                Z.append(z.copy())
                Z.append(z_.copy())
            else:
                z[jj], z_[jj] = 1, 1
                Z.append(z.copy())
                Z.append(z_.copy())
    # Z = np.unique(Z, axis=0)
    return Z


def fast_dp_general(C, D, Q, c, d, lam):
    '''
    Solve the general case:
    min_{y} y'Cy + d'y + min_{x(1-z)=0} x'Dx + x'Qy + c'x + lam'z,
    where y is 2 dimensional, D is diagonal, and the problem is convex.
    '''
    n = len(c)
    c = c.reshape((n, 1))
    d = d.reshape((2, 1))
    Z = find_candidates_dp_general(C, D, Q, c, d, lam)
    f_opt = float('inf')
    x_opt = None
    y_opt = None
    z_opt = None
    for i, z in enumerate(Z):
        s = np.squeeze(z==1)
        D_s = np.diag(D)[s]
        Q_s = Q[s, :]
        c_s = c[s]
        A = C - Q_s.T/D_s@Q_s/4
        b = d - Q_s.T/D_s@c_s/2
        if np.linalg.cond(A) < 1e4 and np.max(np.linalg.eigvals(A)) > 1e-3:
            y = np.linalg.solve(2 * A, -b)
            x_z = - (Q_s @ y + c_s)/D_s.reshape(-1,1) / 2
            # mask where |D_s| is very small
            eps = 1e-3  # tolerance threshold
            mask = np.abs(D_s).reshape(-1, 1) < eps

            # assign zero where D_s is close to zero
            x_z = np.where(mask, 0, x_z)
            f_z = y.T @ C @ y + x_z.T * D_s @ x_z + x_z.T @ Q_s @ y + c_s.T @ x_z + d.T @ y + lam.T @ z
            if f_z < f_opt:
                f_opt, x_opt, y_opt, z_opt = f_z, x_z, y, z
        else:
            continue
    return x_opt, y_opt, z_opt, f_opt.item()


## solving psi using dp

def find_candidates_psi(A1, A2, y, lam, alpha, beta, gamma):
    Z = []
    n = len(y)
    for i in range(n):
        bounds1 = []
        bounds2 = []
        for j in range(n):
            if j != i:
                denominator = A1[j][0]*A2[i][0]/A1[i][0] - A2[j][0]
                if denominator != 0:
                    numerator1 = -np.sqrt(gamma[j]*gamma[j]+2*lam[j][0]-2*alpha[j]) - (y[j][0]+gamma[j]) + A1[j][0]*(y[i][0]+gamma[i])/A1[i][0]
                    numerator2 = np.sqrt(gamma[j]*gamma[j]+2*lam[j][0]-2*alpha[j]) - (y[j][0]+gamma[j]) + A1[j][0]*(y[i][0]+gamma[i])/A1[i][0]
                    al = A1[j][0]*np.sqrt(gamma[j]*gamma[j] + 2*lam[i][0]-2*alpha[i])/A1[i][0]
                    l1, u1 = (numerator1 + al) / denominator, (numerator2 + al) / denominator
                    if l1 > u1: l1, u1 = u1, l1
                    l2, u2 = (numerator1 - al) / denominator, (numerator2 - al) / denominator
                    if l2 > u2: l2, u2 = u2, l2
                    bounds1.append((l1, j, 0))
                    bounds1.append((u1, j, 1))
                    bounds2.append((l2, j, 0))
                    bounds2.append((u2, j, 1))

        # -------------The flowing method produces 2n(4n-2) candidates. ----------------
        bounds1.sort(key=lambda x: x[0])
        z = np.ones((n,1))
        z_ = np.ones((n,1))
        z_[i] = 0
        Z.append(z.copy())
        Z.append(z_.copy())
        for ii in range(len(bounds1)):
            jj = bounds1[ii][1]
            if z[jj] == 1:
                z[jj], z_[jj] = 0, 0
                Z.append(z.copy())
                Z.append(z_.copy())
            else:
                z[jj], z_[jj] = 1, 1
                Z.append(z.copy())
                Z.append(z_.copy())

        bounds2.sort(key=lambda x: x[0])
        z = np.ones((n, 1))
        z_ = np.ones((n, 1))
        z_[i] = 0
        Z.append(z.copy())
        Z.append(z_.copy())
        # position = [0] * n
        for ii in range(len(bounds2)):
            jj = bounds2[ii][1]
            if z[jj] == 1:
                z[jj], z_[jj] = 0, 0
                Z.append(z.copy())
                Z.append(z_.copy())
            else:
                z[jj], z_[jj] = 1, 1
                Z.append(z.copy())
                Z.append(z_.copy())
    # Z = np.unique(Z, axis=0)
    return Z


def fast_psi(X, y, lam, alpha, beta, gamma):
    Z = find_candidates_psi(X[:,0].reshape(-1,1), X[:,1].reshape(-1,1), y.reshape(-1,1), lam, alpha, beta, gamma)
    # Z = []
    f_opt = float('inf')
    x_opt = None
    z_opt = None
    n = len(alpha)
    k = len(beta)
    # Z.append(np.array([1 if zi == 1 else 0 for zi in z_opt_vals]))
    for z in Z:
        # model = gp.Model()
        # # z_opt = model_opt.addMVar(n, vtype=GRB.BINARY, name='z')
        # x_opt = model.addMVar(k, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
        # w_opt = model.addMVar(n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='w')
        # model.addConstrs(w_opt[i] <= BIG_M*z[i] for i in range(n))
        # model.addConstrs(w_opt[i] >= -BIG_M*z[i] for i in range(n))
        # model.setObjective(gp.quicksum((y[i]-X[i,:]@x_opt-w_opt[i])*(y[i]-X[i,:]@x_opt-w_opt[i])/2 for i in range(n))+ lam.T@z-alpha.T@z-beta.T@x_opt-gamma.T@w_opt, GRB.MINIMIZE)
        # model.optimize()
        # if model.objVal < f_opt:
        #     f_opt = model.objVal
        sc = np.squeeze(z==0)
        s = np.squeeze(z==1)
        X_s, X_sc = X[s], X[sc]
        y_s, y_sc = y[s], y[sc]
        if np.linalg.cond(X_sc.T@X_sc) < 1e4:
            x_z = np.linalg.solve(X_sc.T@X_sc, X_sc.T@y_sc.reshape(-1,1)+beta.reshape(-1,1)+X.T@gamma.reshape(-1,1))
            # x_z = np.linalg.inv(X_s.T@X_s)@X_s.T@y_s
            f_z = y_sc.reshape(-1,1)-X_sc@x_z.reshape(-1,1)
            f_z = f_z.T@f_z/2 + lam[0]*np.count_nonzero(z) - alpha.reshape(-1,1).T@z - beta.reshape(-1,1).T@x_z - gamma.reshape(-1,1).T@(y.reshape(-1,1) - X@x_z) + gamma[s].reshape(-1,1).T@gamma[s].reshape(-1,1)/2 - gamma.reshape(-1,1).T@gamma.reshape(-1,1)
            if f_z < f_opt:
                f_opt = f_z
                x_opt = x_z
                z_opt = z
        else:
            continue
    return x_opt, z_opt, f_opt.item()