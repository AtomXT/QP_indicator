"""
Test original MIQP vs. tree-based MISOCP reformulation (perspective on nodes/edges).

Problem:
    min_{x,z} 0.5 x^T Q x + c^T x + lambda^T z
    s.t.      x_i (1 - z_i) = 0,  z_i in {0,1}

Assumption for this script:
- The sparsity graph of Q is a TREE.
- We generate Q in a tree-Laplacian+diag form so we can decompose exactly into:
      0.5 x^T Q x = sum_i (0.5 delta_i x_i^2) + sum_{(i,j)} (0.5 a_ij (x_i - x_j)^2)
  where a_ij > 0 and delta_i >= 0.
  This yields a clean edge-local perspective MISOCP formulation.

Requires: gurobipy, numpy, networkx
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB


@dataclass
class TreeQPData:
    n: int
    edges: List[Tuple[int, int]]                 # 0-indexed
    a: Dict[Tuple[int, int], float]              # edge weights a_ij > 0 on undirected edges (i<j)
    delta: np.ndarray                            # node residuals delta_i >= 0
    Q: np.ndarray                                # PSD, tree-sparse
    c: np.ndarray
    lam: np.ndarray
    M: np.ndarray                                # bounds for |x_i| <= M_i z_i


def make_tree_qp(
    n: int,
    seed: int = 0,
    edge_w_lo: float = 0.5,
    edge_w_hi: float = 2.0,
    delta_lo: float = 0.0,
    delta_hi: float = 1.0,
    c_scale: float = 1.0,
    lam_lo: float = 0.2,
    lam_hi: float = 1.0,
    M: float = 10.0,
) -> TreeQPData:
    rng = np.random.default_rng(seed)

    # random tree on n nodes
    T = nx.random_tree(n, seed=seed)
    edges = []
    a = {}
    for (i, j) in T.edges():
        i0, j0 = (i, j) if i < j else (j, i)
        edges.append((i0, j0))
        a[(i0, j0)] = float(rng.uniform(edge_w_lo, edge_w_hi))

    delta = rng.uniform(delta_lo, delta_hi, size=n)

    # Build Q = Laplacian(a) + diag(delta)
    Q = np.zeros((n, n), dtype=float)
    for (i, j) in edges:
        w = a[(i, j)]
        # Laplacian contribution: w*(e_i - e_j)(e_i - e_j)^T
        Q[i, i] += w
        Q[j, j] += w
        Q[i, j] -= w
        Q[j, i] -= w
    Q[np.diag_indices(n)] += delta

    # Linear and lambda
    c = rng.normal(0.0, c_scale, size=n)
    lam = rng.uniform(lam_lo, lam_hi, size=n)

    Mvec = np.full(n, M, dtype=float)
    return TreeQPData(n=n, edges=edges, a=a, delta=delta, Q=Q, c=c, lam=lam, M=Mvec)


def solve_original_miqp(data: TreeQPData, timelimit: float | None = None, threads: int = 0) -> Dict:
    n = data.n
    Q = data.Q
    c = data.c
    lam = data.lam
    M = data.M

    m = gp.Model("original_miqp")
    m.Params.OutputFlag = 0
    if timelimit is not None:
        m.Params.TimeLimit = timelimit
    if threads:
        m.Params.Threads = threads

    x = m.addMVar(n, lb=-GRB.INFINITY, name="x")
    z = m.addMVar(n, vtype=GRB.BINARY, name="z")

    # x_i(1-z_i)=0 implemented as |x_i| <= M_i z_i
    m.addConstr(x <= M * z, name="ub_x")
    m.addConstr(-x <= M * z, name="lb_x")

    # Objective: 0.5 x'Qx + c'x + lam'z
    obj = 0.5 * x @ (Q @ x) + c @ x + lam @ z
    m.setObjective(obj, GRB.MINIMIZE)

    t0 = time.time()
    m.optimize()
    t1 = time.time()

    x_sol = x.X
    z_sol = z.X
    return {
        "status": int(m.Status),
        "obj": float(m.ObjVal) if m.SolCount else None,
        "runtime": t1 - t0,
        "mipgap": float(m.MIPGap) if m.SolCount and m.IsMIP else None,
        "x": x_sol if m.SolCount else None,
        "z": z_sol if m.SolCount else None,
    }


def solve_tree_misocp_reform(data: TreeQPData, timelimit: float | None = None, threads: int = 0) -> Dict:
    """
    Reformulation using exact decomposition:
      0.5 x^T Q x = sum_i 0.5*delta_i x_i^2 + sum_{(i,j)} 0.5*a_ij (x_i - x_j)^2

    Use edge indicators w_ij = z_i z_j (exact hull on a tree) and perspective constraints:
      delta_i * x_i^2 <= 2 u_i z_i      => u_i >= 0.5 delta_i x_i^2 when z_i=1
      a_ij*(x_i - x_j)^2 <= 2 v_ij w_ij => v_ij >= 0.5 a_ij (x_i - x_j)^2 when w_ij=1

    Objective:
      sum_i u_i + sum_{(i,j)} v_ij + c^T x + lam^T z
    """
    n = data.n
    edges = data.edges
    a = data.a
    delta = data.delta
    c = data.c
    lam = data.lam
    M = data.M

    m = gp.Model("tree_misocp_reform")
    m.Params.OutputFlag = 0
    if timelimit is not None:
        m.Params.TimeLimit = timelimit
    if threads:
        m.Params.Threads = threads

    x = m.addMVar(n, lb=-GRB.INFINITY, name="x")
    z = m.addMVar(n, vtype=GRB.BINARY, name="z")

    # keep same on/off modeling for x: |x_i| <= M_i z_i
    m.addConstr(x <= M * z, name="ub_x")
    m.addConstr(-x <= M * z, name="lb_x")

    # node epigraphs
    u = m.addMVar(n, lb=0.0, name="u")  # u_i >= 0.5 delta_i x_i^2 when z_i=1
    for i in range(n):
        if delta[i] > 0:
            # delta_i * x_i^2 <= 2 u_i z_i (convex if z_i >=0; z_i binary is fine)
            m.addConstr(delta[i] * x[i] * x[i] <= 2.0 * u[i] * z[i], name=f"node_persp[{i}]")
        else:
            # if delta_i = 0 then u_i can be 0
            m.addConstr(u[i] == 0.0, name=f"node_zero[{i}]")

    # edge indicators (OR) and epigraphs
    y = {}
    v = {}
    for (i, j) in edges:
        yij = m.addVar(vtype=GRB.BINARY, name=f"y[{i},{j}]")  # OR indicator
        vij = m.addVar(lb=0.0, name=f"v[{i},{j}]")
        y[(i, j)] = yij
        v[(i, j)] = vij

        # yij = z_i OR z_j
        m.addConstr(yij >= z[i], name=f"y_lb1[{i},{j}]")
        m.addConstr(yij >= z[j], name=f"y_lb2[{i},{j}]")
        m.addConstr(yij <= z[i] + z[j], name=f"y_ub[{i},{j}]")

        # a_ij * (x_i - x_j)^2 <= 2 v_ij * y_ij
        aij = a[(i, j)]
        m.addConstr(aij * (x[i] - x[j]) * (x[i] - x[j]) <= 2.0 * vij * yij,
                    name=f"edge_persp[{i},{j}]")

    obj = u.sum() + gp.quicksum(vij for vij in v.values()) + c @ x + lam @ z

    m.setObjective(obj, GRB.MINIMIZE)

    t0 = time.time()
    m.optimize()
    t1 = time.time()

    x_sol = x.X
    z_sol = z.X
    return {
        "status": int(m.Status),
        "obj": float(m.ObjVal) if m.SolCount else None,
        "runtime": t1 - t0,
        "mipgap": float(m.MIPGap) if m.SolCount and m.IsMIP else None,
        "x": x_sol if m.SolCount else None,
        "z": z_sol if m.SolCount else None,
    }


def solve_miqp_g_hull(data, timelimit=None, mip_gap=None, threads=None, verbose=False):
    """
    Solve the MIQP using the g-based 3-way convex-hull (Balas) reformulation:

      For each i, enforce (x_i, g_i) ∈ D_i^0 ∪ D_i^+ ∪ D_i^- via an ideal hull formulation:
        D_i^0: x_i = 0, |g_i| ≤ tau_i
        D_i^+: Q_ii x_i + g_i = 0, g_i ≥ tau_i
        D_i^-: Q_ii x_i + g_i = 0, g_i ≤ -tau_i
      with g_i = c_i + sum_{j!=i} Q_ij x_j  (off-diagonal field)

    Requires: gurobipy

    Parameters
    ----------
    Q : (n,n) array_like, symmetric PSD recommended. Needs Q_ii > 0 for all i.
    c : (n,) array_like
    lam : (n,) array_like, assumed >= 0
    xbar : float or (n,) array_like, bound |x_i| <= xbar_i (required to bound g and make hull closed)
    time_limit : float, optional (seconds)
    mip_gap : float, optional (relative MIP gap)
    verbose : bool

    Returns
    -------
    result : dict with keys:
        status, obj, x, z_plus, z_minus, z0
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ImportError("This function requires gurobipy (Gurobi Python API).") from e

    n = data.n
    Q = data.Q
    c = data.c
    lam = data.lam
    M = data.M
    xbar = M

    # neighbor sets from off-diagonal nonzeros
    # (tree structure just makes these small; code works for any sparsity)
    N = [[] for _ in range(n)]
    for i in range(n):
        # include j != i with Q_ij != 0
        nz = np.nonzero(Q[i, :])[0]
        for j in nz:
            if j != i and Q[i, j] != 0.0:
                N[i].append(j)

    # constants for hull
    tau = np.sqrt(2.0 * lam * np.diag(Q))
    # valid |g_i| bound from |x_j|<=xbar_j
    G = np.abs(c).copy()
    for i in range(n):
        if N[i]:
            G[i] += np.sum(np.abs(Q[i, list(N[i])]) * xbar[N[i]])

    m = gp.Model("g_hull_miqp")
    m.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)
    if threads:
        m.Params.Threads = threads
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    # Aggregate variables
    x = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    g = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g")

    # Disaggregated per-regime variables
    x0 = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x0")
    xp = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xp")
    xm = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="xm")

    g0 = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g0")
    gpv = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="gplus")
    gmv = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="gminus")

    # Regime binaries
    z0 = m.addVars(n, vtype=GRB.BINARY, name="z0")
    zp = m.addVars(n, vtype=GRB.BINARY, name="zplus")
    zm = m.addVars(n, vtype=GRB.BINARY, name="zminus")

    # Bounds on aggregate x (required)
    for i in range(n):
        m.addConstr(x[i] <= xbar[i], name=f"x_ub[{i}]")
        m.addConstr(x[i] >= -xbar[i], name=f"x_lb[{i}]")

    # g definition and optional explicit g bounds
    for i in range(n):
        expr = c[i]
        if N[i]:
            expr += gp.quicksum(Q[i, j] * x[j] for j in N[i])
        # off-diagonal sum: we excluded j=i already
        m.addConstr(g[i] == expr, name=f"g_def[{i}]")
        m.addConstr(g[i] <= G[i], name=f"g_ub[{i}]")
        m.addConstr(g[i] >= -G[i], name=f"g_lb[{i}]")

    # Hull coupling: x = x0+xp+xm, g = g0+gplus+gminus, select regime
    for i in range(n):
        m.addConstr(x[i] == x0[i] + xp[i] + xm[i], name=f"disagg_x[{i}]")
        m.addConstr(g[i] == g0[i] + gpv[i] + gmv[i], name=f"disagg_g[{i}]")
        m.addConstr(z0[i] + zp[i] + zm[i] == 1, name=f"regime_sum[{i}]")

        # Off piece: x0 = 0, |g0| <= tau_i z0
        m.addConstr(x0[i] == 0, name=f"off_x0[{i}]")
        m.addConstr(g0[i] <= tau[i] * z0[i], name=f"off_g0_ub[{i}]")
        m.addConstr(g0[i] >= -tau[i] * z0[i], name=f"off_g0_lb[{i}]")

        # On+ piece: Q_ii xp + gplus = 0, tau z+ <= gplus <= G z+
        Qii = Q[i, i]
        m.addConstr(Qii * xp[i] + gpv[i] == 0, name=f"onp_stat[{i}]")
        m.addConstr(gpv[i] >= tau[i] * zp[i], name=f"onp_g_lb[{i}]")
        m.addConstr(gpv[i] <= G[i] * zp[i], name=f"onp_g_ub[{i}]")

        # On- piece: Q_ii xm + gminus = 0, -G z- <= gminus <= -tau z-
        m.addConstr(Qii * xm[i] + gmv[i] == 0, name=f"onm_stat[{i}]")
        m.addConstr(gmv[i] >= -G[i] * zm[i], name=f"onm_g_lb[{i}]")
        m.addConstr(gmv[i] <= -tau[i] * zm[i], name=f"onm_g_ub[{i}]")

        # (Optional but often helps) also bound disaggregated vars tightly
        # These follow from x bounds and disaggregation; keeps numerics stable.
        m.addConstr(xp[i] <= xbar[i] * zp[i], name=f"xp_ub[{i}]")
        m.addConstr(xp[i] >= -xbar[i] * zp[i], name=f"xp_lb[{i}]")
        m.addConstr(xm[i] <= xbar[i] * zm[i], name=f"xm_ub[{i}]")
        m.addConstr(xm[i] >= -xbar[i] * zm[i], name=f"xm_lb[{i}]")
        # x0 is fixed 0 so no need for bounds there.

    # Objective: 1/2 x^T Q x + c^T x + sum lambda_i (zplus+zminus)
    # Build quadratic form explicitly
    obj = gp.QuadExpr()
    # 0.5 * sum_{i,j} Q_ij x_i x_j
    for i in range(n):
        # diagonal
        obj.add(0.5 * Q[i, i] * x[i] * x[i])
        # off-diagonal (i<j)
        for j in range(i + 1, n):
            if Q[i, j] != 0.0:
                obj.add(Q[i, j] * x[i] * x[j])
    # linear term c^T x
    obj.add(gp.LinExpr(c.tolist(), [x[i] for i in range(n)]))
    # penalty
    obj.add(gp.quicksum(lam[i] * (zp[i] + zm[i]) for i in range(n)))

    m.setObjective(obj, GRB.MINIMIZE)
    t0 = time.time()
    m.optimize()
    t1 = time.time()

    status = m.Status
    if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        return {"status": status, "obj": None, "x": None, "z_plus": None, "z_minus": None, "z0": None}

    x_sol = np.array([x[i].X for i in range(n)])
    zp_sol = np.array([int(round(zp[i].X)) for i in range(n)])
    zm_sol = np.array([int(round(zm[i].X)) for i in range(n)])
    z0_sol = np.array([int(round(z0[i].X)) for i in range(n)])
    return {
        "status": status,
        "obj": float(m.ObjVal),
        "x": x_sol,
        "z_plus": zp_sol,
        "z_minus": zm_sol,
        "z": zp_sol+zm_sol,
        "z0": z0_sol,
        "runtime": t1-t0,
        "mipgap": float(m.MIPGap) if m.SolCount and m.IsMIP else None,
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timelimit", type=float, default=10)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--M", type=float, default=100.0)
    args = parser.parse_args()

    data = make_tree_qp(n=args.n, seed=args.seed, M=args.M, lam_lo=0.5, lam_hi=1)

    print(f"n={data.n}, |E|={len(data.edges)} (tree), seed={args.seed}")
    print("Solving original MIQP...")
    res1 = solve_original_miqp(data, timelimit=args.timelimit, threads=args.threads)
    print(f"  status={res1['status']}, obj={res1['obj']}, time={res1['runtime']:.3f}s, gap={res1['mipgap']}")

    print("Solving tree MISOCP reformulation...")
    res2 = solve_miqp_g_hull(data, timelimit=args.timelimit, threads=args.threads)
    print(f"  status={res2['status']}, obj={res2['obj']}, time={res2['runtime']:.3f}s, gap={res2['mipgap']}")

    if res1["obj"] is not None and res2["obj"] is not None:
        diff = abs(res1["obj"] - res2["obj"])
        rel = diff / (1.0 + abs(res1["obj"]))
        print(f"Objective diff: abs={diff:.6e}, rel={rel:.6e}")

        # sanity: solutions satisfy x=0 when z=0 (approximately)
        x1, z1 = res1["x"], res1["z"]
        x2, z2 = res2["x"], res2["z"]
        if x1 is not None:
            viol1 = np.max(np.abs(x1) * (1 - np.round(z1)))
            print(f"Original max |x_i|(1-z_i): {viol1:.3e}")
        if x2 is not None:
            viol2 = np.max(np.abs(x2) * (1 - np.round(z2)))
            print(f"Reform max |x_i|(1-z_i): {viol2:.3e}")
        if x1 is not None and x2 is not None:
            print(f"max difference between two solutions: {np.max(np.abs(x1-x2))}")
        print(f"Total number of nonzero in z1: {np.sum(z1)}")


if __name__ == "__main__":
    main()
