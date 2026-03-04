#!/usr/bin/env python3
"""
Run experiments for cardinality-constrained Best Subset Selection (BSS):

Original MIQP:
    min_{beta,z}  1/2 beta^T G beta - c^T beta + (gamma/2)||beta||^2 (+ const)
    s.t.          -bar_beta_i z_i <= beta_i <= bar_beta_i z_i
                  sum_i z_i <= k
                  z_i in {0,1}

CORe (2-way disjunction) formulation (your Eq. (reformulated_card) style):
    For each i, either:
      D_i^0: beta_i = 0  (off)
      D_i^1: (G beta)_i + gamma beta_i = c_i (on)
    encoded with disaggregation (beta^0,beta^1,g^0,g^1,z^0,z^1) and bounds H_i.

Dataset loader expected:
    load_instance_bss(p, n, rep, project_root) -> X,y,G,c,beta_star,support,beta_ridge,bar_beta,H,meta

Outputs:
    ../experiments_results/BSS_{job_name}.csv
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# You should have this in your project already (from earlier)
# from src.utils import load_instance_bss
# If not, paste your load_instance_bss into src/utils.py and import it here.
from src.utils import load_instance_bss

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, ".."))

np.set_printoptions(linewidth=200)

# -----------------------------
# Args / parsing
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--p_list", type=str, default="500",
                   help='e.g. "500" or "500,1000" or "[500,1000]"')
    p.add_argument("--n_list", type=str, default="250,1000",
                   help='e.g. "250" or "250,1000" or "[250,1000]"')
    p.add_argument("--rep_list", type=str, default="0,1,2,3,4",
                   help='e.g. "0" or "0,1,2" or "[0,1,2]"')
    p.add_argument("--k_list", type=str, default="5,10,15,20,25,30,40,50,75,100",
                   help='e.g. "10" or "10,20,50" or "[10,20,50]"')

    p.add_argument("--timelimit", type=float, default=600.0)
    p.add_argument("--mipgap", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=8)

    p.add_argument("--job_name", type=str, default="temp",
                   help="Name of this job (used as output csv filename)")
    p.add_argument("--verbose", type=int, default=1)

    return p.parse_args()


def parse_list(s: str, cast=float):
    """
    Accepts:
      "1,2,3"
      "[1,2,3]"
      "1"
    """
    s = s.strip()
    if s.startswith("["):
        arr = json.loads(s)
        return [cast(x) for x in arr]
    if "," in s:
        return [cast(x) for x in s.split(",") if x.strip() != ""]
    return [cast(s)]


# -----------------------------
# Root bound callback
# -----------------------------
root_bound = [np.inf, -np.inf]  # [best_ub, best_lb]

def record_root_bounds(model, where):
    if where == GRB.Callback.MIPNODE:
        nodecnt = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
        if nodecnt == 0:
            lb = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
            ub = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
            if lb >= root_bound[1]:
                root_bound[1] = lb
            if ub <= root_bound[0]:
                root_bound[0] = ub


def safe_gap(ub: float, lb: float) -> float:
    if not np.isfinite(ub) or ub == 0:
        return np.nan
    return (ub - lb) / ub


# -----------------------------
# Model builders
# -----------------------------

def build_relax_bss(G: np.ndarray, c: np.ndarray, gamma: float, bar_beta: np.ndarray, k: int, y:np.ndarray,
                       timelimit=None, mipgap=None, threads=None, verbose=False) -> gp.Model:
    """
    Original MIQP with big-M bounds using bar_beta:
        -bar_beta_i z_i <= beta_i <= bar_beta_i z_i
        sum z <= k
    Objective: 0.5 beta^T G beta - c^T beta + (gamma/2)||beta||^2
    """
    p = G.shape[0]
    n = len(y)
    m = gp.Model("bss_original")
    m.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)
    if threads is not None:
        m.Params.Threads = int(threads)
    if mipgap is not None:
        m.Params.MIPGap = float(mipgap)

    beta = m.addMVar(p, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta")
    z = m.addMVar(p, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z")

    # Linking bounds
    m.addConstr(beta <= bar_beta * z, name="beta_ub")
    m.addConstr(beta >= -bar_beta * z, name="beta_lb")
    m.addConstr(z.sum() <= k, name="card")

    # Objective
    G = G / n
    c = c / n
    G[np.abs(G) < 0.1] = 0
    G += np.diag(np.abs(G).sum(axis=0))
    Q = G + gamma * np.eye(p)
    obj = 0.5 * beta @ Q @ beta - c @ beta + 0.5 * float(y @ y)
    m.setObjective(obj, GRB.MINIMIZE)

    # Keep handles
    m._beta = beta
    m._z = z
    return m
def build_original_bss(G: np.ndarray, c: np.ndarray, gamma: float, bar_beta: np.ndarray, k: int, y:np.ndarray,
                       timelimit=None, mipgap=None, threads=None, verbose=False) -> gp.Model:
    """
    Original MIQP with big-M bounds using bar_beta:
        -bar_beta_i z_i <= beta_i <= bar_beta_i z_i
        sum z <= k
    Objective: 0.5 beta^T G beta - c^T beta + (gamma/2)||beta||^2
    """
    p = G.shape[0]
    n = len(y)
    m = gp.Model("bss_original")
    m.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)
    if threads is not None:
        m.Params.Threads = int(threads)
    if mipgap is not None:
        m.Params.MIPGap = float(mipgap)

    beta = m.addMVar(p, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta")
    z = m.addMVar(p, vtype=GRB.BINARY, name="z")

    # Linking bounds
    m.addConstr(beta <= bar_beta * z, name="beta_ub")
    m.addConstr(beta >= -bar_beta * z, name="beta_lb")
    m.addConstr(z.sum() <= k, name="card")

    # Objective
    G = G/n
    c = c/n
    G[np.abs(G) < 0.1] = 0
    G += np.diag(np.abs(G).sum(axis=0))
    Q = G + gamma * np.eye(p)
    obj = 0.5 * beta @ Q @ beta - c @ beta + 0.5 * float(y @ y)/n
    m.setObjective(obj, GRB.MINIMIZE)

    # Keep handles
    m._beta = beta
    m._z = z
    return m


def build_core_bss(G: np.ndarray, c: np.ndarray, gamma: float, bar_beta: np.ndarray, H: np.ndarray, k: int, y:np.ndarray,
                   timelimit=None, mipgap=None, threads=None, verbose=False) -> gp.Model:
    """
    CORe disjunctive formulation (2-way) using disaggregation:
      beta = beta0 + beta1
      g = g0 + g1
      z0 + z1 = 1
      D0: beta0_i = 0,  -H_i z0_i <= g0_i <= H_i z0_i
      D1: (G beta1)_i + gamma beta1_i = c_i z1_i,  -H_i z1_i <= g1_i <= H_i z1_i
          plus bounds beta1_i within bar_beta_i z1_i (helps numerics / keeps hull bounded)

    Also enforce:
      g_i = sum_{j!=i} G_ij beta_j - c_i   (definition)
    """
    p = G.shape[0]
    n = len(y)
    G = G/n
    c = c/n
    G[np.abs(G) < 0.1] = 0
    G += np.diag(np.abs(G).sum(axis=0))
    Q = G + gamma * np.eye(p)
    Qabs = np.abs(Q)
    np.fill_diagonal(Qabs, 0)  # optional
    H = np.abs(c) + Qabs @ bar_beta

    m = gp.Model("bss_core")
    m.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)
    if threads is not None:
        m.Params.Threads = int(threads)
    if mipgap is not None:
        m.Params.MIPGap = float(mipgap)

    # Aggregate
    beta = m.addMVar(p, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta")
    g = m.addMVar(p, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g")

    # Disaggregated
    beta0 = m.addMVar(p, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta0")
    beta1 = m.addMVar(p, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta1")
    g0 = m.addMVar(p, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g0")
    g1 = m.addMVar(p, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="g1")

    z0 = m.addMVar(p, vtype=GRB.BINARY, name="z0")
    z1 = m.addMVar(p, vtype=GRB.BINARY, name="z1")

    # Coupling
    m.addConstr(beta == beta0 + beta1, name="disagg_beta")
    m.addConstr(g == g0 + g1, name="disagg_g")
    m.addConstrs(z0[i] + z1[i] == 1 for i in range(p))

    # Cardinality on z1
    m.addConstr(z1.sum() <= k, name="card")

    # g definition: g_i = sum_{j!=i} G_ij beta_j - c_i
    # Implement as: g = (G beta) - diag(G)*beta - c
    diagG = np.diag(G)
    # G @ beta is an MVar expression
    m.addConstr(g == (G @ beta) - (diagG * beta) - c, name="g_def")
    # m.addConstrs(g[i] == gp.quicksum(G[i,j]*beta[j] for j in range(p) if j != i) - c[i] for i in range(p))

    # D0: beta0 = 0
    m.addConstr(beta0 == 0, name="D0_beta0_zero")
    # g0 bounded by H * z0
    m.addConstr(g0 <= H * z0, name="D0_g0_ub")
    m.addConstr(g0 >= -H * z0, name="D0_g0_lb")

    # D1: (G beta1)_i + gamma beta1_i = c_i z1_i
    # Note: this uses full G beta1 (including diagonal), matching your normal equation form.
    # m.addConstr((G @ beta1) + gamma * beta1 == c * z1, name="D1_normal_eq")
    m.addConstrs(beta1[i] == -g1[i]/(G[i,i]+gamma) for i in range(p))
    # g1 bounded by H * z1
    m.addConstr(g1 <= H * z1, name="D1_g1_ub")
    m.addConstr(g1 >= -H * z1, name="D1_g1_lb")
    # Bound beta1 when z1=0 to keep the hull well-behaved
    m.addConstr(beta1 <= bar_beta * z1, name="D1_beta1_ub")
    m.addConstr(beta1 >= -bar_beta * z1, name="D1_beta1_lb")

    # Objective in aggregated beta

    obj = 0.5 * beta @ Q @ beta - c @ beta + 0.5 * float(y @ y)/n
    m.setObjective(obj, GRB.MINIMIZE)

    # Keep handles
    m._beta = beta
    m._z1 = z1
    return m


# -----------------------------
# Main experiment loop
# -----------------------------
def main():
    args = parse_args()

    p_list = parse_list(args.p_list, int)
    n_list = parse_list(args.n_list, int)
    rep_list = parse_list(args.rep_list, int)
    k_list = parse_list(args.k_list, int)

    timelimit = float(args.timelimit)
    mipgap = float(args.mipgap)
    threads = int(args.threads)
    verbose = bool(args.verbose)

    job_name = args.job_name
    results = []

    out_csv = os.path.join(current_dir, "..", "experiments_results", f"BSS_{job_name}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    for p in p_list:
        for n in n_list:
            for rep in rep_list:
                # Load one dataset (no k inside)
                X, y, G, c, beta_star, support, beta_ridge, bar_beta, H, meta = load_instance_bss(
                    p=p, n=n, rep=rep, project_root=project_root
                )

                # Sweep k on the same data
                for k in k_list:
                    if k < 1 or k > p:
                        continue

                    try:
                        if verbose:
                            print(f"[p={p}, n={n}, rep={rep}, k={k}]")

                        m_relax = build_relax_bss(
                            G=G, c=c, gamma=meta["gamma"], bar_beta=bar_beta, k=k, y=y,
                            timelimit=timelimit, mipgap=mipgap, threads=threads, verbose=verbose
                        )
                        m_relax.optimize()
                        print(f"The relaxed obj is {m_relax.objVal}.")
                        x_relax_vals = np.array([m_relax._beta[i].X for i in range(p)])
                        print(f"The larges value of x is {max(abs(x_relax_vals))}")
                        BIG_M = min(1000, 2 * max(abs(x_relax_vals)))
                        print(f'Use new Big-M {BIG_M}.')
                        bar_beta = BIG_M*np.ones(p)
                        # ---------- Original ----------
                        global root_bound
                        root_bound = [np.inf, -np.inf]



                        m_ori = build_original_bss(
                            G=G, c=c, gamma=meta["gamma"], bar_beta=bar_beta, k=k, y=y,
                            timelimit=timelimit, mipgap=mipgap, threads=threads, verbose=verbose
                        )
                        t0 = time.time()
                        m_ori.optimize(record_root_bounds)
                        t_ori = time.time() - t0

                        # Collect
                        z_vals = None
                        support_size = np.nan
                        end_ub = np.nan
                        end_lb = np.nan
                        if m_ori.SolCount > 0:
                            z_vals = m_ori._z.X
                            support_size = int(np.count_nonzero(z_vals > 0.5))
                            end_ub = float(m_ori.ObjVal)
                        end_lb = float(m_ori.ObjBound) if np.isfinite(m_ori.ObjBound) else np.nan

                        r = [
                            p, n, rep, k, "original",
                            float(root_bound[0]) if np.isfinite(root_bound[0]) else np.nan,
                            float(root_bound[1]) if np.isfinite(root_bound[1]) else np.nan,
                            safe_gap(root_bound[0], root_bound[1]),
                            end_ub,
                            end_lb,
                            safe_gap(end_ub, end_lb) if np.isfinite(end_ub) and np.isfinite(end_lb) else np.nan,
                            support_size,
                            float(m_ori.NodeCount),
                            float(m_ori.Runtime),
                            int(m_ori.Status),
                        ]
                        results.append(r)

                        # ---------- CORe ----------
                        root_bound = [np.inf, -np.inf]

                        m_cor = build_core_bss(
                            G=G, c=c, gamma=meta["gamma"], bar_beta=bar_beta, H=H, k=k, y=y,
                            timelimit=timelimit, mipgap=mipgap, threads=threads, verbose=verbose
                        )
                        t0 = time.time()
                        m_cor.optimize(record_root_bounds)
                        t_cor = time.time() - t0

                        z1_vals = None
                        support_size = np.nan
                        end_ub = np.nan
                        end_lb = np.nan
                        if m_cor.SolCount > 0:
                            z1_vals = m_cor._z1.X
                            support_size = int(np.count_nonzero(z1_vals > 0.5))
                            end_ub = float(m_cor.ObjVal)
                        end_lb = float(m_cor.ObjBound) if np.isfinite(m_cor.ObjBound) else np.nan

                        r = [
                            p, n, rep, k, "core",
                            float(root_bound[0]) if np.isfinite(root_bound[0]) else np.nan,
                            float(root_bound[1]) if np.isfinite(root_bound[1]) else np.nan,
                            safe_gap(root_bound[0], root_bound[1]),
                            end_ub,
                            end_lb,
                            safe_gap(end_ub, end_lb) if np.isfinite(end_ub) and np.isfinite(end_lb) else np.nan,
                            support_size,
                            float(m_cor.NodeCount),
                            float(m_cor.Runtime),
                            int(m_cor.Status),
                        ]
                        results.append(r)

                        # Save incrementally
                        df = pd.DataFrame(
                            results,
                            columns=[
                                "p", "n", "rep", "k", "formulation",
                                "root_ub", "root_lb", "root_gap",
                                "end_ub", "end_lb", "end_gap",
                                "support_size", "node_count", "time", "status"
                            ],
                        )
                        df.to_csv(out_csv, index=False)

                    except Exception as e:
                        print(f"Error at p={p}, n={n}, rep={rep}, k={k}: {e}")
                        continue

    print(f"Done. Results saved to: {out_csv}")


if __name__ == "__main__":
    main()