"""
Bradley-Terry scoring from pairwise comparison records.
"""

import numpy as np
from scipy.optimize import minimize


def compute_bt_scores(n: int, comparisons: list[dict], lam: float = 0.01) -> list[float]:
    """
    Compute BT scores for n solutions given comparison records.
    comparisons: list of {idx_a, idx_b, winner}  (winner in "A","B","TIE")
    Returns list of floats (higher = better), z-scored.
    """
    if not comparisons:
        return [0.0] * n

    ia_list, ib_list, out_list = [], [], []
    for c in comparisons:
        ia, ib = c["idx_a"], c["idx_b"]
        w = c["winner"]
        if w == "A":
            out = 1.0
        elif w == "B":
            out = 0.0
        else:  # TIE
            out = 0.5
        ia_list.append(ia)
        ib_list.append(ib)
        out_list.append(out)

    ia_arr = np.array(ia_list, dtype=int)
    ib_arr = np.array(ib_list, dtype=int)
    oa_arr = np.array(out_list, dtype=float)

    def nll_grad(s):
        diff = np.clip(s[ia_arr] - s[ib_arr], -500, 500)
        p = 1.0 / (1.0 + np.exp(-diff))
        nll = -np.sum(
            oa_arr * np.log(np.clip(p, 1e-10, 1)) +
            (1 - oa_arr) * np.log(np.clip(1 - p, 1e-10, 1))
        )
        g = lam * s.copy()
        err = p - oa_arr
        np.add.at(g, ia_arr,  err)
        np.add.at(g, ib_arr, -err)
        return float(nll + 0.5 * lam * np.dot(s, s)), g

    res = minimize(nll_grad, np.zeros(n), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    s = res.x
    std = s.std()
    if std > 1e-6:
        s = (s - s.mean()) / std
    else:
        s = s - s.mean()
    return s.tolist()


def rank_solutions(scores: list[float]) -> list[int]:
    """Return indices sorted best-first."""
    return sorted(range(len(scores)), key=lambda i: -scores[i])
