"""Conformal risk control for a "confident all-clear" decision.

An hour gets a confident all-clear when its risk score is below a threshold tau. The risk we control is the
share of a septic patient's sepsis hours that receive an all-clear (missed sepsis hours), averaged over septic
patients. Patients, not hours, are the exchangeable units, so the guarantee is valid even though hours of one
patient are correlated.

For a calibration set of n septic patients with losses L_i(tau) in [0, 1], non-decreasing in tau, conformal
risk control (Angelopoulos et al., 2024) picks

    tau_hat = max { tau : (n * R_n(tau) + 1) / (n + 1) <= alpha },   R_n(tau) = mean_i L_i(tau),

which guarantees E[L_new(tau_hat)] <= alpha for a new exchangeable patient. All other hours (score >= tau) are
not cleared: the model abstains and asks for a clinical check.

Marginal: one tau for everyone. Mondrian: one tau per sparsity group, with the loss computed on the sepsis
hours that fall in that group, so the guarantee holds within each group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def crc_threshold(scores: np.ndarray, patient: np.ndarray, alpha: float) -> float:
    """Largest tau with (n * R_n(tau) + 1) / (n + 1) <= alpha, from the sepsis hours of the calibration set.

    scores, patient: score and patient id of every sepsis hour (positive hour) in the calibration set.
    Returns 0.0 (nobody gets an all-clear) when no threshold satisfies the bound.
    """
    if len(scores) == 0:
        return 0.0
    pats, inv, counts = np.unique(patient, return_inverse=True, return_counts=True)
    n = len(pats)
    w = 1.0 / (counts[inv] * n)  # each patient contributes total weight 1/n, split over its sepsis hours
    order = np.argsort(scores, kind="stable")
    s, cw = scores[order], np.cumsum(w[order])
    # candidate thresholds: each distinct score (all strictly smaller scores are cleared) and +inf
    uniq, first = np.unique(s, return_index=True)
    risk_at = np.r_[0.0, cw][first]  # R_n(tau = uniq[k]) = weight of scores strictly below uniq[k]
    cand = np.r_[uniq, np.inf]
    risk = np.r_[risk_at, cw[-1]]
    ok = (n * risk + 1) / (n + 1) <= alpha
    if not ok.any():
        return 0.0
    return float(cand[np.flatnonzero(ok).max()])


def fit_thresholds(cal: pd.DataFrame, score: str, alpha: float, label: str = "SepsisLabel") -> dict:
    """Marginal threshold and one Mondrian threshold per sparsity group."""
    pos = cal[cal[label] == 1]
    out = {"marginal": crc_threshold(pos[score].to_numpy(), pos.patient.to_numpy(), alpha)}
    for g in sorted(cal.group.unique()):
        pg = pos[pos.group == g]
        out[int(g)] = crc_threshold(pg[score].to_numpy(), pg.patient.to_numpy(), alpha)
    return out


def apply(test: pd.DataFrame, score: str, thr: dict, mode: str) -> np.ndarray:
    """Boolean all-clear decision per hour."""
    s = test[score].to_numpy()
    if mode == "marginal":
        return s < thr["marginal"]
    t = test.group.map({k: v for k, v in thr.items() if k != "marginal"}).to_numpy(dtype=float)
    return s < t


def evaluate(test: pd.DataFrame, clear: np.ndarray, label: str = "SepsisLabel") -> list[dict]:
    """Missed-sepsis risk (patient-averaged, the guaranteed quantity), hour-level miss rate and check rate."""
    rows = []
    t = test.assign(clear=clear)
    for g, d in [("all", t)] + [(int(k), v) for k, v in t.groupby("group")]:
        pos, neg = d[d[label] == 1], d[d[label] == 0]
        per_pat = pos.groupby("patient").clear.mean()
        rows.append(
            {
                "group": g,
                "septic_patients": int(len(per_pat)),
                "sepsis_hours": int(len(pos)),
                "missed_risk_patient": float(per_pat.mean()) if len(per_pat) else np.nan,
                "missed_rate_hour": float(pos.clear.mean()) if len(pos) else np.nan,
                "check_rate": float(1 - neg.clear.mean()) if len(neg) else np.nan,  # non-sepsis hours not cleared
                "all_clear_rate": float(d.clear.mean()),
            }
        )
    return rows
