"""Discrimination, calibration and conformal experiments on the scored hours."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import conformal as C
from .config import ALPHAS, LABEL, N_SPLITS, SEED
from .data import SPARSITY_LABELS


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins (suits rare outcomes better than equal-width bins)."""
    if len(y) == 0:
        return np.nan
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    e = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def boot_ci(y, p, fn, n=200, seed=SEED, groups=None):
    """Patient-cluster bootstrap 95% CI for a metric."""
    rng = np.random.default_rng(seed)
    if groups is None:
        return np.nan, np.nan
    pats, inv = np.unique(groups, return_inverse=True)
    idx_by = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    vals = []
    for _ in range(n):
        take = np.concatenate([idx_by[i] for i in rng.integers(0, len(pats), len(pats))])
        yy, pp = y[take], p[take]
        if yy.min() == yy.max():
            continue
        vals.append(fn(yy, pp))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def discrimination(scores: pd.DataFrame, models: dict, n_boot: int = 200) -> pd.DataFrame:
    """AUROC, AUPRC, ECE and Brier per model, cohort (A_cal = same hospital, B = new hospital) and sparsity group."""
    rows = []
    cohorts = {
        "Hospital A (held-out patients)": scores.split == "A_cal",
        "Hospital B (external)": scores.hospital == "B",
    }
    for cname, cm in cohorts.items():
        for gname, gm in [("all", np.ones(len(scores), bool))] + [
            (SPARSITY_LABELS[g], (scores.group == g).to_numpy()) for g in range(3)
        ]:
            d = scores[cm.to_numpy() & gm]
            y = d[LABEL].to_numpy()
            for key, label in models.items():
                p = d[f"p_{key}"].to_numpy()
                row = {
                    "cohort": cname,
                    "group": gname,
                    "model": label,
                    "hours": len(d),
                    "sepsis_hours": int(y.sum()),
                    "auroc": roc_auc_score(y, p),
                    "auprc": average_precision_score(y, p),
                    "prevalence": float(y.mean()),
                    "ece": ece(y, p),
                    "brier": float(np.mean((p - y) ** 2)),
                    "mean_predicted": float(p.mean()),
                }
                if gname == "all" and n_boot:
                    pg = d.patient.to_numpy()
                    row["auroc_lo"], row["auroc_hi"] = boot_ci(y, p, roc_auc_score, n_boot, groups=pg)
                    row["auprc_lo"], row["auprc_hi"] = boot_ci(y, p, average_precision_score, n_boot, groups=pg)
                rows.append(row)
    return pd.DataFrame(rows)


def reliability(scores: pd.DataFrame, key: str, bins: int = 10) -> pd.DataFrame:
    """Reliability curve (equal-mass bins) per cohort and sparsity group."""
    rows = []
    for cname, cm in (("A", scores.split == "A_cal"), ("B", scores.hospital == "B")):
        for g in range(3):
            d = scores[(cm & (scores.group == g)).to_numpy()]
            p, y = d[f"p_{key}"].to_numpy(), d[LABEL].to_numpy()
            edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
            idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
            for b in range(len(edges) - 1):
                m = idx == b
                if m.any():
                    rows.append({"cohort": cname, "group": SPARSITY_LABELS[g], "bin": b, "predicted": p[m].mean(),
                                 "observed": y[m].mean(), "hours": int(m.sum())})
    return pd.DataFrame(rows)


def b_splits(scores: pd.DataFrame, n: int = N_SPLITS, seed: int = SEED):
    """n random 50/50 patient splits of hospital B into calibration and test halves."""
    pats = np.array(sorted(scores.loc[scores.hospital == "B", "patient"].unique()))
    rng = np.random.default_rng(seed)
    for k in range(n):
        perm = rng.permutation(pats)
        yield k, set(perm[: len(perm) // 2])


def conformal_experiment(scores: pd.DataFrame, score_cols: dict, alphas=ALPHAS, n: int = N_SPLITS) -> pd.DataFrame:
    """For each score, alpha, calibration source (A or B) and mode (marginal or Mondrian), evaluate on B test halves."""
    B = scores[scores.hospital == "B"].reset_index(drop=True)
    A_cal = scores[scores.split == "A_cal"]
    thr_A = {(s, a): {m: C.fit_thresholds(A_cal, s, a)} for s in score_cols for a in alphas for m in ("x",)}
    rows = []
    for k, cal_set in b_splits(scores, n):
        in_cal = B.patient.isin(cal_set).to_numpy()
        cal, test = B[in_cal], B[~in_cal]
        for s, slabel in score_cols.items():
            for a in alphas:
                thr = {"A": thr_A[(s, a)]["x"], "B": C.fit_thresholds(cal, s, a)}
                for src in ("A", "B"):
                    for mode in ("marginal", "mondrian"):
                        clear = C.apply(test, s, thr[src], mode)
                        for r in C.evaluate(test, clear):
                            rows.append({"split": k, "score": slabel, "alpha": a, "calibrated_on": f"Hospital {src}",
                                         "mode": mode, **r})
    out = pd.DataFrame(rows)
    out["group"] = out.group.map(lambda g: g if g == "all" else SPARSITY_LABELS[int(g)])
    return out


def summarise_conformal(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean and 5th-95th percentile over splits, plus how often the target was exceeded."""
    keys = ["score", "alpha", "calibrated_on", "mode", "group"]
    g = raw.groupby(keys, sort=False)
    out = g.agg(
        septic_patients=("septic_patients", "mean"),
        missed_risk_patient=("missed_risk_patient", "mean"),
        missed_p05=("missed_risk_patient", lambda v: v.quantile(0.05)),
        missed_p95=("missed_risk_patient", lambda v: v.quantile(0.95)),
        missed_rate_hour=("missed_rate_hour", "mean"),
        check_rate=("check_rate", "mean"),
        all_clear_rate=("all_clear_rate", "mean"),
    ).reset_index()
    exceed = raw.assign(over=raw.missed_risk_patient > raw.alpha).groupby(keys, sort=False).over.mean()
    out["share_of_splits_over_target"] = exceed.to_numpy()
    return out
