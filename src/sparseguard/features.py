"""Feature matrices for the imputation strategies compared (all causal: current and earlier hours only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLINICAL, STATIC
from .impute import OUParams, draws, fit_ou, gp_posterior, last_obs

# the five strategies, in the order they are reported
KINDS = {
    "mean": "Mean imputation",
    "locf": "Carry-forward",
    "locf_ts": "Carry-forward + time since last",
    "gp": "Causal GP (mean + sd)",
    "gp_mi": "Causal GP, multiple imputation",
}


def fit_imputer(df: pd.DataFrame, train_mask: np.ndarray) -> dict[str, OUParams]:
    """OU hyperparameters per clinical variable, from training patients only."""
    d = df[train_mask]
    pat, hr = d.patient.to_numpy(), d.hour.to_numpy()
    return {v: fit_ou(d[v].to_numpy(np.float64), hr, pat) for v in CLINICAL}


def impute_all(df: pd.DataFrame, params: dict[str, OUParams]) -> dict[str, np.ndarray]:
    """Raw value, carry-forward value, hours since measured, GP mean and GP sd for every variable and row."""
    pat, hr = df.patient.to_numpy(), df.hour.to_numpy().astype(float)
    out = {}
    for v in CLINICAL:
        x = df[v].to_numpy(np.float64)
        locf, since = last_obs(x, hr, pat)
        mean, sd, _ = gp_posterior(x, hr, pat, params[v])
        out[f"{v}__raw"] = x.astype(np.float32)
        out[f"{v}__locf"] = locf.astype(np.float32)
        out[f"{v}__since"] = since.astype(np.float32)
        out[f"{v}__gp"] = mean
        out[f"{v}__sd"] = sd
    return out


def _context(df: pd.DataFrame) -> dict[str, np.ndarray]:
    ctx = {c: df[c].to_numpy(np.float32) for c in STATIC}
    ctx["ICULOS"] = df["ICULOS"].to_numpy(np.float32)
    return ctx


def matrix(
    df: pd.DataFrame,
    imp: dict,
    kind: str,
    rows: np.ndarray,
    gp_values: np.ndarray | None = None,
    variables=CLINICAL,
    means: dict[str, float] | None = None,
):
    """Feature matrix for the given rows.

    kind = "mean":    current value if measured this hour, else the training mean (no missingness information)
    kind = "locf":    last value carried forward; never-measured values stay missing (the trees handle NaN)
    kind = "locf_ts": carry-forward plus hours since the last measurement
    kind = "gp":      causal GP posterior mean (or a supplied draw, gp_values of shape (n_vars, n_rows)) plus GP sd
    variables: clinical variables to include (those measured in the training hospital).
    """
    cols, names = [], []
    for j, v in enumerate(variables):
        if kind == "mean":
            raw = imp[f"{v}__raw"][rows]
            cols.append(np.where(np.isnan(raw), np.float32(means[v]), raw))
            names.append(v)
        elif kind == "locf":
            cols.append(imp[f"{v}__locf"][rows])
            names.append(v)
        elif kind == "locf_ts":
            cols += [imp[f"{v}__locf"][rows], imp[f"{v}__since"][rows]]
            names += [v, f"{v}_hours_since"]
        elif kind in ("gp", "gp_mi"):
            val = imp[f"{v}__gp"][rows] if gp_values is None else gp_values[j]
            cols += [val, imp[f"{v}__sd"][rows]]
            names += [v, f"{v}_gp_sd"]
        else:
            raise ValueError(kind)
    for c, a in _context(df).items():
        cols.append(a[rows])
        names.append(c)
    return np.column_stack(cols).astype(np.float32), names


def gp_draws(imp: dict, rows: np.ndarray, m: int, rng: np.random.Generator, variables=CLINICAL) -> np.ndarray:
    """Array (m, n_vars, n_rows) of draws from the causal GP posterior."""
    out = np.empty((m, len(variables), len(rows)), dtype=np.float32)
    for j, v in enumerate(variables):
        out[:, j, :] = draws(imp[f"{v}__gp"][rows], imp[f"{v}__sd"][rows], m, rng)
    return out
