"""Causal imputation of irregular ICU measurements.

Two imputers, both using only the current and earlier hours of the same patient:

1. LOCF baseline: last observation carried forward; never-measured values stay missing.
2. Causal GP: each standardised variable z(t) is modelled as a stationary Gaussian process with an
   Ornstein-Uhlenbeck (Matern-1/2) kernel plus a nugget,  corr(z(t), z(t + d)) = rho0 * exp(-d / ell).
   Conditioning on the most recent measurement, taken d hours ago, gives

       mean(t) = rho0 * exp(-d / ell) * z_last,    var(t) = 1 - (rho0 * exp(-d / ell))**2,

   so the uncertainty grows from 0 at a fresh measurement towards the population variance as the last
   value ages. Never-measured variables get mean 0 (population mean) and variance 1. Because the OU process
   is Markov, the most recent measurement carries almost all the information in the past; the nugget term
   makes this an approximation rather than the exact posterior on the full history.

Hyperparameters (clip range, mean, sd, rho0, ell) are fitted per variable on training patients only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ELL_GRID = np.geomspace(0.5, 2000.0, 160)  # hours
RHO_GRID = np.linspace(0.30, 1.00, 71)


@dataclass
class OUParams:
    lo: float
    hi: float
    mean: float
    sd: float
    rho0: float
    ell: float


def _segments(patient: np.ndarray):
    starts = np.flatnonzero(np.r_[True, patient[1:] != patient[:-1]])
    return starts, np.r_[starts[1:], len(patient)]


def last_obs(values: np.ndarray, hour: np.ndarray, patient: np.ndarray):
    """Last observed value and hours since it, per row (NaN where nothing has been observed yet)."""
    v = pd.Series(values)
    t = pd.Series(np.where(np.isnan(values), np.nan, hour))
    grp = pd.Series(patient)
    last_v = v.groupby(grp).ffill().to_numpy()
    last_t = t.groupby(grp).ffill().to_numpy()
    return last_v, hour - last_t


def fit_ou(values: np.ndarray, hour: np.ndarray, patient: np.ndarray) -> OUParams:
    """Fit clip range, standardisation and (rho0, ell) from pairs of consecutive measurements."""
    obs = values[~np.isnan(values)]
    if len(obs) < 100:  # never (or almost never) measured in the training hospital: prior only
        return OUParams(-np.inf, np.inf, float(np.nanmean(obs)) if len(obs) else 0.0, 1.0, 0.0, 1.0)
    lo, hi = np.quantile(obs, [0.005, 0.995])
    x = np.clip(values, lo, hi)
    mean, sd = float(np.nanmean(x)), float(np.nanstd(x) or 1.0)
    z = (x - mean) / sd
    ok = ~np.isnan(z)
    zi, ti, pi = z[ok], hour[ok].astype(float), patient[ok]
    same = pi[1:] == pi[:-1]
    z1, z2, d = zi[:-1][same], zi[1:][same], (ti[1:] - ti[:-1])[same]
    if len(d) > 200_000:  # subsample pairs for speed
        idx = np.random.default_rng(0).choice(len(d), 200_000, replace=False)
        z1, z2, d = z1[idx], z2[idx], d[idx]
    best = (np.inf, 1.0, 24.0)
    for ell in ELL_GRID:
        decay = np.exp(-d / ell)
        a = decay * z1
        # least squares for rho0 at this ell, then clip to the grid range
        rho = float(np.clip((a @ z2) / max(a @ a, 1e-12), RHO_GRID[0], RHO_GRID[-1]))
        sse = float(((z2 - rho * a) ** 2).sum())
        if sse < best[0]:
            best = (sse, rho, float(ell))
    return OUParams(float(lo), float(hi), mean, sd, best[1], best[2])


def gp_posterior(values: np.ndarray, hour: np.ndarray, patient: np.ndarray, p: OUParams):
    """Causal GP posterior mean and sd in original units, plus hours since the last measurement."""
    x = np.clip(values, p.lo, p.hi)
    last_v, since = last_obs(x, hour.astype(float), patient)
    z_last = (last_v - p.mean) / p.sd
    k = np.where(np.isnan(since), 0.0, p.rho0 * np.exp(-np.nan_to_num(since) / p.ell))
    k = np.where(since == 0, 1.0, k)  # measured this hour: no uncertainty
    mean_z = np.where(np.isnan(z_last), 0.0, k * np.nan_to_num(z_last))
    sd_z = np.sqrt(np.clip(1.0 - k**2, 0.0, 1.0))
    return (mean_z * p.sd + p.mean).astype(np.float32), (sd_z * p.sd).astype(np.float32), since.astype(np.float32)


def draws(mean: np.ndarray, sd: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """m independent draws per row from the marginal GP posterior, shape (m, n)."""
    return (mean[None, :] + sd[None, :] * rng.standard_normal((m, len(mean)), dtype=np.float32)).astype(np.float32)
