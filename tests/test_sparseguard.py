"""Unit tests for SparseGuard: causal imputation, no look-ahead, patient-level splits and the conformal guarantee."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from sparseguard import conformal as C  # noqa: E402
from sparseguard.config import LABS  # noqa: E402
from sparseguard.data import lab_count_past, split_patients  # noqa: E402
from sparseguard.evaluate import ece  # noqa: E402
from sparseguard.impute import OUParams, gp_posterior, last_obs  # noqa: E402


def test_gp_uncertainty_grows_with_time_since_measurement():
    hour = np.arange(10, dtype=float)
    vals = np.array([5.0] + [np.nan] * 9)
    p = OUParams(lo=-np.inf, hi=np.inf, mean=0.0, sd=2.0, rho0=0.9, ell=5.0)
    mean, sd, since = gp_posterior(vals, hour, np.zeros(10), p)
    assert sd[0] == 0.0 and abs(mean[0] - 5.0) < 1e-6  # measured now: exact
    assert np.all(np.diff(sd[1:]) > 0)  # uncertainty grows with time since the last test
    assert np.all(np.diff(np.abs(mean[1:])) < 0)  # mean decays towards the population mean
    assert sd[-1] < 2.0 + 1e-6


def test_gp_never_measured_is_the_prior():
    hour = np.arange(4, dtype=float)
    p = OUParams(lo=-np.inf, hi=np.inf, mean=3.0, sd=1.5, rho0=0.8, ell=10.0)
    mean, sd, _ = gp_posterior(np.full(4, np.nan), hour, np.zeros(4), p)
    assert np.allclose(mean, 3.0) and np.allclose(sd, 1.5)


def test_last_obs_does_not_cross_patients():
    vals = np.array([1.0, np.nan, np.nan, 7.0])
    lv, since = last_obs(vals, np.array([0.0, 1.0, 0.0, 1.0]), np.array([0, 0, 1, 1]))
    assert lv[1] == 1.0 and np.isnan(lv[2]) and lv[3] == 7.0 and since[1] == 1.0


def test_lab_count_uses_only_the_past():
    n = 40
    df = pd.DataFrame({"patient": ["p"] * n})
    rng = np.random.default_rng(0)
    for lab in LABS:
        df[lab] = np.where(rng.random(n) < 0.2, 1.0, np.nan)
    before = lab_count_past(df, window=24)
    df.loc[30:, LABS] = 1.0  # change only the future
    after = lab_count_past(df, window=24)
    assert np.array_equal(before[:30], after[:30])
    assert after[30] == len(LABS)


def test_split_is_patient_level():
    df = pd.DataFrame({"hospital": ["A"] * 60 + ["B"] * 60, "patient": [f"p{i // 3}" for i in range(120)]})
    s = split_patients(df, seed=1)
    assert (df.assign(s=s).groupby("patient").s.nunique() == 1).all()
    assert set(s[df.hospital == "A"]) == {"A_train", "A_cal"} and set(s[df.hospital == "B"]) == {"B_cal", "B_test"}


def test_crc_threshold_bound_on_calibration():
    rng = np.random.default_rng(0)
    pats = np.repeat(np.arange(300), 5)
    scores = rng.random(len(pats))
    for alpha in (0.05, 0.1, 0.3):
        t = C.crc_threshold(scores, pats, alpha)
        per = pd.Series(scores < t).groupby(pats).mean()
        n = len(per)
        assert (n * per.mean() + 1) / (n + 1) <= alpha + 1e-12


def test_crc_guarantee_holds_on_average():
    """Average missed-sepsis risk on fresh exchangeable patients stays at or below alpha."""
    rng = np.random.default_rng(1)
    alpha, risks = 0.1, []
    for _ in range(300):
        def draw(n):
            k = rng.integers(1, 12, n)  # sepsis hours per patient
            pat = np.repeat(np.arange(n), k)
            frailty = np.repeat(rng.normal(0, 1, n), k)  # hours of one patient are correlated
            return 1 / (1 + np.exp(-(frailty + rng.normal(0.5, 1, len(pat))))), pat
        s_cal, p_cal = draw(200)
        t = C.crc_threshold(s_cal, p_cal, alpha)
        s_te, p_te = draw(200)
        risks.append(pd.Series(s_te < t).groupby(p_te).mean().mean())
    assert np.mean(risks) <= alpha + 0.005


def test_mondrian_thresholds_differ_by_group():
    df = pd.DataFrame({
        "patient": np.repeat(np.arange(400), 3),
        "group": np.repeat(np.arange(400) % 2, 3),
        "SepsisLabel": 1,
    })
    rng = np.random.default_rng(2)
    df["s"] = np.where(df.group == 0, rng.random(len(df)) * 0.5, 0.5 + rng.random(len(df)) * 0.5)
    thr = C.fit_thresholds(df, "s", 0.1)
    assert thr[0] < thr[1]  # the low-scoring group needs a lower threshold to keep its own misses at 10%
    clear = C.apply(df, "s", thr, "mondrian")
    for g in (0, 1):
        assert df.assign(c=clear)[df.group == g].groupby("patient").c.mean().mean() <= 0.12


def test_ece_is_zero_for_perfect_calibration():
    p = np.repeat(np.linspace(0.05, 0.95, 10), 1000)
    rng = np.random.default_rng(3)
    y = (rng.random(len(p)) < p).astype(int)
    assert ece(y, p) < 0.02
    assert ece(y, np.clip(p + 0.3, 0, 1)) > 0.2
