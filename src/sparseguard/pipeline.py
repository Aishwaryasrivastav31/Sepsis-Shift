"""Stage 1: load data, split patients, impute, train five models on hospital A, score A_cal and hospital B.

Writes data/cache/scores.pkl (one row per scored hour) and reports/tables/00_run_meta.json.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import features as F
from . import model as M
from .config import CACHE, CLINICAL, LABEL, N_IMPUTATIONS, SEED, TAB
from .data import SPARSITY_LABELS, lab_count_past, load_raw, sparsity_cuts, sparsity_groups, split_patients


def prepare():
    """Hourly table with splits, past-24h lab counts, sparsity groups, imputations and GP hyperparameters."""
    df = load_raw()
    df["split"] = split_patients(df).to_numpy()
    df["lab_count_24h"] = lab_count_past(df)
    train = (df.split == "A_train").to_numpy()
    cuts = sparsity_cuts(df.lab_count_24h.to_numpy()[train])
    df["group"] = sparsity_groups(df.lab_count_24h.to_numpy(), cuts)
    params = F.fit_imputer(df, train)
    imp = F.impute_all(df, params)
    use = [v for v in CLINICAL if np.isfinite(params[v].lo)]  # variables never measured in hospital A are dropped
    return df, params, imp, use, cuts


def run(log=print) -> pd.DataFrame:
    t0 = time.time()
    df, params, imp, use, cuts = prepare()
    log(f"{len(df):,} hours; sparsity cuts (labs in past 24 h): sparse <= {cuts[0]}, dense > {cuts[1]}; "
        f"{len(use)} of {len(CLINICAL)} variables measured in hospital A ({time.time() - t0:.0f}s)")
    pd.DataFrame([{"variable": v, **params[v].__dict__} for v in CLINICAL]).to_csv(
        TAB / "01_gp_hyperparameters.csv", index=False
    )
    means = {v: params[v].mean for v in use}

    tr = np.flatnonzero((df.split == "A_train").to_numpy())
    sc = np.flatnonzero((df.split != "A_train").to_numpy())
    y = df[LABEL].to_numpy()[tr]
    rng = np.random.default_rng(SEED)
    scores = df.loc[sc, ["hospital", "patient", "hour", "split", "group", "lab_count_24h", "ICULOS", LABEL]]
    scores = scores.reset_index(drop=True)
    trees = {}

    for kind in ("mean", "locf", "locf_ts", "gp"):
        X, _ = F.matrix(df, imp, kind, tr, variables=use, means=means)
        m = M.fit(X, y)
        del X
        scores[f"p_{kind}"] = M.predict(m, F.matrix(df, imp, kind, sc, variables=use, means=means)[0])
        trees[kind] = int(m.n_iter_)
        log(f"{F.KINDS[kind]}: {m.n_iter_} trees ({time.time() - t0:.0f}s)")

    # multiple imputation: train on one posterior draw, score every hour under N_IMPUTATIONS draws
    X, names = F.matrix(df, imp, "gp", tr, gp_values=F.gp_draws(imp, tr, 1, rng, use)[0], variables=use)
    m = M.fit(X, y)
    del X
    P = np.empty((N_IMPUTATIONS, len(sc)), dtype=np.float32)
    for k in range(N_IMPUTATIONS):
        draw = F.gp_draws(imp, sc, 1, rng, use)[0]
        P[k] = M.predict(m, F.matrix(df, imp, "gp", sc, gp_values=draw, variables=use)[0])
    scores["p_gp_mi"] = P.mean(0)
    scores["p_gp_mi_upper"] = np.quantile(P, 0.9, axis=0).astype(np.float32)  # pessimistic draw
    scores["p_gp_mi_sd"] = P.std(0)
    trees["gp_mi"] = int(m.n_iter_)
    log(f"{F.KINDS['gp_mi']}: {m.n_iter_} trees, {N_IMPUTATIONS} draws ({time.time() - t0:.0f}s)")

    meta = {
        "hours": int(len(df)),
        "patients": {
            s: int(df.loc[df.split == s, "patient"].nunique()) for s in ("A_train", "A_cal", "B_cal", "B_test")
        },
        "sparsity_cuts": list(cuts),
        "group_labels": list(SPARSITY_LABELS),
        "trees": trees,
        "gp_mi_features": len(names),
        "variables_used": use,
        "runtime_s": round(time.time() - t0),
    }
    (TAB / "00_run_meta.json").write_text(json.dumps(meta, indent=2))
    scores.to_pickle(CACHE / "scores.pkl")
    return scores


if __name__ == "__main__":
    run()
