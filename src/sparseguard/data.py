"""Load the PhysioNet/CinC 2019 sepsis data into one hourly table, split patients and define chart sparsity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CACHE, LABEL, LABS, RAW, SEED, SPARSITY_WINDOW_H, SPLIT_A_TRAIN, SPLIT_B_CAL


def load_raw(raw_dir=RAW, cache_dir=CACHE) -> pd.DataFrame:
    """All patients of both hospitals as one hourly table (cached as a pickle after the first read)."""
    cache = cache_dir / "hourly.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    frames = []
    for hosp in ("A", "B"):
        files = sorted((raw_dir / f"training_set{hosp}").glob("*.psv"))
        if not files:
            raise FileNotFoundError(f"no .psv files in {raw_dir / f'training_set{hosp}'}; see README for download")
        for f in files:
            d = pd.read_csv(f, sep="|")
            d.insert(0, "hour", np.arange(len(d), dtype=np.int16))
            d.insert(0, "patient", f.stem)
            d.insert(0, "hospital", hosp)
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    num = df.columns.difference(["hospital", "patient"])
    df[num] = df[num].astype(np.float32)
    df[LABEL] = df[LABEL].astype(np.int8)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def split_patients(df: pd.DataFrame, seed: int = SEED, b_cal: float = SPLIT_B_CAL) -> pd.Series:
    """Patient-level split: A_train, A_cal, B_cal, B_test. Hours of one patient never cross splits."""
    rng = np.random.default_rng(seed)
    out = {}
    for hosp, frac, names in (("A", SPLIT_A_TRAIN, ("A_train", "A_cal")), ("B", b_cal, ("B_cal", "B_test"))):
        pats = np.array(sorted(df.loc[df.hospital == hosp, "patient"].unique()))
        rng.shuffle(pats)
        k = int(round(frac * len(pats)))
        out.update({p: names[0] for p in pats[:k]})
        out.update({p: names[1] for p in pats[k:]})
    return df.patient.map(out)


def lab_count_past(df: pd.DataFrame, window: int = SPARSITY_WINDOW_H) -> np.ndarray:
    """Distinct lab variables measured in the last `window` hours, including the current hour.

    Uses only the current and earlier rows of the same patient, so it never looks ahead.
    """
    measured = df[LABS].notna().to_numpy(np.int8)
    out = np.zeros(len(df), dtype=np.int16)
    starts = np.flatnonzero(np.r_[True, df.patient.to_numpy()[1:] != df.patient.to_numpy()[:-1]])
    ends = np.r_[starts[1:], len(df)]
    for s, e in zip(starts, ends, strict=True):
        m = measured[s:e]
        c = np.cumsum(m, axis=0)
        lagged = np.vstack([np.zeros((window, m.shape[1]), dtype=c.dtype), c])[: e - s]
        out[s:e] = ((c - lagged) > 0).sum(axis=1)
    return out


SPARSITY_LABELS = ("sparse", "medium", "dense")


def sparsity_groups(lab_count: np.ndarray, cuts: tuple[int, int]) -> np.ndarray:
    """0 = sparse (<= cuts[0] labs), 1 = medium, 2 = dense (> cuts[1] labs)."""
    return np.where(lab_count <= cuts[0], 0, np.where(lab_count <= cuts[1], 1, 2)).astype(np.int8)


def sparsity_cuts(lab_count_train: np.ndarray) -> tuple[int, int]:
    """Tertile cut points of the past-24h lab count, from training hours only (ties go to the lower group)."""
    q1, q2 = np.quantile(lab_count_train, [1 / 3, 2 / 3])
    return int(q1), int(q2)
