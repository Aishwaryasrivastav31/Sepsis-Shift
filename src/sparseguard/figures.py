"""Figures for the SparseGuard report."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import SPARSITY_LABELS  # noqa: E402

INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
GROUP_COL = {"sparse": "#c2410c", "medium": "#a3a3a3", "dense": "#1d4ed8"}
MODE_COL = {"marginal": "#c2410c", "mondrian": "#1d4ed8"}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left",
})


def fig_shift(scores: pd.DataFrame, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bins = np.arange(0, 27) - 0.5
    for h, col, lab in (("A", "#6b7280", "Hospital A (held-out patients)"), ("B", "#c2410c", "Hospital B")):
        d = scores[scores.hospital == "B"] if h == "B" else scores[scores.split == "A_cal"]
        ax.hist(d.lab_count_24h, bins=bins, density=True, histtype="step", lw=2, color=col, label=lab)
    ax.set_xlabel("Distinct lab tests measured in the previous 24 hours")
    ax.set_ylabel("Share of patient-hours")
    ax.set_title("Hospital B's charts are much sparser")
    ax.legend(frameon=False)
    fig.savefig(out / "fig1_sparsity_shift.png")
    plt.close(fig)


def fig_discrimination(disc: pd.DataFrame, out):
    d = disc[(disc.cohort.str.startswith("Hospital B")) & (disc.group != "all")]
    models = list(dict.fromkeys(d.model))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    w = 0.8 / len(models)
    cols = ["#d4d4d8", "#a1a1aa", "#52525b", "#60a5fa", "#1d4ed8"]
    for ax, metric, title in ((axes[0], "auroc", "AUROC by chart sparsity (hospital B)"),
                              (axes[1], "ece", "Calibration error by chart sparsity (hospital B)")):
        for i, m in enumerate(models):
            v = [d[(d.model == m) & (d.group == g)][metric].iloc[0] for g in SPARSITY_LABELS]
            ax.bar(np.arange(3) + (i - (len(models) - 1) / 2) * w, v, w, color=cols[i % len(cols)], label=m)
        ax.set_xticks(range(3), SPARSITY_LABELS)
        ax.set_title(title)
    axes[0].set_ylim(0.5, 1.0)
    axes[1].set_ylabel("ECE (equal-mass bins)")
    axes[0].set_ylabel("AUROC")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(out / "fig2_discrimination_by_group.png")
    plt.close(fig)


def fig_reliability(rel: pd.DataFrame, out):
    models = list(dict.fromkeys(rel.model))
    fig, axes = plt.subplots(1, len(models), figsize=(9.5, 3.8), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models, strict=True):
        d = rel[(rel.model == m) & (rel.cohort == "B")]
        hi = max(d.predicted.max(), d.observed.max()) * 1.05
        ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--")
        for g in SPARSITY_LABELS:
            dg = d[d.group == g]
            ax.plot(dg.predicted, dg.observed, "o-", color=GROUP_COL[g], label=g, ms=4)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("Predicted probability (bin mean)")
    np.atleast_1d(axes)[0].set_ylabel("Observed sepsis-hour rate")
    np.atleast_1d(axes)[0].legend(frameon=False, title="Chart")
    fig.suptitle("Reliability on hospital B, sparse vs dense charts", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig3_reliability.png")
    plt.close(fig)


def fig_conformal(summ: pd.DataFrame, score: str, out, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, src in zip(axes, ("Hospital B", "Hospital A"), strict=True):
        d = summ[(summ.score == score) & (summ.alpha == 0.10) & (summ.calibrated_on == src) & (summ.group != "all")]
        for i, mode in enumerate(("marginal", "mondrian")):
            dm = d[d["mode"] == mode].set_index("group").reindex(SPARSITY_LABELS)
            x = np.arange(3) + (i - 0.5) * 0.36
            ax.bar(x, dm.missed_risk_patient, 0.36, color=MODE_COL[mode], alpha=0.9,
                   label="one threshold (marginal)" if mode == "marginal" else "per-group thresholds (Mondrian)")
            ax.errorbar(x, dm.missed_risk_patient, yerr=[dm.missed_risk_patient - dm.missed_p05,
                        dm.missed_p95 - dm.missed_risk_patient], fmt="none", ecolor=INK, lw=1, capsize=3)
        ax.axhline(0.10, color=INK, lw=1, ls="--")
        ax.text(2.45, 0.103, "target 10%", ha="right", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(3), SPARSITY_LABELS)
        ax.set_title(f"Calibrated on {src}, tested on hospital B")
        ax.set_ylabel("Missed sepsis hours (patient-averaged)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Share of sepsis hours given a confident all-clear ({score})", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / name)
    plt.close(fig)


def fig_cost(summ: pd.DataFrame, score: str, out):
    d = summ[(summ.score == score) & (summ.calibrated_on == "Hospital B") & (summ.group != "all")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (a, mode) in enumerate([(0.1, "marginal"), (0.1, "mondrian"), (0.2, "marginal"), (0.2, "mondrian")]):
        dm = d[(d.alpha == a) & (d["mode"] == mode)].set_index("group").reindex(SPARSITY_LABELS)
        x = np.arange(3) + (i - 1.5) * 0.2
        ax.bar(x, dm.check_rate, 0.2, color=MODE_COL[mode], alpha=1.0 if a == 0.1 else 0.45,
               label=f"{'marginal' if mode == 'marginal' else 'Mondrian'}, alpha = {a:.0%}")
    ax.set_xticks(range(3), SPARSITY_LABELS)
    ax.set_ylabel("Non-sepsis hours sent for a check")
    ax.set_title("The price of the guarantee: hours the model declines to clear")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.savefig(out / "fig5_check_rate.png")
    plt.close(fig)


def make_all(scores, disc, rel, summ, raw, out):
    out.mkdir(parents=True, exist_ok=True)
    fig_shift(scores, out)
    fig_discrimination(disc, out)
    fig_reliability(rel, out)
    fig_conformal(summ, "Causal GP, multiple imputation", out, "fig4_conformal_gp_mi.png")
    fig_conformal(summ, "Carry-forward + time since last", out, "fig4b_conformal_locf_ts.png")
    fig_cost(summ, "Causal GP, multiple imputation", out)
