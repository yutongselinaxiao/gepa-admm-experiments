"""Figures for the projection diagnostics and the fluency (perplexity) analysis.

Inputs: artifacts/projection_diagnostics/report.json, artifacts/fluency_analysis/{agnews,emotion}.json,
artifacts/classification_results/summary.json. Outputs: artifacts/figures/*.png.

Colors follow the method in every figure (validated light-mode categorical slots, at most three
per chart, all-pairs): reflective GEPA = blue, Euclidean ADMM = orange, top-p ADMM = aqua,
beam ADMM = violet. Gray is reserved for references. Series also differ by marker or line style.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
REFERENCE = "#8f8e8a"
COLOR = {"reflective": "#2a78d6", "euclidean": "#eb6834", "topp": "#1baf7a", "beam": "#4a3aa7"}
MARKER = {"reflective": "o", "euclidean": "s", "topp": "^", "beam": "D"}
LINESTYLE = {"euclidean": "-", "topp": "--", "beam": ":"}
OUT = Path("artifacts/figures")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
    "lines.linewidth": 2.0,
})


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(fig, name: str, caption: str) -> None:
    """Captions are written in the documents that embed the figure, not drawn in the image."""
    del caption
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {OUT / name}")


def fig_gap_vs_bound(report: dict) -> None:
    rows = report["iterates"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    r = [row["primal_residual"] for row in rows]
    ax.scatter(r, [row["lipschitz_bound"] for row in rows], s=42, marker="s", color=COLOR["euclidean"],
               edgecolor=SURFACE, linewidth=1.5, label="Lipschitz bound  L·‖e − E(z)‖", zorder=3)
    ax.scatter(r, [max(abs(row["projection_gap"]), 1e-1) for row in rows], s=42, marker="o",
               color=COLOR["reflective"], edgecolor=SURFACE, linewidth=1.5,
               label="actual gap  |F(E(z)) − F(e)|", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("primal residual ‖e − E(z)‖ (18 saved standalone-ADMM iterates, ρ = 1)")
    ax.set_ylabel("loss change from projection (nats)")
    ax.set_title("The Lipschitz bound is 100–1000× the actual projection gap")
    ax.legend(loc="center right")
    _save(fig, "fig1_gap_vs_bound.png",
          "Each pair of points is one iterate. Gap = discrete minus continuous label cross-entropy on a "
          "32-example training batch; one iterate with gap −0.24 is shown at |gap|.")


def fig_random_displacement(report: dict) -> None:
    rows = report["random_displacement"]
    radius = [row["radius"] for row in rows]
    kept = [100 * row["fraction_kept_seed_token"] for row in rows]
    medians = [row["target_to_token_distance_median"] for row in report["iterates"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvspan(min(medians), max(medians), color=GRID, alpha=0.9, zorder=0, lw=0)
    ax.text((min(medians) + max(medians)) / 2, 6, "where ADMM's projection\ntargets sit (median per iterate)",
            ha="center", va="bottom", fontsize=8, color=TEXT_2)
    ax.axvline(report["token_norm"]["median"], color=REFERENCE, lw=1, ls="--")
    ax.text(report["token_norm"]["median"] + 0.3, 40, "median token\nnorm (1.58)", fontsize=8, color=TEXT_2)
    ax.plot(radius, kept, color=COLOR["reflective"], marker="o", markersize=6, markeredgecolor=SURFACE)
    ax.set_xlabel("random displacement radius per position (embedding units)")
    ax.set_ylabel("positions keeping the seed token (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Nearest-token projection is very sticky")
    _save(fig, "fig2_random_displacement.png",
          "Seed-instruction embeddings displaced in random directions (200 samples per radius), then projected "
          "to the nearest allowed token. Tokens are ~1.4 apart, yet a radius of ~10 is needed to change any.")


def _ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    xs = sorted(values)
    return xs, [100 * (i + 1) / len(xs) for i in range(len(xs))]


def fig_filter_keep_rate(fluency_by_task: dict) -> None:
    panels = {
        "agnews": [("euclidean", "euclidean (no constraint)", "Euclidean (no constraint)"),
                   ("topp", "top-p (hard constraint)", "top-p (hard constraint)"),
                   ("beam", "beam λ=0.3 (soft penalty)", "beam λ = 0.3 (soft penalty)")],
        "emotion": [("topp", "top-p (hard constraint)", "top-p (hard constraint)"),
                    ("beam", "beam λ=0.3 (soft penalty)", "beam λ = 0.3 (soft penalty)")],
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, (task, series) in zip(axes, panels.items()):
        for key, arm, label in series:
            values = [e["edit_nll"] for e in fluency_by_task[task]["proposals"][arm]]
            xs, ys = _ecdf(values)
            median = sorted(values)[len(values) // 2]
            ax.step(xs, ys, where="post", color=COLOR[key], ls=LINESTYLE[key],
                    label=f"{label}: median {median:.1f} nats, n={len(values)}")
        ax.axvline(12, color=REFERENCE, lw=1, ls="--")
        ax.text(12.4, 4, "filter at\n12 nats", fontsize=8, color=TEXT_2)
        ax.set_xlim(0, 35)
        ax.set_ylim(0, 102)
        ax.set_xlabel("filter threshold: max NLL of an inserted token (nats)")
        ax.set_title("AG News" if task == "agnews" else "Emotion")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8)
    axes[0].set_ylabel("ADMM edits the filter would keep (%)")
    fig.suptitle("A perplexity filter keeps most constrained edits and almost no Euclidean ones",
                 fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.subplots_adjust(top=0.84, wspace=0.08)
    _save(fig, "fig3_filter_keep_rate.png", "")


def fig_alternative_projections(report: dict) -> None:
    names = ["euclidean", "euclidean_ascii", "cosine", "whitened", "all_but_top20"]
    labels = ["Euclidean (current)", "Euclidean, ASCII only", "cosine", "whitened", "all-but-the-top"]
    changed, ascii_share = [], []
    for name in names:
        rows = [it["projections"][name] for it in report["iterates"]]
        changed.append(sum(r["changed_from_seed"] for r in rows) / len(rows))
        shares = [r["ascii_fraction_of_changes"] for r in rows if r["ascii_fraction_of_changes"] is not None]
        ascii_share.append(100 * sum(shares) / len(shares) if shares else 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    for ax, values, title, fmt in [(axes[0], changed, "tokens changed from seed (mean)", "{:.1f}"),
                                   (axes[1], ascii_share, "changed tokens that are ASCII (%)", "{:.0f}%")]:
        y = list(range(len(names)))[::-1]
        ax.barh(y, values, color=COLOR["reflective"], height=0.6)
        for yi, v in zip(y, values):
            ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8, color=TEXT)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", visible=False)
        if fmt.endswith("%"):
            ax.set_xlim(0, 115)
            ax.set_xticks([0, 25, 50, 75, 100])
        else:
            ax.set_xlim(0, max(values) * 1.25)
    axes[0].set_yticks(list(range(len(names)))[::-1], labels)
    fig.suptitle("Changing the distance doesn't help: more changes means more garbage", fontsize=11,
                 fontweight="bold", color=TEXT, x=0.02, ha="left", y=1.02)
    fig.subplots_adjust(wspace=0.35)
    _save(fig, "fig4_alternative_projections.png",
          "The same 18 projection targets e + u projected under each distance. Whitening and all-but-the-top "
          "change more tokens, mostly into non-ASCII tokens, and score worse.")


def fig_fluency_vs_gain(fluency: dict) -> None:
    series = [("euclidean", "euclidean (no constraint)", "Euclidean (no constraint)"),
              ("topp", "top-p (hard constraint)", "top-p (hard constraint)"),
              ("beam", "beam λ=0.3 (soft penalty)", "beam λ = 0.3 (soft penalty)")]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axhline(0, color=REFERENCE, lw=1)
    ax.axvline(12, color=REFERENCE, lw=1, ls="--")
    ax.text(12.4, 9.2, "a post-hoc perplexity filter\nat 12 nats keeps edits left\nof this line",
            fontsize=8, color=TEXT_2, va="top")
    for key, arm, label in series:
        edits = [e for e in fluency["proposals"][arm] if e["validation_gain_pp"] is not None]
        ax.scatter([e["edit_nll"] for e in edits], [e["validation_gain_pp"] for e in edits], s=26,
                   marker=MARKER[key], color=COLOR[key], alpha=0.75, edgecolor=SURFACE, linewidth=0.8,
                   label=f"{label} (n={len(edits)})", zorder=3)
    ax.set_xlabel("NLL of the most surprising inserted token (nats; higher = less fluent)")
    ax.set_ylabel("validation accuracy change vs. parent (pp)")
    ax.set_title("Euclidean gains need disfluent tokens; constrained edits stay fluent")
    ax.legend(loc="lower right")
    _save(fig, "fig5_fluency_vs_gain.png",
          "AG News ADMM proposals with both texts audited on full validation. Euclidean edits that help sit "
          "almost entirely right of the filter line, so filtering afterwards discards them; top-p and beam "
          "search find edits within the fluent region.")


def fig_accuracy_vs_perplexity(fluency_by_task: dict) -> None:
    panels = {
        "agnews": [("reflective", "agnews/reflective_p12", "reflective GEPA"),
                   ("euclidean", "agnews/p1_admm_euclid_rho0.3_heldout", "GEPA + Euclidean ADMM"),
                   ("beam", "agnews/admm_beam_lam0.3_indep_accept", "GEPA + beam ADMM (fair)")],
        "emotion": [("reflective", "emotion/reflective_p12", "reflective GEPA"),
                    ("topp", "emotion/admm_topp", "GEPA + top-p ADMM"),
                    ("beam", "emotion/admm_beam_lam0.3_indep_accept", "GEPA + beam ADMM (fair)")],
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for ax, (task, series) in zip(axes, panels.items()):
        data = fluency_by_task[task]
        extra = []
        if task == "agnews":  # both Euclidean held-out rhos, re-scored on the report half
            extra = data["arms"]["agnews/p1_admm_euclid_rho1_heldout"]
        at_seed = []
        for key, arm, label in series:
            rows = data["arms"][arm] + (extra if key == "euclidean" else [])
            on_seed = sum(r["text"] == data["seed"]["text"] for r in rows)
            if on_seed:
                at_seed.append(f"{on_seed} {key if key != 'topp' else 'top-p'}")
            ax.scatter([r["per_token_perplexity"] for r in rows], [r["report_accuracy"] for r in rows], s=40,
                       marker=MARKER[key], color=COLOR[key], edgecolor=SURFACE, linewidth=1.2, alpha=0.9,
                       label=f"{label} (n={len(rows)})", zorder=3)
        seed = data["seed"]
        seed_acc = data["arms"][series[0][1]][0]
        seed_report = next(iter(_load("artifacts/classification_results/summary.json")["arms"][series[0][1]]["runs"]))
        ax.scatter([seed["per_token_perplexity"]], [seed_report["seed_report_accuracy"]], s=110, marker="*",
                   color=TEXT, zorder=4)
        note = "seed instruction"
        if at_seed:
            note += "\nruns ending on it:\n" + ", ".join(at_seed)
        ax.annotate(note, (seed["per_token_perplexity"], seed_report["seed_report_accuracy"]),
                    xytext=(-8, -6), textcoords="offset points", fontsize=7.5, color=TEXT,
                    va="top", ha="right")
        ax.set_xscale("log")
        ticks = [10, 20, 50, 100, 200, 500, 1000]
        lo = min(min(c.get_offsets()[:, 0]) for c in ax.collections) * 0.8
        hi = max(max(c.get_offsets()[:, 0]) for c in ax.collections) * 1.25
        ax.set_xlim(lo, hi)
        ax.set_xticks([t for t in ticks if lo <= t <= hi], [str(t) for t in ticks if lo <= t <= hi])
        ax.minorticks_off()
        ax.set_xlabel("prompt perplexity per token (log scale)")
        ax.set_title("AG News" if task == "agnews" else "Emotion")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.8)
        del seed_acc
    axes[0].set_ylabel("report-half accuracy (%)")
    fig.suptitle("Accuracy vs. readability of the prompt each run selected", fontsize=11, fontweight="bold",
                 x=0.02, ha="left")
    fig.subplots_adjust(top=0.86, wspace=0.18)
    _save(fig, "fig6_accuracy_vs_perplexity.png",
          "Each point is one run's final prompt. Perplexity is measured on the instruction slot given the fixed "
          "prompt prefix (the label mapping comes after the slot, so task words are surprising; compare "
          "against the seed). Euclidean ADMM points are the six Phase 1 held-out runs, re-scored on the report half.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = _load("artifacts/projection_diagnostics/report.json")
    fluency = {task: _load(f"artifacts/fluency_analysis/{task}.json") for task in ("agnews", "emotion")}
    fig_gap_vs_bound(report)
    fig_random_displacement(report)
    fig_filter_keep_rate(fluency)
    fig_alternative_projections(report)
    fig_fluency_vs_gain(fluency["agnews"])
    fig_accuracy_vs_perplexity(fluency)


if __name__ == "__main__":
    main()
