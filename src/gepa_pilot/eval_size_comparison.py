"""Side by side: the same runs evaluated on the 256-example report half vs. the ~2,000-example
report sets. Selection is identical (each run chose its prompt on the 256-example selection
half); only the evaluation of that prompt differs.

Writes artifacts/eval_size/comparison.json, artifacts/eval_size/tables.md and
artifacts/figures/fig8_eval_size.png.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from .make_figures import COLOR, GRID, REFERENCE, SURFACE, TEXT, TEXT_2  # noqa: F401  (shared style)

ARMS = {
    "agnews": ["reflective_p12", "reflective_heldout_p30", "reflective_p40", "admm_topp",
               "admm_beam_lam0.3", "admm_beam_lam0.3_indep_accept"],
    "emotion": ["reflective_p12", "reflective_heldout_p30", "admm_topp",
                "admm_beam_lam0.3", "admm_beam_lam0.3_indep_accept"],
}
LABELS = {
    "reflective_p12": "Reflective GEPA (12 proposals)",
    "reflective_heldout_p30": "Matched reflective (30, held-out)",
    "reflective_p40": "Reflective GEPA (40 proposals)",
    "admm_topp": "GEPA + top-p ADMM proposer",
    "admm_beam_lam0.3": "GEPA + beam ADMM (S used twice)",
    "admm_beam_lam0.3_indep_accept": "GEPA + beam ADMM (fair S')",
}
METHOD = {"reflective_p12": "reflective", "reflective_heldout_p30": "reflective", "reflective_p40": "reflective",
          "admm_topp": "topp", "admm_beam_lam0.3": "beam", "admm_beam_lam0.3_indep_accept": "beam"}
COMPARISONS = [
    ("admm_beam_lam0.3_indep_accept", "reflective_heldout_p30"),
    ("admm_beam_lam0.3_indep_accept", "reflective_p12"),
    ("admm_topp", "reflective_p12"),
]


def ci(values: list[float]) -> tuple[float, float, float]:
    mean, sd = statistics.mean(values), statistics.stdev(values)
    return mean, sd, stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values))


def welch(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Difference of means, Welch 95% CI half-width, and p-value."""
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = statistics.mean(a) - statistics.mean(b)
    return diff, stats.t.ppf(0.975, df) * math.sqrt(va + vb), float(stats.ttest_ind(a, b, equal_var=False).pvalue)


SHORT = {"admm_beam_lam0.3_indep_accept": "fair beam ADMM", "reflective_heldout_p30": "matched reflective",
         "reflective_p12": "reflective GEPA", "admm_topp": "top-p ADMM"}


def main() -> None:
    summary = json.loads(Path("artifacts/classification_results/summary.json").read_text(encoding="utf-8"))
    big = {t: json.loads(Path(f"artifacts/report2k/{t}.json").read_text(encoding="utf-8")) for t in ARMS}
    data: dict = {}
    for task, arms in ARMS.items():
        for arm in arms:
            runs = summary["arms"][f"{task}/{arm}"]["runs"]
            data[(task, arm)] = ([r["report_accuracy"] for r in runs],
                                 [100 * big[task]["by_run"][r["run"]] for r in runs])
    lines: list[str] = []
    out: dict = {}
    for task, arms in ARMS.items():
        name = "AG News" if task == "agnews" else "Emotion"
        size = big[task]["report_size"]
        seed_small = summary["arms"][f"{task}/reflective_p12"]["seed_instruction_report_accuracy"]
        seed_big = 100 * big[task]["seed_accuracy"]
        lines += [f"#### {name}: per-arm results (10 seeds)", "",
                  f"| Arm | 256 examples: mean ± sd | 95% CI | {size:,} examples: mean ± sd | 95% CI | Spread (sd) change |",
                  "|---|---:|---:|---:|---:|---:|",
                  f"| *Seed instruction* | *{seed_small:.2f}* | | *{seed_big:.2f}* | | |"]
        for arm in arms:
            small, large = data[(task, arm)]
            ms, ss, hs = ci(small)
            ml, sl, hl = ci(large)
            out[f"{task}/{arm}"] = {"256": [ms, ss, hs], "2k": [ml, sl, hl]}
            lines.append(f"| {LABELS[arm]} | {ms:.2f} ± {ss:.2f} | ±{hs:.2f} | {ml:.2f} ± {sl:.2f} | ±{hl:.2f} | "
                         f"{100 * (sl - ss) / ss:+.0f}% |")
        lines += ["", f"#### {name}: comparisons (10 seeds each, Welch's t-test)", "",
                  f"| Comparison | 256 examples: difference | p | {size:,} examples: difference | p |",
                  "|---|---:|---:|---:|---:|"]
        for a, b in COMPARISONS:
            (sa, la), (sb, lb) = data[(task, a)], data[(task, b)]
            ds, ps = statistics.mean(sa) - statistics.mean(sb), float(stats.ttest_ind(sa, sb, equal_var=False).pvalue)
            dl, pl = statistics.mean(la) - statistics.mean(lb), float(stats.ttest_ind(la, lb, equal_var=False).pvalue)
            out[f"{task}/{a} - {b}"] = {"256": [ds, ps], "2k": [dl, pl]}
            lines.append(f"| {LABELS[a]} − {LABELS[b]} | {ds:+.2f} | {ps:.3f} | {dl:+.2f} | {pl:.3f} |")
        lines.append("")
    Path("artifacts/eval_size").mkdir(parents=True, exist_ok=True)
    Path("artifacts/eval_size/comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    Path("artifacts/eval_size/tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # Figure: per arm, each run's score on 256 (hollow) and ~2,000 (filled) examples, with mean ± 95% CI.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=False)
    for ax, (task, arms) in zip(axes, ARMS.items()):
        for row, arm in enumerate(reversed(arms)):
            color = COLOR[METHOD[arm]]
            for values, dy, filled in ((data[(task, arm)][0], 0.17, False), (data[(task, arm)][1], -0.17, True)):
                ax.scatter(values, [row + dy] * len(values), s=26, marker="o",
                           facecolors=color if filled else SURFACE, edgecolors=color, linewidths=1.2,
                           alpha=0.85, zorder=3)
                mean, _, half = ci(values)
                ax.errorbar([mean], [row + dy], xerr=[[half], [half]], fmt="|", color=TEXT, markersize=11,
                            capsize=3, lw=1.4, zorder=4)
        seed_small = summary["arms"][f"{task}/reflective_p12"]["seed_instruction_report_accuracy"]
        seed_big = 100 * big[task]["seed_accuracy"]
        ax.axvline(seed_small, color=REFERENCE, lw=1, ls=":")
        ax.axvline(seed_big, color=REFERENCE, lw=1, ls="--")
        ax.set_yticks(range(len(arms)), [LABELS[a] for a in reversed(arms)], fontsize=8)
        ax.set_xlabel("report accuracy (%)")
        ax.set_title("AG News" if task == "agnews" else "Emotion")
        ax.grid(axis="y", visible=False)
    axes[1].tick_params(axis="y", labelsize=8)
    handles = [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=TEXT_2, label="run scored on 256 examples"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor=TEXT_2, markeredgecolor=TEXT_2, label="same run scored on ~2,000 examples"),
        plt.Line2D([], [], marker="|", ls="", color=TEXT, markersize=11, label="mean ± 95% CI (10 seeds)"),
        plt.Line2D([], [], ls=":", color=REFERENCE, label="seed instruction (256)"),
        plt.Line2D([], [], ls="--", color=REFERENCE, label="seed instruction (~2,000)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Same runs, two evaluation-set sizes", fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.subplots_adjust(wspace=0.55, bottom=0.2, top=0.88)
    fig.savefig("artifacts/figures/fig8_eval_size.png", dpi=200, bbox_inches="tight")
    print("WROTE artifacts/figures/fig8_eval_size.png")

    # Figure 8b: each comparison's difference with its 95% CI, on 256 vs. ~2,000 examples.
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
    for ax, task in zip(axes, ARMS):
        for row, (a, b) in enumerate(reversed(COMPARISONS)):
            for measure, dy, filled in ((0, 0.15, False), (1, -0.15, True)):
                diff, half, p = welch(data[(task, a)][measure], data[(task, b)][measure])
                color = COLOR[METHOD[a]]
                ax.errorbar([diff], [row + dy], xerr=[[half], [half]], fmt="o", color=color, capsize=3, lw=1.3,
                            markerfacecolor=color if filled else SURFACE, markeredgecolor=color, markersize=6)
                ax.text(diff + half + 0.15, row + dy, f"p = {p:.3f}", va="center", fontsize=7.5,
                        color=TEXT if p < 0.05 else TEXT_2, fontweight="bold" if p < 0.05 else "normal")
        ax.axvline(0, color=REFERENCE, lw=1)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.3 * (hi - lo))
        ax.set_yticks(range(len(COMPARISONS)), [f"{SHORT[a]} − {SHORT[b]}" for a, b in reversed(COMPARISONS)],
                      fontsize=8.5)
        ax.set_xlabel("difference in report accuracy (pp), mean ± 95% CI")
        ax.set_title("AG News" if task == "agnews" else "Emotion")
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=TEXT_2,
                          label="scored on 256 examples"),
               plt.Line2D([], [], marker="o", ls="", markerfacecolor=TEXT_2, markeredgecolor=TEXT_2,
                          label="scored on ~2,000 examples")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Key comparisons on both evaluation sets (10 seeds each, Welch's t-test; bold = p < 0.05)",
                 fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.subplots_adjust(wspace=0.05, bottom=0.25, top=0.84)
    fig.savefig("artifacts/figures/fig8b_eval_size_differences.png", dpi=200, bbox_inches="tight")
    print("WROTE artifacts/figures/fig8b_eval_size_differences.png")


if __name__ == "__main__":
    main()
