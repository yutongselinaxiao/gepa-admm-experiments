"""Cross-task comparison of all GEPA arms (Phases 6-14) on five task settings, scored on each task's
report set: AG News (letter answers), emotion, Yahoo Answers topics, AG News answered with label names,
and GoEmotions (multi-label; example / micro / macro F1).

Writes artifacts/cross_task/{tables.md,summary.json} and figures fig18-fig20.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from .config import PilotConfig  # noqa: E402
from .make_figures import COLOR, REFERENCE, SURFACE, TEXT, TEXT_2  # noqa: E402  (shared style)

ARMS = ["A", "B", "C", "E", "D", "F", "G", "I", "H"]
NAMES = {
    "A": "A: reflection, minibatch rule", "B": "B: reflection, S′ judge (30)", "C": "C: fixed-length ADMM",
    "D": "D: multi-length, LM insertion", "E": "E: reflection + ADMM polish", "F": "F: multi-length, gradient insertion",
    "G": "G: gated variable length", "H": "H: fixed-length ADMM, 36 proposals", "I": "I: multi-start (3 runs)",
}
FAMILY = {"A": "reflective", "B": "reflective", "E": "euclidean", "C": "beam", "H": "beam", "I": "beam",
          "D": "topp", "F": "topp", "G": "topp"}
OLD = {  # AG News (letters) and emotion: run directories of Phases 2-10
    "A": {"agnews": r"phase2/reflective_s\d", "emotion": r"phase4_emotion/reflective_p12_s\d"},
    "B": {"agnews": r"phase3/reflective_heldout_p30_s\d", "emotion": r"phase4_emotion/reflective_heldout_p30_s\d"},
    "C": {t: rf"phase6_independent_acceptance/{t}_beam_indep_s\d" for t in ("agnews", "emotion")},
    "D": {t: rf"phase7/{t}_multilength_s\d" for t in ("agnews", "emotion")},
    "E": {t: rf"phase7/{t}_polish_fair_s\d" for t in ("agnews", "emotion")},
    "F": {t: rf"phase10/{t}_mlgrad_s\d" for t in ("agnews", "emotion")},
    "G": {t: rf"phase8/{t}_gated_s\d" for t in ("agnews", "emotion")},
    "H": {t: rf"phase9/{t}_beam_p36_s\d" for t in ("agnews", "emotion")},
    "I": {t: rf"phase9/{t}_multistart_s\d" for t in ("agnews", "emotion")},
}
NEW_ARM = {"A": "A_reflective_p12", "B": "B_reflective_heldout_p30", "C": "C_beam", "E": "E_polish_fair",
           "F": "F_mlgrad", "G": "G_gated", "H": "H_beam_p36", "I": "I_multistart"}
SETTINGS = [  # (key, title, report file, metric, run-directory pattern per arm)
    ("agnews", "AG News, letter answers", "agnews", "accuracy", lambda a: OLD[a]["agnews"]),
    ("emotion", "Emotion", "emotion", "accuracy", lambda a: OLD[a]["emotion"]),
    ("yahoo", "Yahoo Answers topics", "yahoo", "accuracy", lambda a: rf"phase11_yahoo/yahoo_{NEW_ARM.get(a, '-')}_s\d"),
    ("agnews_names", "AG News, label-name answers", "agnews_names", "accuracy",
     lambda a: rf"phase12_13/agnews_names_{NEW_ARM.get(a, '-')}_s\d"),
    ("goemotions_example", "GoEmotions, example F1", "goemotions_f1", "example_f1",
     lambda a: rf"phase14_goemotions_f1/goemotions_{NEW_ARM.get(a, '-')}_s\d"),
    ("goemotions_micro", "GoEmotions, micro F1", "goemotions_f1", "micro_f1",
     lambda a: rf"phase14_goemotions_f1/goemotions_{NEW_ARM.get(a, '-')}_s\d"),
]


def welch(x: list[float], y: list[float]) -> tuple[float, float, float]:
    vx, vy = statistics.variance(x) / len(x), statistics.variance(y) / len(y)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    return (statistics.mean(x) - statistics.mean(y), stats.t.ppf(0.975, df) * math.sqrt(vx + vy),
            float(stats.ttest_ind(x, y, equal_var=False).pvalue))


def load_setting(report: str, metric: str, pattern) -> tuple[float, dict[str, list[float]]]:
    big = json.loads(Path(f"artifacts/report2k/{report}.json").read_text(encoding="utf-8"))
    if metric == "accuracy":
        by_run, seed = big["by_run"], big["seed_accuracy"]
    else:
        by_run = {k: v[metric] for k, v in big["f1_by_run"].items()}
        seed = big["seed_f1"][metric]
    arms = {}
    for arm in ARMS:
        values = [100 * v for k, v in sorted(by_run.items()) if re.search(pattern(arm), k)]
        if len(values) >= 2:
            arms[arm] = values
    return 100 * seed, arms


def goemotions_hedging() -> dict[str, list[dict[str, float]]]:
    """Per run: emotions predicted per comment, precision and recall of the final prompt on the report set."""
    big = json.loads(Path("artifacts/report2k/goemotions_f1.json").read_text(encoding="utf-8"))
    gold = {r["example_id"]: r["gold_label"] for r in json.loads(Path("artifacts/data_goemotions/report.json").read_text())}

    def stats_of(prompt: str) -> dict[str, float]:
        predicted = big["by_prompt"][prompt]["predicted_by_id"]
        tp = fp = fn = count = 0
        for i, g in gold.items():
            p, gs = set(predicted[i]), set(g)
            tp, fp, fn, count = tp + len(p & gs), fp + len(p - gs), fn + len(gs - p), count + len(p)
        return {"per_comment": count / len(gold), "precision": 100 * tp / (tp + fp), "recall": 100 * tp / (tp + fn)}

    seed = PilotConfig.from_yaml("configs/goemotions.yaml").prompt.seed_instruction
    out = {"seed": [stats_of(seed) | {m: 100 * v for m, v in big["seed_f1"].items()}],
           "true_per_comment": sum(len(g) for g in gold.values()) / len(gold)}
    for arm, name in NEW_ARM.items():
        rows = []
        for run, f1 in sorted(big["f1_by_run"].items()):
            if re.search(rf"phase14_goemotions_f1/goemotions_{name}_s\d", run):
                prompt = json.loads(Path(run, "gepa_result.json").read_text(encoding="utf-8"))["best_instruction"]
                rows.append(stats_of(prompt) | {m: 100 * v for m, v in f1.items()})
        if rows:
            out[arm] = rows
    return out


def label_name_ceiling() -> list[str]:
    seed = PilotConfig.from_yaml("configs/agnews_names.yaml").prompt.seed_instruction.strip()
    lines = ["## AG News with label-name answers: how often runs leave the seed instruction", "",
             "| Arm | Runs whose final prompt is the seed | Accepted children per run |", "|---|---:|---:|"]
    for arm, name in NEW_ARM.items():
        results = [json.loads(p.read_text(encoding="utf-8")) for p in
                   sorted(Path("artifacts/phase12_13").glob(f"agnews_names_{name}_s*/gepa_result.json"))]
        if results:
            stay = sum(r["best_instruction"].strip() == seed for r in results)
            accepted = statistics.mean(sum(c["status"] == "accepted" for c in r["candidate_table"]) for r in results)
            lines.append(f"| {NAMES[arm]} | {stay}/{len(results)} | {accepted:.1f} |")
    return lines + [""]


def main() -> None:
    out = Path("artifacts/cross_task")
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    lines = ["# Cross-task comparison (report sets; mean ± sd over 10 seeds; differences vs. C, Welch)", ""]
    header = "| Arm | " + " | ".join(title for _, title, *_ in SETTINGS) + " |"
    lines += [header, "|---|" + "---:|" * len(SETTINGS)]
    data = {}
    for key, title, report, metric, pattern in SETTINGS:
        seed, arms = load_setting(report, metric, pattern)
        data[key] = {"title": title, "seed": seed, "arms": arms}
    lines.append("| Seed instruction | " + " | ".join(f"{data[k]['seed']:.2f}" for k, *_ in SETTINGS) + " |")
    for arm in ARMS:
        cells = []
        for key, *_ in SETTINGS:
            arms = data[key]["arms"]
            if arm not in arms:
                cells.append("—")
            elif arm == "C":
                cells.append(f"**{statistics.mean(arms['C']):.2f}** (reference)")
            else:
                diff, half, p = welch(arms[arm], arms["C"])
                summary.setdefault(key, {})[arm] = {"mean": statistics.mean(arms[arm]), "diff_vs_C": diff, "ci": half, "p": p}
                bold = "**" if p < 0.05 else ""
                cells.append(f"{statistics.mean(arms[arm]):.2f} ({bold}{diff:+.2f}{bold}, p={p:.3f})")
        lines.append(f"| {NAMES[arm]} | " + " | ".join(cells) + " |")
    lines += ["", "Cells: mean, then (difference vs. C, Welch p); bold = p < 0.05.", ""]
    # GoEmotions: hedging
    hedging = goemotions_hedging()
    lines += ["## GoEmotions: what the final prompts do on the report set", "",
              f"True emotions per comment: {hedging['true_per_comment']:.2f}.", "",
              "| Arm | Example F1 | Micro F1 | Macro F1 | Emotions predicted per comment | Precision | Recall |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ["seed", "A", "B", "E", "C", "I", "H", "G", "F"]:
        rows = hedging.get(arm)
        if rows:
            m = lambda k: statistics.mean(r[k] for r in rows)  # noqa: E731
            name = "Seed instruction" if arm == "seed" else NAMES[arm]
            lines.append(f"| {name} | {m('example_f1'):.1f} | {m('micro_f1'):.1f} | {m('macro_f1'):.1f} | "
                         f"{m('per_comment'):.2f} | {m('precision'):.1f} | {m('recall'):.1f} |")
    f_vs = {}
    for metric in ("example_f1", "micro_f1", "macro_f1"):
        f = [r[metric] for r in hedging["F"]]
        for ref in ("B", "C", "I", "H"):
            d, _, p = welch(f, [r[metric] for r in hedging[ref]])
            f_vs[f"{metric} F-{ref}"] = (d, p)
    lines += ["", "F vs. B, C and the equal-compute controls I, H (Welch):", ""] + [
        f"- {k}: {d:+.2f} (p={p:.3f})" for k, (d, p) in f_vs.items()] + [""]
    lines += label_name_ceiling()
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"settings": summary, "f_vs": f_vs}, indent=2), encoding="utf-8")
    print("\n".join(lines))

    # Figure 18: every arm minus C, with 95% CI, per task setting.
    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(17, 4.6), sharey=True)
    order = [a for a in ARMS if a != "C"][::-1]
    for ax, (key, title, *_rest) in zip(axes, SETTINGS):
        arms = data[key]["arms"]
        for y, arm in enumerate(order):
            if arm not in arms:
                ax.text(0.5, y, "not run", ha="center", va="center", fontsize=7, color=TEXT_2,
                        transform=ax.get_yaxis_transform())
                continue
            diff, half, p = welch(arms[arm], arms["C"])
            color = COLOR[FAMILY[arm]]
            ax.errorbar([diff], [y], xerr=[[half], [half]], fmt="o", color=color, ms=6, capsize=3, lw=1.5,
                        markerfacecolor=color if p < 0.05 else SURFACE)
        ax.axvline(0, color=REFERENCE, lw=1)
        ax.set_title(f"{title}\n(C = {statistics.mean(arms['C']):.1f}, seed = {data[key]['seed']:.1f})", fontsize=9)
        ax.set_xlabel("difference vs. C (pp)" if "F1" not in title else "difference vs. C (F1 points)", fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [NAMES[a] for a in order], fontsize=8)
    fig.suptitle("Each arm minus fixed-length ADMM (C), mean ± 95% CI over 10 seeds; filled = p < 0.05",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(wspace=0.12, top=0.8, bottom=0.14)
    fig.savefig("artifacts/figures/fig18_cross_task.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 19: GoEmotions hedging: emotions predicted per comment vs. example F1 and micro F1, one dot per run.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, metric, label in zip(axes, ("example_f1", "micro_f1"), ("example F1 (optimized)", "micro F1 (per-emotion pooled)")):
        for arm in ["A", "B", "E", "C", "I", "H", "G", "F"]:
            rows = hedging[arm]
            ax.scatter([r["per_comment"] for r in rows], [r[metric] for r in rows], s=28, color=COLOR[FAMILY[arm]],
                       marker={"A": "o", "B": "s", "E": "P", "C": "D", "I": "X", "H": "p", "G": "^", "F": "v"}[arm],
                       alpha=0.75, label=NAMES[arm])
        s = hedging["seed"][0]
        ax.scatter([s["per_comment"]], [s[metric]], s=110, marker="*", color=TEXT, label="seed instruction", zorder=5)
        ax.axvline(hedging["true_per_comment"], color=REFERENCE, ls="--", lw=1)
        ax.text(hedging["true_per_comment"] + 0.02, ax.get_ylim()[0] + 0.3, "true emotions\nper comment", fontsize=7,
                color=TEXT_2)
        ax.set_xlabel("emotions predicted per comment (report set)")
        ax.set_ylabel(label)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("GoEmotions: predicting more emotions raises example F1 but not micro F1 (one dot per run)",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(wspace=0.22, top=0.86, bottom=0.24)
    fig.savefig("artifacts/figures/fig19_goemotions_hedging.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("WROTE artifacts/figures/fig18_cross_task.png, fig19_goemotions_hedging.png")


if __name__ == "__main__":
    main()
