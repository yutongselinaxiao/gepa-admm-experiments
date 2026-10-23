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


def method_examples() -> list[str]:
    """Real accepted proposals of E (polish), F (gradient insertion) and G (gated) on Yahoo and GoEmotions.

    Effect on validation (diagnostic only): Yahoo = change in the number of correct answers (of 1,280);
    GoEmotions = change in mean comment F1, in points (192 comments)."""
    from .phase_7_10_analysis import word_diff

    tasks = [("Yahoo", "artifacts/phase11_yahoo/yahoo_{arm}_s*", False),
             ("GoEmotions", "artifacts/phase14_goemotions_f1/goemotions_{arm}_s*", True)]

    def effect(audit: dict, parent: str, child: str, multi: bool) -> float | None:
        if parent not in audit or child not in audit:
            return None
        p, c = audit[parent]["correct_by_id"], audit[child]["correct_by_id"]
        delta = sum(c[i] - p[i] for i in p)
        return 100 * delta / len(p) if multi else delta

    def unit(multi: bool) -> str:
        return "F1 points" if multi else "answers"

    lines = ["# Example proposals of E, F and G (accepted children; bold = inserted, strikethrough = deleted)", ""]
    for task, pattern, multi in tasks:
        lines += [f"## {task}", ""]
        # E: GEPA parent -> reflection draft -> ADMM polish (judged against the GEPA parent on S')
        rows = []
        for run in sorted(Path().glob(pattern.format(arm="E_polish_fair"))):
            if not run.is_dir():
                continue
            audit = json.loads((run / "gepa_result.json").read_text())["audit_by_instruction"]
            judged = {j["call_index"]: j for j in map(json.loads, (run / "heldout_judgements.jsonl").read_text().splitlines())}
            for p in map(json.loads, (run / "admm_proposals.jsonl").read_text().splitlines()):
                j = judged.get(p["call_index"])
                if p["outcome"] == "proposed" and j and j["heldout_accept"] and j["child_instruction"] == p["child_instruction"]:
                    e = effect(audit, j["parent_instruction"], p["child_instruction"], multi)
                    e_draft = effect(audit, j["parent_instruction"], p["parent_instruction"], multi)
                    if e is not None:
                        rows.append((e, e_draft, j["parent_instruction"], p["parent_instruction"], p["child_instruction"], run.name))
        rows.sort(key=lambda r: r[0])
        lines += [f"### E: reflect-then-polish ({len(rows)} accepted children where the polish changed the draft)", ""]
        for tag, r in (("largest effect", rows[-1]), ("median effect", rows[len(rows) // 2])) if rows else ():
            draft_effect = "n/a" if r[1] is None else f"{r[1]:+.1f}"
            lines += [f"**{tag}: {r[0]:+.1f} {unit(multi)} vs. GEPA's parent** (draft alone: {draft_effect}; {r[5]})", "",
                      f"- GEPA parent: {r[2]}", f"- reflection draft: {word_diff(r[2], r[3])}",
                      f"- after ADMM polish (vs. draft): {word_diff(r[3], r[4])}", ""]
        # F: raw +2 variant (gradient-chosen insertions) and the refined child that won
        rows = []
        for run in sorted(Path().glob(pattern.format(arm="F_mlgrad"))):
            if not run.is_dir():
                continue
            audit = json.loads((run / "gepa_result.json").read_text())["audit_by_instruction"]
            for p in map(json.loads, (run / "admm_proposals.jsonl").read_text().splitlines()):
                if p["outcome"] == "proposed" and p["heldout_accept"] and p["chosen"].get("start_offset", 0) > 0:
                    raw = next((v["text"] for v in p.get("length_variants", []) if v["offset"] > 0), None)
                    e = effect(audit, p["parent_instruction"], p["child_instruction"], multi)
                    if e is not None and raw:
                        rows.append((e, p["parent_instruction"], raw, p["child_instruction"], p["chosen"]["iteration"], run.name))
        rows.sort(key=lambda r: r[0])
        lines += [f"### F: gradient insertion ({len(rows)} accepted children from the 2-longer variant)", ""]
        for tag, r in (("largest effect", rows[-1]), ("median effect", rows[len(rows) // 2])) if rows else ():
            lines += [f"**{tag}: {r[0]:+.1f} {unit(multi)} vs. parent** (refined child from ADMM round {r[4]}; {r[5]})", "",
                      f"- parent: {r[1]}", f"- 2-longer variant (2 gradient-chosen insertions, before ADMM): {word_diff(r[1], r[2])}",
                      f"- accepted child (vs. parent): {word_diff(r[1], r[3])}", ""]
        # G: gated, with what the gates did at the chosen round
        rows = []
        for run in sorted(Path().glob(pattern.format(arm="G_gated"))):
            if not run.is_dir():
                continue
            audit = json.loads((run / "gepa_result.json").read_text())["audit_by_instruction"]
            for p in map(json.loads, (run / "admm_proposals.jsonl").read_text().splitlines()):
                if p["outcome"] == "proposed" and p["heldout_accept"]:
                    it = next((r for r in p["iterations"] if r["iteration"] == p["chosen"]["iteration"]), None)
                    e = effect(audit, p["parent_instruction"], p["child_instruction"], multi)
                    if e is not None and it:
                        rows.append((e, p["parent_instruction"], p["child_instruction"], it["filled_spare"], it["deleted_original"],
                                     p["chosen"]["start_offset"], run.name))
        rows.sort(key=lambda r: r[0])
        lines += [f"### G: gated ({len(rows)} accepted children)", ""]
        for tag, r in (("largest effect", rows[-1]), ("median effect", rows[len(rows) // 2])) if rows else ():
            lines += [f"**{tag}: {r[0]:+.1f} {unit(multi)} vs. parent** (spare slots opened: {r[3]}, original tokens "
                      f"deleted: {r[4]}, length change {r[5]:+d} tokens; {r[6]})", "",
                      f"- parent: {r[1]}", f"- accepted child: {word_diff(r[1], r[2])}", ""]
    lines += d_examples()
    return lines


def d_examples() -> list[str]:
    """D (multi-length, LM-top-token insertion) was only run on AG News (letters) and emotion (Phase 7):
    accepted children from its shorter and longer variants, with the raw variant before ADMM."""
    from .phase_7_10_analysis import word_diff

    lines = ["## D: multi-length, LM insertion (Phase 7; AG News with letters and emotion)", ""]
    for task in ("agnews", "emotion"):
        for offset, label in ((-2, "2-shorter"), (2, "2-longer")):
            rows = []
            for run in sorted(Path("artifacts/phase7").glob(f"{task}_multilength_s*")):
                if not run.is_dir():
                    continue
                audit = json.loads((run / "gepa_result.json").read_text())["audit_by_instruction"]
                for p in map(json.loads, (run / "admm_proposals.jsonl").read_text().splitlines()):
                    if p["outcome"] == "proposed" and p["heldout_accept"] and p["chosen"].get("start_offset", 0) == offset:
                        raw = next((v["text"] for v in p.get("length_variants", []) if v["offset"] == offset), None)
                        pa, ch = p["parent_instruction"], p["child_instruction"]
                        if raw and pa in audit and ch in audit:
                            e = sum(audit[ch]["correct_by_id"][i] - audit[pa]["correct_by_id"][i] for i in audit[pa]["correct_by_id"])
                            rows.append((e, pa, raw, ch, p["chosen"]["iteration"], run.name))
            rows.sort(key=lambda r: r[0])
            name = "AG News" if task == "agnews" else "Emotion"
            lines += [f"### {name}, {label} variant ({len(rows)} accepted)", ""]
            for tag, r in (("largest effect", rows[-1]), ("median effect", rows[len(rows) // 2])) if rows else ():
                lines += [f"**{tag}: {r[0]:+d} answers of 512 vs. parent** (ADMM round {r[4]}; {r[5]})", "",
                          f"- parent: {r[1]}", f"- {label} variant (before ADMM): {word_diff(r[1], r[2])}",
                          f"- accepted child (vs. parent): {word_diff(r[1], r[3])}", ""]
    return lines


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
    (out / "examples.md").write_text("\n".join(method_examples()) + "\n", encoding="utf-8")
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
    scorecard(data)


SHORT = {"agnews": "AG News\n(letters)", "emotion": "Emotion", "yahoo": "Yahoo", "agnews_names": "AG News\n(label names)",
         "goemotions_example": "GoEmotions\nexample F1", "goemotions_micro": "GoEmotions\nmicro F1"}


def scorecard(data: dict) -> None:
    """Figure 20 + table: every arm x setting, difference vs. C (color scaled per column, value printed,
    bold = p < 0.05); and F / G vs. the equal-compute controls I and H."""
    import matplotlib.colors as mcolors

    rows = [a for a in ARMS if a != "C"]
    cols = [k for k, *_ in SETTINGS]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    cmap = mcolors.LinearSegmentedColormap.from_list("div", ["#c2410c", "#f7f6f2", "#1d6fa5"])
    for j, key in enumerate(cols):
        arms = data[key]["arms"]
        diffs = {a: welch(arms[a], arms["C"]) for a in rows if a in arms}
        for i, a in enumerate(rows):
            if a not in diffs:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color="#ebeae6"))
                ax.text(j + 0.5, i + 0.5, "not run", ha="center", va="center", fontsize=7, color=TEXT_2)
                continue
            d, half, p = diffs[a]
            strength = max(-2.5, min(2.5, d / half)) / 2.5      # evidence: difference / 95% CI half-width
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cmap(0.5 + 0.5 * strength)))
            ax.text(j + 0.5, i + 0.5, f"{d:+.1f}" + ("*" if p < 0.05 else ""), ha="center", va="center",
                    fontsize=9, color=TEXT, fontweight="bold" if p < 0.05 else "normal")
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([j + 0.5 for j in range(len(cols))],
                  [f"{SHORT[k]}\n(C = {statistics.mean(data[k]['arms']['C']):.1f})" for k in cols], fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(rows))], [NAMES[a] for a in rows], fontsize=8.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    fig.suptitle("Scorecard: each arm minus fixed-length ADMM (C) on every task setting", fontsize=11,
                 fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.03, "Cells: mean difference over 10 seeds (pp of accuracy, or F1 points for GoEmotions); "
             "* and bold = p < 0.05 (Welch).\nColor = strength of evidence: difference / its 95% CI half-width, clipped at 2.5 "
             "(blue = better than C, orange = worse; |ratio| ≥ 1 ⇔ p < 0.05; pale = no clear difference).",
             fontsize=7.5, color=TEXT_2)
    fig.savefig("artifacts/figures/fig20_scorecard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    lines = ["## F and G vs. the equal-compute controls (difference, Welch p)", "",
             "| Setting | F − I | F − H | G − I | G − H |", "|---|---:|---:|---:|---:|"]
    for key in cols:
        arms = data[key]["arms"]
        if not all(a in arms for a in "FGIH"):
            continue
        cells = []
        for a, b in (("F", "I"), ("F", "H"), ("G", "I"), ("G", "H")):
            d, _, p = welch(arms[a], arms[b])
            cells.append(f"{'**' if p < 0.05 else ''}{d:+.2f}{'**' if p < 0.05 else ''} (p={p:.3f})")
        lines.append(f"| {data[key]['title']} | " + " | ".join(cells) + " |")
    with open("artifacts/cross_task/tables.md", "a", encoding="utf-8") as handle:
        handle.write("\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print("WROTE artifacts/figures/fig20_scorecard.png")


if __name__ == "__main__":
    main()
