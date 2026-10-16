"""Worked examples and plots that explain the proposal diagnostics (see proposal_diagnostics.py).

- Figure 9b: per proposer, the average number of validation examples a child fixes, breaks, or
  changes from one wrong label to another, relative to its parent.
- Worked examples: for one typical reflection proposal and one typical beam-ADMM proposal, the
  full breakdown of the 512 validation predictions and a few real articles from each category.
- Acceptance evidence: 2x2 tables (accepted/rejected x actually better/not) and real examples of
  children GEPA's minibatch rule rejected although they were better on validation.

Writes artifacts/proposal_diagnostics/examples.md and artifacts/figures/fig9b_fixes_vs_breaks.png.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .make_figures import GRID, REFERENCE, SURFACE, TEXT, TEXT_2  # noqa: F401  (shared style)

FIX, BREAK, WRONG = "#4f4e4b", "#a8a7a2", "#d9d8d3"
VALIDATION = {"agnews": "artifacts/data/validation.json", "emotion": "artifacts/data_emotion/validation.json"}
NAMES = {"agnews": {"A": "World", "B": "Sports", "C": "Business", "D": "Sci/Tech"},
         "emotion": {"A": "Sadness", "B": "Joy", "C": "Love", "D": "Anger", "E": "Fear", "F": "Surprise"}}


def load_rows() -> list[dict]:
    rows = [json.loads(line) for line in Path("artifacts/proposal_diagnostics/proposals.jsonl").read_text().splitlines()]
    for r in rows:
        r["flips"] = round(r["flip_rate"] * 512)
        r["wrong_to_wrong"] = r["flips"] - r["fixes"] - r["breaks"]
    return rows


def breakdown(run: str, parent: str, child: str) -> dict:
    audit = json.loads(Path(run, "gepa_result.json").read_text())["audit_by_instruction"]
    pa, ch = audit[parent], audit[child]
    cats: dict[str, list[str]] = defaultdict(list)
    for i in pa["predicted_by_id"]:
        p_ok, c_ok = pa["correct_by_id"][i], ch["correct_by_id"][i]
        same = pa["predicted_by_id"][i] == ch["predicted_by_id"][i]
        if same:
            cats["unchanged, correct" if p_ok else "unchanged, wrong"].append(i)
        elif c_ok and not p_ok:
            cats["fix (wrong → right)"].append(i)
        elif p_ok and not c_ok:
            cats["break (right → wrong)"].append(i)
        else:
            cats["wrong → different wrong"].append(i)
    return {"cats": cats, "pa": pa, "ch": ch}


def fmt_label(task: str, label: str) -> str:
    return f"{label} ({NAMES[task][label]})"


def worked_example(task: str, row: dict, title: str) -> list[str]:
    articles = {e["example_id"]: e for e in json.loads(Path(VALIDATION[task]).read_text())}
    b = breakdown(row["run"], row["parent"], row["child"])
    cats = b["cats"]
    order = ["unchanged, correct", "unchanged, wrong", "fix (wrong → right)", "break (right → wrong)",
             "wrong → different wrong"]
    lines = [f"**{title}**", "",
             f"- Parent: “{row['parent']}”",
             f"- Child: “{row['child']}”",
             f"- Parent accuracy {100 * b['pa']['accuracy']:.2f}% → child {100 * b['ch']['accuracy']:.2f}% "
             f"({row['accuracy_change_pp']:+.2f} pp).", "",
             "| Category (512 validation examples) | Count | Counted as a flip? |", "|---|---:|---|"]
    for c in order:
        lines.append(f"| {c} | {len(cats[c])} | {'yes' if c not in ('unchanged, correct', 'unchanged, wrong') else 'no'} |")
    flips = sum(len(cats[c]) for c in order[2:])
    lines += [f"| **flips (predictions changed)** | **{flips}** ({100 * flips / 512:.1f}%) | |",
              f"| net = fixes − breaks | {len(cats[order[2]]) - len(cats[order[3]]):+d} examples "
              f"= {row['accuracy_change_pp']:+.2f} pp | |", "",
              "| Category | Article (first 110 characters) | Gold | Parent predicted | Child predicted |",
              "|---|---|---|---|---|"]
    for c in order[2:]:
        for i in cats[c][:2]:
            text = articles[i]["article"].replace("|", "/").replace("\n", " ")[:110]
            lines.append(f"| {c} | {text}… | {fmt_label(task, articles[i]['gold_label'])} | "
                         f"{fmt_label(task, b['pa']['predicted_by_id'][i])} | {fmt_label(task, b['ch']['predicted_by_id'][i])} |")
    return lines + [""]


def typical(rows: list[dict], task: str, proposer: str, want_net: str) -> dict:
    sub = [r for r in rows if r["task"] == task and r["proposer"] == proposer]
    flips = statistics.median(r["flips"] for r in sub)
    if want_net == "balanced":
        key = lambda r: abs(r["flips"] - flips) + 3 * abs(r["fixes"] - r["breaks"])  # noqa: E731
    else:
        net = statistics.median(r["fixes"] - r["breaks"] for r in sub)
        key = lambda r: abs(r["flips"] - flips) + 3 * abs((r["fixes"] - r["breaks"]) - net)  # noqa: E731
    return min(sub, key=key)


def minibatch_scores(run: str) -> dict[int, tuple[float, float]]:
    """iteration -> (parent score, child score) on GEPA's 8-example minibatch, for decided proposals."""
    out = {}
    for line in Path(run, "callback_events.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "candidate_rejected":
            out[e["data"]["iteration"]] = (e["data"]["old_score"], e["data"]["new_score"])
    return out


def rejected_but_better(rows: list[dict], proposer: str, rule: str, k: int = 3) -> list[str]:
    sub = [r for r in rows if r["task"] == "agnews" and r["proposer"] == proposer and r["rule"] == rule
           and r["status"] == "rejected" and r["accuracy_change_pp"] > 0]
    sub.sort(key=lambda r: -r["accuracy_change_pp"])
    lines = ["| Minibatch score: parent → child (out of 8) | Validation accuracy change (512 ex.) | Child prompt |",
             "|---|---:|---|"]
    shown = 0
    for r in sub:
        table = json.loads(Path(r["run"], "gepa_result.json").read_text())["candidate_table"]
        iteration = next((p["iteration"] for p in table if p["instruction"] == r["child"]
                          and p["parent_instruction"] == r["parent"]), None)
        scores = minibatch_scores(r["run"]).get(iteration)
        if not scores:
            continue
        child = r["child"].replace("|", "/").replace("\n", " ⏎ ")
        verdict = "tie" if scores[0] == scores[1] else "lower"
        lines.append(f"| {scores[0]:.0f} → {scores[1]:.0f} ({verdict}: rejected) | {r['accuracy_change_pp']:+.2f} pp | {child} |")
        shown += 1
        if shown == k:
            break
    return lines + [""]


def contingency(rows: list[dict], task: str, proposer: str, rule: str) -> list[str]:
    sub = [r for r in rows if r["task"] == task and r["proposer"] == proposer and r["rule"] == rule]
    cell = lambda status, better: sum(1 for r in sub if r["status"] == status and (r["accuracy_change_pp"] > 0) == better)  # noqa: E731
    return [f"| {proposer} / {rule} | {cell('accepted', True)} | {cell('accepted', False)} | "
            f"{cell('rejected', True)} | {cell('rejected', False)} |"]


def main() -> None:
    rows = load_rows()
    md: list[str] = []
    # Figure 9b
    groups = [("agnews", ["reflection", "Euclidean ADMM", "top-p ADMM", "beam ADMM", "reflect-then-polish"]),
              ("emotion", ["reflection", "top-p ADMM", "beam ADMM"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [5, 3.3]})
    for ax, (task, proposers) in zip(axes, groups):
        ys = list(range(len(proposers)))[::-1]
        for y, p in zip(ys, proposers):
            sub = [r for r in rows if r["task"] == task and r["proposer"] == p]
            fixes = statistics.mean(r["fixes"] for r in sub)
            breaks = statistics.mean(r["breaks"] for r in sub)
            w2w = statistics.mean(r["wrong_to_wrong"] for r in sub)
            ax.barh(y, -breaks, color=BREAK, height=0.6, edgecolor=SURFACE)
            ax.barh(y, fixes, color=FIX, height=0.6, edgecolor=SURFACE)
            ax.barh(y, w2w, left=fixes, color=WRONG, height=0.6, edgecolor=SURFACE)
            ax.text(-breaks / 2, y, f"{breaks:.1f}", ha="center", va="center", fontsize=8, color=TEXT)
            ax.text(fixes / 2, y, f"{fixes:.1f}", ha="center", va="center", fontsize=8, color="white")
            ax.text(fixes + w2w + 1, y, f"net {fixes - breaks:+.1f}", ha="left", va="center", fontsize=8, color=TEXT)
        ax.axvline(0, color=TEXT, lw=0.8)
        ax.set_yticks(ys, proposers, fontsize=8.5)
        ax.set_xlim(-48, 120)
        ax.set_title("AG News" if task == "agnews" else "Emotion")
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BREAK, FIX, WRONG)]
    fig.legend(handles, ["breaks (right → wrong)", "fixes (wrong → right)", "wrong → different wrong"],
               loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("What a typical child changes relative to its parent (mean per proposal)", fontsize=11,
                 fontweight="bold", x=0.01, ha="left")
    fig.supxlabel("validation examples per proposal (out of 512):  ← breaks   |   fixes, then wrong → different wrong →",
                  fontsize=9, color=TEXT_2, y=0.04)
    fig.subplots_adjust(wspace=0.3, bottom=0.22, top=0.84)
    fig.savefig("artifacts/figures/fig9b_fixes_vs_breaks.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    md += worked_example("agnews", typical(rows, "agnews", "reflection", "balanced"),
                         "Example A: a typical reflection proposal (AG News)")
    md += worked_example("agnews", typical(rows, "agnews", "beam ADMM", "typical"),
                         "Example B: a typical beam ADMM proposal (AG News)")
    md += ["#### 2×2 tables: acceptance decision vs. whether the child was actually better (AG News)", "",
           "\"Better\" = higher accuracy than the parent on the 512 validation examples.", "",
           "| Proposer / acceptance rule | Accepted & better | Accepted & not better | Rejected & better | Rejected & not better |",
           "|---|---:|---:|---:|---:|"]
    for proposer, rule in [("reflection", "minibatch (GEPA default)"), ("reflection", "held-out judge"),
                           ("Euclidean ADMM", "minibatch (GEPA default)"),
                           ("Euclidean ADMM", "held-out on S (S used twice)"),
                           ("beam ADMM", "held-out on S (S used twice)"), ("beam ADMM", "held-out on independent S'")]:
        md += contingency(rows, "agnews", proposer, rule)
    md += ["", "#### Rejected by GEPA's minibatch rule, yet better on validation (AG News, largest gains)", "",
           "**Reflection** (12-proposal runs):", ""]
    md += rejected_but_better(rows, "reflection", "minibatch (GEPA default)")
    md += ["**Euclidean ADMM proposer** (Phase 1):", ""]
    md += rejected_but_better(rows, "Euclidean ADMM", "minibatch (GEPA default)")
    md += ["", "#### Average child vs. what GEPA keeps (like-for-like runs, 512 validation examples)", ""]
    md += outcome_decomposition()
    Path("artifacts/proposal_diagnostics/examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("WROTE artifacts/figures/fig9b_fixes_vs_breaks.png")
    tail_figure()



def tail_figure() -> None:
    """Figure 9c: share of children whose net gain (fixes - breaks) is at least x, per proposer.

    Like-for-like runs only (one experiment per proposer, 10 runs x 12 proposals)."""
    from .make_figures import COLOR, MARKER  # noqa: F401
    rows = load_rows()
    sets = {
        "agnews": [("reflection", "artifacts/phase2/reflective_s", "reflective", "-"),
                   ("top-p ADMM", "artifacts/phase2/admm_topp_s", "topp", "--"),
                   ("beam ADMM (fair S')", "phase6_independent_acceptance/agnews", "beam", ":")],
        "emotion": [("reflection", "phase4_emotion/reflective_p12_s", "reflective", "-"),
                    ("top-p ADMM", "phase4_emotion/admm_topp_s", "topp", "--"),
                    ("beam ADMM (fair S')", "phase6_independent_acceptance/emotion", "beam", ":")],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (task, series) in zip(axes, sets.items()):
        for name, key, color, style in series:
            nets = sorted(r["fixes"] - r["breaks"] for r in rows if r["task"] == task and key in r["run"])
            xs = list(range(-40, 61))
            ax.plot(xs, [100 * sum(n >= x for n in nets) / len(nets) for x in xs], color=COLOR[color], ls=style,
                    label=f"{name} (n = {len(nets)})")
        ax.axvline(0, color=REFERENCE, lw=1)
        ax.axvline(5, color=REFERENCE, lw=1, ls=":")
        ax.text(5.8, 45, "net ≥ +5", fontsize=8, color=TEXT_2)
        ax.set_xlim(-40, 60)
        ax.set_xlabel("net gain x of a child over its parent (fixes − breaks, of 512 examples)")
        ax.set_title("AG News" if task == "agnews" else "Emotion")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("children with net gain ≥ x (%)")
    fig.suptitle("Upper tail of child quality: how often a proposer produces a clearly better child",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(top=0.84, wspace=0.08)
    fig.savefig("artifacts/figures/fig9c_child_quality_tail.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("WROTE artifacts/figures/fig9c_child_quality_tail.png")


LIKE_FOR_LIKE = {
    ("agnews", "reflection"): "artifacts/phase2/reflective_s",
    ("agnews", "top-p ADMM"): "artifacts/phase2/admm_topp_s",
    ("agnews", "beam ADMM (fair S')"): "phase6_independent_acceptance/agnews",
    ("emotion", "reflection"): "phase4_emotion/reflective_p12_s",
    ("emotion", "top-p ADMM"): "phase4_emotion/admm_topp_s",
    ("emotion", "beam ADMM (fair S')"): "phase6_independent_acceptance/emotion",
}


def outcome_decomposition() -> list[str]:
    """Average child vs. what GEPA keeps: all children, the upper tail, accepted children, the best
    child per run, and the final selected prompt vs. the seed (all on the 512 validation examples)."""
    rows = load_rows()
    lines = ["| Task / proposer | Mean net, all children | Children with net ≥ +5 | Accepted: count, mean net | "
             "Best child per run (mean) | Final prompt vs. seed (mean) |", "|---|---:|---:|---:|---:|---:|"]
    for (task, name), key in LIKE_FOR_LIKE.items():
        sub = [r for r in rows if r["task"] == task and key in r["run"]]
        net = [r["fixes"] - r["breaks"] for r in sub]
        accepted = [r["fixes"] - r["breaks"] for r in sub if r["status"] == "accepted"]
        runs = sorted({r["run"] for r in sub})
        best = [max(r["fixes"] - r["breaks"] for r in sub if r["run"] == run) for run in runs]
        final = []
        for run in runs:
            result = json.loads(Path(run, "gepa_result.json").read_text(encoding="utf-8"))
            audit, seed = result["audit_by_instruction"], result["all_accepted_candidates"][0]["instruction"]
            final.append(512 * (audit[result["best_instruction"]]["accuracy"] - audit[seed]["accuracy"]))
        lines.append(f"| {'AG News' if task == 'agnews' else 'Emotion'} / {name} | {statistics.mean(net):+.1f} | "
                     f"{100 * sum(x >= 5 for x in net) / len(net):.0f}% | {len(accepted)}, "
                     f"{statistics.mean(accepted):+.1f} | {statistics.mean(best):+.1f} | {statistics.mean(final):+.1f} |")
    return lines


if __name__ == "__main__":
    main()
