"""Phases 7-10: every arm on the 2,000-example report sets, accuracy vs. ADMM compute, and the
proposal-level bench of the variable-length / multi-start proposers.

Arms (10 seeds each, same protocol: selection split, minibatch 8, 12 GEPA proposals unless noted):
  A  reflective GEPA (minibatch acceptance)       B  matched reflective (30 proposals, held-out judge)
  A' reflective + held-out judge (12; AG only)    C  fair beam ADMM (fixed length, S' acceptance)
  D  multi-length, LM insertion (Phase 7)         E  fair reflect-then-polish (Phase 7)
  G  gated variable-length ADMM (Phase 8)         I  multi-start, 3 ADMM runs per proposal (Phase 9)
  H  fixed beam ADMM, 36 proposals (Phase 9)      F  multi-length, gradient insertion (Phase 10)

Writes artifacts/phase_7_10/{tables.md,summary.json} and figures fig12-fig14.
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

from .make_figures import COLOR, REFERENCE, SURFACE, TEXT, TEXT_2  # noqa: E402  (shared style)

FAMILY = {"reflection": COLOR["reflective"], "reflection + ADMM": COLOR["euclidean"],
          "fixed-length ADMM": COLOR["beam"], "variable-length ADMM": COLOR["topp"]}


def arms(task: str) -> list[tuple[str, str, str]]:
    """(label, run-key regex, family)."""
    t = task
    rows = [
        ("A  reflective GEPA", r"phase2/reflective_s\d" if t == "agnews" else r"phase4_emotion/reflective_p12_s\d", "reflection"),
        ("A' reflective + held-out judge", r"phase3/reflective_heldout_p12_s\d", "reflection"),
        ("B  matched reflective (30)", r"phase3/reflective_heldout_p30_s\d" if t == "agnews"
         else r"phase4_emotion/reflective_heldout_p30_s\d", "reflection"),
        ("E  fair reflect-then-polish", rf"phase7/{t}_polish_fair_s\d", "reflection + ADMM"),
        ("C  fair beam ADMM", rf"phase6_independent_acceptance/{t}_beam_indep_s\d", "fixed-length ADMM"),
        ("I  multi-start (3 runs)", rf"phase9/{t}_multistart_s\d", "fixed-length ADMM"),
        ("H  fixed beam, 36 proposals", rf"phase9/{t}_beam_p36_s\d", "fixed-length ADMM"),
        ("D  multi-length, LM insertion", rf"phase7/{t}_multilength_s\d", "variable-length ADMM"),
        ("F  multi-length, gradient ins.", rf"phase10/{t}_mlgrad_s\d", "variable-length ADMM"),
        ("G  gated", rf"phase8/{t}_gated_s\d", "variable-length ADMM"),
    ]
    return [r for r in rows if not (t == "emotion" and r[0].startswith("A'"))]


def ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    half = stats.t.ppf(0.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0
    return mean, half


def admm_forward_calls(run: str) -> float | None:
    result = json.loads(Path(run, "gepa_result.json").read_text(encoding="utf-8"))
    return (result.get("admm_proposer") or {}).get("forward_calls")


def net_by_pair(run: str) -> dict[tuple[str, str], int]:
    result = json.loads(Path(run, "gepa_result.json").read_text(encoding="utf-8"))
    audit, nets = result["audit_by_instruction"], {}
    for c in result["candidate_table"]:
        parent, child = c["parent_instruction"], c["instruction"]
        if parent in audit and child in audit and parent != child:
            p, ch = audit[parent]["correct_by_id"], audit[child]["correct_by_id"]
            nets[(parent, child)] = sum(ch[i] and not p[i] for i in p) - sum(p[i] and not ch[i] for i in p)
    return nets


def phase7_diagnostics() -> tuple[list[str], dict]:
    """Which length variant wins in multi-length runs, and what the polish does (all Phase 7 runs)."""
    lines, out = ["## Multi-length variants and polish outcomes (all runs; validation net vs. parent, of 512)", ""], {}
    lines += ["| Arm | Task | Variant | Proposed | Accepted on S' | Mean validation net | Share better |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for (arm, phase, stem), task in [(a, t) for a in (("D", "phase7", "multilength"), ("F", "phase10", "mlgrad"))
                                     for t in ("agnews", "emotion")]:
        stats_by: dict[int, dict[str, list]] = {}
        for run in sorted(str(p) for p in Path(f"artifacts/{phase}").glob(f"{task}_{stem}_s*") if p.is_dir()):
            nets = net_by_pair(run)
            for line in Path(run, "admm_proposals.jsonl").read_text().splitlines():
                p = json.loads(line)
                if p["outcome"] != "proposed":
                    continue
                o = p["chosen"].get("start_offset", 0)
                row = stats_by.setdefault(o, {"accepted": [], "nets": []})
                row["accepted"].append(bool(p["heldout_accept"]))
                if (p["parent_instruction"], p["child_instruction"]) in nets:
                    row["nets"].append(nets[(p["parent_instruction"], p["child_instruction"])])
        for o in sorted(stats_by):
            r = stats_by[o]
            out[f"{arm}/{task}/{o:+d}"] = {"proposed": len(r["accepted"]), "accepted": sum(r["accepted"]),
                                           "mean_net": statistics.mean(r["nets"]) if r["nets"] else None}
            lines.append(f"| {arm} | {task} | {o:+d} tokens | {len(r['accepted'])} | {sum(r['accepted'])} | "
                         f"{statistics.mean(r['nets']):+.1f} | {100 * sum(x > 0 for x in r['nets']) / len(r['nets']):.0f}% |")
    lines += ["", "| Task | Polish outcome on reflection's draft | Count | Share |", "|---|---|---:|---:|"]
    for task in ("agnews", "emotion"):
        outcomes: dict[str, int] = {}
        for run in Path("artifacts/phase7").glob(f"{task}_polish_fair_s*"):
            if run.is_dir():
                for line in Path(run, "admm_proposals.jsonl").read_text().splitlines():
                    o = json.loads(line)["outcome"]
                    outcomes[o] = outcomes.get(o, 0) + 1
        total = sum(outcomes.values())
        out[f"{task}/polish_outcomes"] = outcomes
        for o, n in sorted(outcomes.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {task} | {o} | {n} | {100 * n / total:.0f}% |")
    return lines + [""], out


FOCUS = {
    "fig15_q1_multistart.png": ("Q1. Do parallel ADMM runs per proposal help?",
                                ["C  fair beam ADMM", "I  multi-start (3 runs)", "H  fixed beam, 36 proposals",
                                 "B  matched reflective (30)"]),
    "fig16_q2_variable_length.png": ("Q2. Variable-length ADMM proposers vs. fixed length",
                                     ["C  fair beam ADMM", "I  multi-start (3 runs)", "D  multi-length, LM insertion",
                                      "F  multi-length, gradient ins.", "G  gated"]),
    "fig17_q3_polish.png": ("Q3. Reflect-then-polish vs. reflection alone",
                            ["A  reflective GEPA", "A' reflective + held-out judge", "B  matched reflective (30)",
                             "E  fair reflect-then-polish", "C  fair beam ADMM"]),
}
DISPLAY = {
    "A  reflective GEPA": "A: reflection, minibatch rule (12)",
    "A' reflective + held-out judge": "A′: reflection, S′ judge (12)",
    "B  matched reflective (30)": "B: reflection, S′ judge (30)",
    "C  fair beam ADMM": "C: fixed-length beam ADMM",
    "D  multi-length, LM insertion": "D: multi-length, LM insertion",
    "E  fair reflect-then-polish": "E: reflection + ADMM polish, S′ judge",
    "F  multi-length, gradient ins.": "F: multi-length, gradient insertion",
    "G  gated": "G: gated variable length",
    "H  fixed beam, 36 proposals": "H: fixed-length ADMM, 36 proposals",
    "I  multi-start (3 runs)": "I: multi-start, 3 ADMM runs/proposal",
}


def focus_figures(data: dict) -> None:
    """One small figure per question: the arms that answer it, each run as a dot, mean ± 95% CI,
    with fair beam ADMM (C) as a shaded reference band."""
    for name, (title, labels) in FOCUS.items():
        fig, axes = plt.subplots(2, 1, figsize=(10, 2 * (0.5 * len(labels) + 1.1)))
        for ax, task in zip(axes, data):
            rows = data[task]["arms"]
            shown = [l for l in labels if l in rows][::-1]
            c_mean, c_half = ci(rows["C  fair beam ADMM"]["scores"])
            ax.axvspan(c_mean - c_half, c_mean + c_half, color=FAMILY["fixed-length ADMM"], alpha=0.08, lw=0)
            for y, label in enumerate(shown):
                row = rows[label]
                color = FAMILY[row["family"]]
                ax.scatter(row["scores"], [y] * len(row["scores"]), s=20, color=color, alpha=0.5, zorder=3)
                mean, half = ci(row["scores"])
                ax.errorbar([mean], [y], xerr=[[half], [half]], fmt="o", color=color, ms=6, capsize=3, lw=1.6, zorder=4)
                if label != "C  fair beam ADMM":
                    p = float(stats.ttest_ind(row["scores"], rows["C  fair beam ADMM"]["scores"], equal_var=False).pvalue)
                    note = f"{mean:.2f}   {mean - c_mean:+.2f} vs C (p = {p:.3f})"
                else:
                    note = f"{mean:.2f}   reference"
                ax.text(1.01, y, note, transform=ax.get_yaxis_transform(), va="center", fontsize=7.5,
                        color=TEXT if label == "C  fair beam ADMM" or p >= 0.05 else TEXT, fontweight="bold" if
                        label != "C  fair beam ADMM" and p < 0.05 else "normal")
            ax.axvline(data[task]["seed"], color=REFERENCE, lw=1, ls="--")
            ax.set_yticks(range(len(shown)), [DISPLAY[l] for l in shown], fontsize=8)
            ax.set_xlabel("accuracy on the 2k report set (%)")
            ax.set_title("AG News" if task == "agnews" else "Emotion", fontsize=10)
            ax.grid(axis="y", visible=False)
        fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
        fig.text(0.01, 0.005, "Dots: one per seed. Bars: mean ± 95% CI. Shaded band: C's 95% CI. Dashed line: seed "
                 "instruction. Bold: p < 0.05 vs C (Welch).", fontsize=7.5, color=TEXT_2)
        fig.subplots_adjust(hspace=0.55, top=0.9, bottom=0.12, right=0.68)
        fig.savefig(f"artifacts/figures/{name}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def seed_planning(data: dict) -> list[str]:
    """Seeds per arm for 80% power (two-sided alpha = 0.05) to detect a given difference, using the
    pooled sd of the two arms at n = 10 (normal approximation: n = 2 (1.96 + 0.84)^2 sd^2 / d^2)."""
    pairs = [("agnews", "F  multi-length, gradient ins.", "I  multi-start (3 runs)"),
             ("agnews", "F  multi-length, gradient ins.", "H  fixed beam, 36 proposals"),
             ("agnews", "G  gated", "C  fair beam ADMM"),
             ("agnews", "I  multi-start (3 runs)", "C  fair beam ADMM"),
             ("agnews", "E  fair reflect-then-polish", "A' reflective + held-out judge"),
             ("emotion", "G  gated", "C  fair beam ADMM"),
             ("emotion", "I  multi-start (3 runs)", "C  fair beam ADMM")]
    lines = ["## Seeds needed (80% power, two-sided α = 0.05)", "",
             "| Task | Comparison | Observed diff | Pooled sd | Seeds/arm for the observed diff | for 1 pp | for 2 pp |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for task, a, b in pairs:
        x, y = data[task]["arms"][a]["scores"], data[task]["arms"][b]["scores"]
        diff = statistics.mean(x) - statistics.mean(y)
        sd = math.sqrt((statistics.variance(x) + statistics.variance(y)) / 2)
        need = lambda d: math.ceil(2 * (1.96 + 0.8416) ** 2 * sd ** 2 / d ** 2)  # noqa: E731
        lines.append(f"| {task} | {a.split('  ')[0]} vs {b.split('  ')[0]} | {diff:+.2f} | {sd:.2f} | "
                     f"{need(abs(diff)) if abs(diff) > 0.05 else '—'} | {need(1.0)} | {need(2.0)} |")
    return lines + [""]


def acceptance_evidence() -> list[str]:
    """Polish (and reflection) children under the two acceptance rules: accepted/rejected counts, how
    many were actually better on the 512 validation examples, and how many minibatch rejections were
    exact ties on the 8 minibatch examples."""
    import glob

    from .proposal_examples import minibatch_scores

    groups = [
        ("AG News", "reflect-then-polish (old)", "minibatch (8 examples)", "artifacts/phase2/polish_s[0-9]", True),
        ("AG News", "reflect-then-polish (E)", "S′ judge (32 examples)", "artifacts/phase7/agnews_polish_fair_s[0-9]", False),
        ("AG News", "reflection (A)", "minibatch (8 examples)", "artifacts/phase2/reflective_s[0-9]", True),
        ("AG News", "reflection (A′)", "S′ judge (32 examples)", "artifacts/phase3/reflective_heldout_p12_s[0-9]", False),
        ("Emotion", "reflect-then-polish (E)", "S′ judge (32 examples)", "artifacts/phase7/emotion_polish_fair_s[0-9]", False),
    ]
    lines = ["## Acceptance evidence (children vs. parent on 512 validation examples)", "",
             "| Task | Proposer | Rule | Runs | Accepted | …actually better (mean net) | Rejected | "
             "…actually better (mean net) | Rejections that tied on the minibatch |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for task, proposer, rule, pattern, minibatch in groups:
        runs = sorted(glob.glob(pattern))
        accepted, rejected, ties, decided = [], [], 0, 0
        for run in runs:
            nets = net_by_pair(run)
            result = json.loads(Path(run, "gepa_result.json").read_text(encoding="utf-8"))
            for c in result["candidate_table"]:
                key = (c["parent_instruction"], c["instruction"])
                if key in nets:
                    (accepted if c["status"] == "accepted" else rejected).append(nets[key])
            if minibatch:
                for old, new in minibatch_scores(run).values():
                    decided += 1
                    ties += abs(old - new) < 1e-9
        share = lambda xs: f"{100 * sum(x > 0 for x in xs) / len(xs):.0f}% ({statistics.mean(xs):+.1f})"  # noqa: E731
        lines.append(f"| {task} | {proposer} | {rule} | {len(runs)} | {len(accepted)} | {share(accepted)} | "
                     f"{len(rejected)} | {share(rejected)} | {f'{ties}/{decided}' if minibatch else '—'} |")
    return lines + [""]


def candidate_pool_effect() -> list[str]:
    """Within beam ADMM proposals: does a larger pool of candidates competing on S go with a worse
    chosen child on validation? (Correlational check of the 'picking from more candidates overfits S'
    hypothesis.)"""
    import glob

    lines = ["## Candidate-pool size vs. validation net of the chosen child (beam ADMM proposals)", "",
             "| Task | Proposals | Median candidates on S | Few: n, mean net | Many: n, mean net | Spearman ρ (p) |",
             "|---|---:|---:|---:|---:|---:|"]
    for task, patterns in (("agnews", ["artifacts/phase6_independent_acceptance/agnews_beam_indep_s*",
                                       "artifacts/phase2/beam_lam0.3_s*"]),
                           ("emotion", ["artifacts/phase6_independent_acceptance/emotion_beam_indep_s*",
                                        "artifacts/phase4_emotion/beam_lam0.3_s*"])):
        rows = []
        for run in sorted(d for p in patterns for d in glob.glob(p) if Path(d).is_dir()):
            nets = net_by_pair(run)
            for line in Path(run, "admm_proposals.jsonl").read_text().splitlines():
                p = json.loads(line)
                if p["outcome"] != "proposed":
                    continue
                n = sum(1 for it in p["iterations"] if (it.get("new_projection") or {}).get("selection_loss") is not None)
                key = (p["parent_instruction"], p["child_instruction"])
                if key in nets:
                    rows.append((n, nets[key]))
        median = statistics.median(r[0] for r in rows)
        few, many = [r[1] for r in rows if r[0] <= median], [r[1] for r in rows if r[0] > median]
        rho, pval = stats.spearmanr([r[0] for r in rows], [r[1] for r in rows])
        lines.append(f"| {task} | {len(rows)} | {median:.0f} | {len(few)}, {statistics.mean(few):+.1f} | "
                     f"{len(many)}, {statistics.mean(many):+.1f} | {rho:+.2f} ({pval:.3f}) |")
    return lines + [""]


def per_proposal_quality() -> list[str]:
    """For the fixed-length ADMM arms (C, I, H): share of proposals that changed the prompt, and the
    validation net of changed / accepted children (vs. their parent, of 512)."""
    import glob

    arms_ = [("C", "artifacts/phase6_independent_acceptance/{t}_beam_indep_s*"),
             ("I", "artifacts/phase9/{t}_multistart_s*"), ("H", "artifacts/phase9/{t}_beam_p36_s*")]
    lines = ["## Per-proposal quality of fixed-length ADMM arms (validation net vs. parent, of 512)", "",
             "| Task | Arm | Proposals | Changed the prompt | Mean net, changed | Accepted | Mean net, accepted |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for task in ("agnews", "emotion"):
        for arm, pattern in arms_:
            made, changed, accepted = 0, [], []
            for run in sorted(d for d in glob.glob(pattern.format(t=task)) if Path(d).is_dir()):
                nets = net_by_pair(run)
                result = json.loads(Path(run, "gepa_result.json").read_text(encoding="utf-8"))
                for c in result["candidate_table"]:
                    made += 1
                    key = (c["parent_instruction"], c["instruction"])
                    if key in nets:
                        changed.append(nets[key])
                        if c["status"] == "accepted":
                            accepted.append(nets[key])
            lines.append(f"| {task} | {arm} | {made} | {100 * len(changed) / made:.0f}% | {statistics.mean(changed):+.1f} | "
                         f"{len(accepted)} | {statistics.mean(accepted):+.1f} |")
    return lines + [""]


def word_diff(parent: str, child: str) -> str:
    """Child text with inserted words in **bold** and deleted parent words as ~~strikethrough~~."""
    import difflib

    a, b = parent.split(), child.split()
    out: list[str] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op == "equal":
            out += a[i1:i2]
        if op in ("delete", "replace"):
            out.append("~~" + " ".join(a[i1:i2]) + "~~")
        if op in ("insert", "replace"):
            out.append("**" + " ".join(b[j1:j2]) + "**")
    return " ".join(out)


def multistart_diversity() -> list[str]:
    """Do the 3 starts of a multi-start proposal find different candidates? Candidates are de-duplicated
    across starts, so starts 1 and 2 only log candidates no earlier start found."""
    import glob

    lines = ["## Multi-start: what the extra starts contribute (Phase 9, all proposals)", "",
             "| Task | Proposals | New candidates per start (0 / 1 / 2) | Winner came from start 0 / 1 / 2 | "
             "Only an extra start beat the parent |", "|---|---:|---|---|---:|"]
    for task in ("agnews", "emotion"):
        counts, wins, only_extra, proposals = [0, 0, 0], [0, 0, 0], 0, 0
        for run in sorted(glob.glob(f"artifacts/phase9/{task}_multistart_s*/")):
            for line in Path(run, "admm_proposals.jsonl").read_text().splitlines():
                p = json.loads(line)
                proposals += 1
                rows = {0: p["iterations"], **{int(k): v for k, v in p.get("start_iterations", {}).items()}}
                scored = {k: [r["new_projection"] for r in v if (r.get("new_projection") or {}).get("selection_loss")
                              is not None] for k, v in rows.items()}
                for k, v in scored.items():
                    counts[k] += len(v)
                if p["outcome"] == "proposed":
                    wins[p["chosen"].get("start", 0)] += 1
                    if not any(c["selection_loss"] < p["parent_selection_loss"] for c in scored.get(0, [])):
                        only_extra += 1
        lines.append(f"| {task} | {proposals} | {' / '.join(f'{c / proposals:.1f}' for c in counts)} | "
                     f"{' / '.join(map(str, wins))} | {only_extra} |")
    return lines + [""]


def method_examples() -> list[str]:
    """Real accepted proposals per method (largest and median validation net) and each arm's final prompt
    for its median seed (AG News)."""
    import glob

    def proposals(pattern: str, want=lambda p: True):
        out = []
        for run in sorted(d for d in glob.glob(pattern) if Path(d).is_dir()):
            nets = net_by_pair(run)
            for line in Path(run, "admm_proposals.jsonl").read_text().splitlines():
                p = json.loads(line)
                key = (p["parent_instruction"], p["child_instruction"])
                if p["outcome"] == "proposed" and p.get("heldout_accept") and key in nets and want(p):
                    out.append((nets[key], p, run))
        return sorted(out, key=lambda x: x[0])

    sections = [
        ("C fixed-length beam ADMM", "artifacts/phase6_independent_acceptance/agnews_beam_indep_s*", lambda p: True),
        ("I multi-start, winner from an extra start", "artifacts/phase9/agnews_multistart_s*",
         lambda p: p["chosen"].get("start", 0) > 0),
        ("D multi-length, LM insertion (2-longer variant)", "artifacts/phase7/agnews_multilength_s*",
         lambda p: p["chosen"].get("start_offset", 0) > 0),
        ("D multi-length, LM insertion (2-shorter variant)", "artifacts/phase7/agnews_multilength_s*",
         lambda p: p["chosen"].get("start_offset", 0) < 0),
        ("F multi-length, gradient insertion (2-longer variant)", "artifacts/phase10/agnews_mlgrad_s*",
         lambda p: p["chosen"].get("start_offset", 0) > 0),
        ("G gated, length changed", "artifacts/phase8/agnews_gated_s*", lambda p: p["chosen"].get("start_offset", 0) != 0),
        ("G gated, same length", "artifacts/phase8/agnews_gated_s*", lambda p: p["chosen"].get("start_offset", 0) == 0),
    ]
    lines = ["## Example accepted proposals (AG News)", "",
             "Bold = inserted words, strikethrough = deleted words. Net = validation net of the child vs. its parent "
             "(correct-count change on the 512 validation examples; +5 ≈ +1 pp).", ""]
    for title, pattern, want in sections:
        rows = proposals(pattern, want)
        if not rows:
            continue
        lines += [f"### {title} ({len(rows)} accepted)", ""]
        for tag, (net, p, run) in (("largest net", rows[-1]), ("median net", rows[len(rows) // 2])):
            lines.append(f"- **{tag}, net {net:+d}** ({Path(run).name}): {word_diff(p['parent_instruction'], p['child_instruction'])}")
        lines.append("")
    lines += ["## Final prompts (AG News, each arm's median-scoring seed on the 2k set)", "",
              "| Arm | 2k accuracy | Final prompt |", "|---|---:|---|"]
    by_run = json.loads(Path("artifacts/report2k/agnews.json").read_text(encoding="utf-8"))["by_run"]
    for label, pattern, _ in arms("agnews"):
        runs = sorted((100 * v, k) for k, v in by_run.items() if re.search(pattern, k))
        if not runs:
            continue
        score, run = runs[len(runs) // 2]
        prompt = json.loads(Path(run, "gepa_result.json").read_text(encoding="utf-8"))["best_instruction"].strip()
        lines.append(f"| {DISPLAY.get(label, label)} | {score:.2f} | {prompt} |")
    return lines + [""]


def main() -> None:
    out = Path("artifacts/phase_7_10")
    out.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    lines = ["# Phases 7-10: all arms on the 2k report sets", "",
             "Mean ± sd over seeds; differences are Welch's t-tests against C (fair beam ADMM) and B "
             "(matched reflective). Many comparisons at n = 10: treat single p ≈ 0.03-0.05 with care.", ""]
    for task in ("agnews", "emotion"):
        big = json.loads(Path(f"artifacts/report2k/{task}.json").read_text(encoding="utf-8"))
        by_run = big["by_run"]
        rows = {}
        for label, pattern, family in arms(task):
            runs = sorted(k for k in by_run if re.search(pattern, k))
            if len(runs) < 2:
                continue
            calls = [admm_forward_calls(r) for r in runs]
            rows[label] = {"family": family, "runs": runs, "scores": [100 * by_run[r] for r in runs],
                           "admm_forward_calls": statistics.mean(c for c in calls if c) if any(calls) else None}
        data[task] = {"seed": 100 * big["seed_accuracy"], "arms": rows}
        C, B = rows["C  fair beam ADMM"]["scores"], rows["B  matched reflective (30)"]["scores"]
        lines += [f"## {'AG News' if task == 'agnews' else 'Emotion'} (seed instruction {data[task]['seed']:.2f})", "",
                  "| Arm | n | Mean ± sd | vs. C (p) | vs. B (p) | ADMM forward passes per run |",
                  "|---|---:|---:|---:|---:|---:|"]
        for label, row in rows.items():
            v = row["scores"]
            cells = []
            for ref in (C, B):
                if v is ref:
                    cells.append("—")
                else:
                    p = float(stats.ttest_ind(v, ref, equal_var=False).pvalue)
                    cells.append(f"{statistics.mean(v) - statistics.mean(ref):+.2f} ({p:.3f})")
            fc = f"{row['admm_forward_calls']:,.0f}" if row["admm_forward_calls"] else "—"
            lines.append(f"| {label} | {len(v)} | {statistics.mean(v):.2f} ± {statistics.stdev(v):.2f} | "
                         f"{cells[0]} | {cells[1]} | {fc} |")
        lines.append("")
        if task == "agnews":
            E, A2 = rows["E  fair reflect-then-polish"]["scores"], rows["A' reflective + held-out judge"]["scores"]
            p = float(stats.ttest_ind(E, A2, equal_var=False).pvalue)
            data[task]["polish_effect"] = {"diff": statistics.mean(E) - statistics.mean(A2), "p": p}
            lines += [f"Polish effect (E − A', same judge and proposal count): "
                      f"{statistics.mean(E) - statistics.mean(A2):+.2f} pp (p = {p:.3f}).", ""]

    # Figure 12: every run of every arm on the 2k sets.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, task in zip(axes, data):
        rows = data[task]["arms"]
        labels = list(rows)[::-1]
        for y, label in enumerate(labels):
            row = rows[label]
            color = FAMILY[row["family"]]
            ax.scatter(row["scores"], [y] * len(row["scores"]), s=22, color=color, alpha=0.55, zorder=3)
            mean, half = ci(row["scores"])
            ax.errorbar([mean], [y], xerr=[[half], [half]], fmt="|", color=TEXT, markersize=12, capsize=3, lw=1.4, zorder=4)
            ax.text(ax.get_xlim()[1] if False else mean, y + 0.32, f"{mean:.2f}" + ("" if len(row["scores"]) == 10
                    else f" (n={len(row['scores'])})"), ha="center", fontsize=7, color=TEXT_2)
        ax.axvline(data[task]["seed"], color=REFERENCE, lw=1, ls="--")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("accuracy on the 2k report set (%)")
        ax.set_title("AG News" if task == "agnews" else "Emotion")
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAMILY.items()]
    handles += [plt.Line2D([], [], marker="|", ls="", color=TEXT, markersize=11, label="mean ± 95% CI"),
                plt.Line2D([], [], ls="--", color=REFERENCE, label="seed instruction")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Every arm on the 2k report sets (one dot per seed)", fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(wspace=0.55, bottom=0.17, top=0.9)
    fig.savefig("artifacts/figures/fig12_all_arms_2k.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 13: accuracy vs. ADMM compute relative to C. Compute = model forward passes made inside the
    # ADMM proposer over one GEPA run (E-steps, projection steps, candidate scoring on S and S').
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, task in zip(axes, data):
        base = data[task]["arms"]["C  fair beam ADMM"]["admm_forward_calls"]
        means = {r.split("  ")[0]: statistics.mean(v["scores"]) for r, v in data[task]["arms"].items()}
        upper = "H" if means.get("H", 0) >= means.get("I", 0) else "I"
        for label, row in data[task]["arms"].items():
            if not row["admm_forward_calls"]:
                continue
            x = row["admm_forward_calls"] / base
            mean, half = ci(row["scores"])
            color = FAMILY[row["family"]]
            ax.errorbar([x], [mean], yerr=[[half], [half]], fmt="o", color=color, capsize=3, ms=6)
            letter = label.split("  ")[0]
            offset = {upper: (8, 8), ("I" if upper == "H" else "H"): (8, -12), "D": (-16, 4), "F": (7, 4),
                      "G": (8, 8)}.get(letter, (6, 4))
            ax.annotate(letter, (x, mean), textcoords="offset points", xytext=offset, fontsize=9, color=color,
                        fontweight="bold")
        ax.axhline(data[task]["seed"], color=REFERENCE, lw=1, ls="--")
        ax.axhline(statistics.mean(data[task]["arms"]["B  matched reflective (30)"]["scores"]),
                   color=COLOR["reflective"], lw=1, ls=":")
        ax.set_xlim(0.6, 3.8)
        ax.set_xlabel("ADMM compute per GEPA run, relative to C")
        ax.set_ylabel("2k report accuracy (%), mean ± 95% CI")
        ax.set_title("AG News" if task == "agnews" else "Emotion")
    bases = {t: data[t]["arms"]["C  fair beam ADMM"]["admm_forward_calls"] for t in data}
    fig.text(0.01, -0.04, "Compute = model forward passes inside the ADMM proposer over one 12-proposal run (E-steps, projection "
             f"steps, candidate scoring); 1× = C's {bases['agnews']:,.0f} (AG News) / {bases['emotion']:,.0f} (emotion).\n"
             "Dashed: seed instruction. Dotted: matched reflection (B). Bars: mean ± 95% CI over 10 seeds.",
             fontsize=7.5, color=TEXT_2)
    fig.suptitle("Accuracy vs. ADMM compute (letters as in the arms table)", fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(wspace=0.25, top=0.86)
    fig.savefig("artifacts/figures/fig13_compute_vs_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 14: proposal bench (same 6 parents per task, identical splits).
    methods = [("fixed", "C-type: fixed"), ("multistart", "I-type: multi-start"), ("ml_lm", "D-type: multi-length, LM"),
               ("ml_grad", "F-type: multi-length, gradient"), ("gated", "G-type: gated")]
    bench: dict = {}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax, task in zip(axes, ("agnews", "emotion")):
        rows = [json.loads(l) for l in Path(f"artifacts/length_bench/{task}.jsonl").read_text().splitlines()]
        for y, (method, label) in enumerate(methods[::-1]):
            sub = [r for r in rows if r["method"] == method]
            nets = [r["fixes"] - r["breaks"] for r in sub]
            nll = [r["nll_per_token"] - r["parent_nll_per_token"] for r in sub if r["nll_per_token"]]
            seconds = statistics.mean(r["wall_time_seconds"] for r in sub)
            bench.setdefault(task, {})[method] = {"nets": nets, "mean": statistics.mean(nets), "seconds": seconds,
                                                  "nll_change": statistics.mean(nll) if nll else 0.0}
            ax.scatter(nets, [y] * len(nets), s=22, color=TEXT_2, alpha=0.6, zorder=3)
            ax.scatter([statistics.mean(nets)], [y], marker="|", s=260, color=TEXT, zorder=4)
            ax.text(1.02, y, f"{statistics.mean(nets):+.1f} | {seconds:.0f}s | ΔNLL {bench[task][method]['nll_change']:+.2f}",
                    transform=ax.get_yaxis_transform(), fontsize=7.5, va="center", color=TEXT)
        ax.axvline(0, color=REFERENCE, lw=1)
        ax.set_yticks(range(len(methods)), [m[1] for m in methods[::-1]], fontsize=8.5)
        ax.set_xlabel("validation net of the child vs. its parent (fixes − breaks, of 512)")
        ax.set_title(("AG News" if task == "agnews" else "Emotion") + "   (right: mean | s/proposal | ΔNLL/token)",
                     fontsize=10)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Proposal bench: one child per method from the same 6 parents, identical splits",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(wspace=0.55, top=0.84)
    fig.savefig("artifacts/figures/fig14_proposal_bench.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    lines += ["## Proposal bench (6 parents per task)", "",
              "| Task | Method | Mean net | Per parent | s / proposal | ΔNLL per token |", "|---|---|---:|---|---:|---:|"]
    for task, by_method in bench.items():
        for method, b in by_method.items():
            lines.append(f"| {task} | {method} | {b['mean']:+.1f} | {b['nets']} | {b['seconds']:.0f} | {b['nll_change']:+.2f} |")
    diagnostic_lines, diagnostics = phase7_diagnostics()
    lines += [""] + diagnostic_lines + seed_planning(data) + acceptance_evidence() + candidate_pool_effect() + per_proposal_quality() + multistart_diversity()
    focus_figures(data)
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "examples.md").write_text("\n".join(method_examples()) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"arms": data, "bench": bench, "phase7": diagnostics}, indent=2),
                                      encoding="utf-8")
    print("\n".join(lines))
    print("WROTE artifacts/figures/fig12_all_arms_2k.png, fig13_compute_vs_accuracy.png, fig14_proposal_bench.png")


if __name__ == "__main__":
    main()
