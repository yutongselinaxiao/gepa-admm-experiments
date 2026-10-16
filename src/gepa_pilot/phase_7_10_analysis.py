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

    # Figure 13: accuracy vs. ADMM compute (forward passes per run) for the ADMM arms.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, task in zip(axes, data):
        for label, row in data[task]["arms"].items():
            if not row["admm_forward_calls"]:
                continue
            mean, half = ci(row["scores"])
            color = FAMILY[row["family"]]
            ax.errorbar([row["admm_forward_calls"]], [mean], yerr=[[half], [half]], fmt="o", color=color, capsize=3, ms=6)
            letter = label.split("  ")[0]
            means = {r.split("  ")[0]: statistics.mean(v["scores"]) for r, v in data[task]["arms"].items()}
            upper = "H" if means.get("H", 0) >= means.get("I", 0) else "I"  # H and I sit at the same compute
            offset = {upper: (8, 8), ("I" if upper == "H" else "H"): (8, -12),
                      "D": (-16, 4), "F": (7, 4)}.get(letter, (6, 4))
            ax.annotate(letter, (row["admm_forward_calls"], mean), textcoords="offset points",
                        xytext=offset, fontsize=9, color=color, fontweight="bold")
        ax.axhline(data[task]["seed"], color=REFERENCE, lw=1, ls="--")
        ref = data[task]["arms"]["B  matched reflective (30)"]["scores"]
        ax.axhline(statistics.mean(ref), color=COLOR["reflective"], lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("ADMM forward passes per run (log scale)")
        ax.set_ylabel("2k report accuracy (%), mean ± 95% CI")
        ax.set_title("AG News" if task == "agnews" else "Emotion")
    axes[0].text(0.01, 0.02, "dashed: seed instruction · dotted: matched reflective (B)", transform=axes[0].transAxes,
                 fontsize=7.5, color=TEXT_2)
    fig.suptitle("Accuracy vs. ADMM compute (letters as in the table)", fontsize=11, fontweight="bold", x=0.01, ha="left")
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
    lines += [""] + diagnostic_lines
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"arms": data, "bench": bench, "phase7": diagnostics}, indent=2),
                                      encoding="utf-8")
    print("\n".join(lines))
    print("WROTE artifacts/figures/fig12_all_arms_2k.png, fig13_compute_vs_accuracy.png, fig14_proposal_bench.png")


if __name__ == "__main__":
    main()
