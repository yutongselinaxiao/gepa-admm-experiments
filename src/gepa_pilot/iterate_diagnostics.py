"""Which ADMM iterate does the proposer return, and what would "take the last round" return?

For every beam/top-p ADMM proposal (GEPA + ADMM proposer arms), from admm_proposals.jsonl:
- the round (1-20) whose candidate was returned (lowest held-out loss on S, if it beat the parent),
- whether the final round's iterate is just the parent again (no token changed),
- how often the iterate returns to the parent within one proposal (oscillation),
- for proposals whose final-round iterate is identifiable, its loss on S vs. the returned one.

The final-round iterate is identifiable when it is the parent (no change) or a candidate first
seen in that round; otherwise it is a revisit of an earlier candidate (logged only once).

Writes artifacts/iterate_diagnostics/summary.json and figure fig7_admm_iterates.png.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .make_figures import COLOR, GRID, REFERENCE, SURFACE, TEXT, TEXT_2  # noqa: F401  (shared style)

BEAM_LIGHT = "#9085e9"   # lighter step of the beam (violet) hue
BEAM_PALE = "#d3cff5"    # palest step, for "no change"
EXAMPLE: dict = {}

ARMS = {
    "AG News": {
        "beam (S twice)": [f"artifacts/phase2/beam_lam0.3_s{i}" for i in range(10)],
        "beam (fair S')": [f"artifacts/phase6_independent_acceptance/agnews_beam_indep_s{i}" for i in range(10)],
        "top-p": [f"artifacts/phase2/admm_topp_s{i}" for i in range(10)],
    },
    "Emotion": {
        "beam (S twice)": [f"artifacts/phase4_emotion/beam_lam0.3_s{i}" for i in range(10)],
        "beam (fair S')": [f"artifacts/phase6_independent_acceptance/emotion_beam_indep_s{i}" for i in range(10)],
        "top-p": [f"artifacts/phase4_emotion/admm_topp_s{i}" for i in range(10)],
    },
}


def analyze(records: list[dict]) -> dict:
    proposals = len(records)
    final_parent = 0
    returns_to_parent: list[int] = []
    chosen_rounds: list[int] = []
    final_status = Counter()
    gaps: list[float] = []  # S-loss(final-round candidate) - S-loss(returned), when identifiable
    distinct: list[int] = []
    for r in records:
        rows = r.get("iterations", [])
        if not rows:
            continue
        changed = [row["changed_token_count_from_parent"] for row in rows]
        returns_to_parent.append(sum(1 for a, b in zip(changed, changed[1:]) if a > 0 and b == 0))
        distinct.append(sum(1 for row in rows if "new_projection" in row))
        last = rows[-1]
        if changed[-1] == 0:
            final_parent += 1
            final_status["parent (no change)"] += 1
        elif "new_projection" in last:
            final_status["new candidate in round 20"] += 1
        else:
            final_status["revisit of an earlier candidate"] += 1
        if r.get("outcome") == "proposed":
            chosen_rounds.append(r["chosen"]["iteration"])
            if "new_projection" in last and last["new_projection"].get("selection_loss") is not None:
                gaps.append(last["new_projection"]["selection_loss"] - r["chosen"]["selection_loss"])
            elif changed[-1] == 0:
                gaps.append(r["parent_selection_loss"] - r["chosen"]["selection_loss"])
    return {
        "proposals": proposals,
        "returned_a_child": len(chosen_rounds),
        "final_round_is_parent": final_parent,
        "final_round_status": dict(final_status),
        "returned_from_round_20": sum(1 for x in chosen_rounds if x == 20),
        "chosen_round_median": statistics.median(chosen_rounds) if chosen_rounds else None,
        "chosen_round_histogram": {str(k): v for k, v in sorted(Counter(chosen_rounds).items())},
        "returns_to_parent_per_proposal_mean": statistics.mean(returns_to_parent) if returns_to_parent else None,
        "distinct_candidates_per_proposal_median": statistics.median(distinct) if distinct else None,
        "final_minus_returned_selection_loss": {
            "n": len(gaps),
            "median": statistics.median(gaps) if gaps else None,
            "share_final_worse": sum(1 for g in gaps if g > 0) / len(gaps) if gaps else None,
        },
    }


def example_trajectory(records: list[dict]) -> dict | None:
    """A typical proposal: chosen mid-run, final round back at the parent, several distinct candidates."""
    best = None
    for r in records:
        rows = r.get("iterations", [])
        if r.get("outcome") != "proposed" or not rows or rows[-1]["changed_token_count_from_parent"] != 0:
            continue
        score = sum(1 for row in rows if "new_projection" in row)
        if 6 <= r["chosen"]["iteration"] <= 15 and (best is None or score > best[0]):
            best = (score, r)
    return best[1] if best else None


def main() -> None:
    out = Path("artifacts/iterate_diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    records_by: dict = {}
    for task, arms in ARMS.items():
        for arm, runs in arms.items():
            records = [json.loads(line) for run in runs
                       for line in Path(run, "admm_proposals.jsonl").read_text(encoding="utf-8").splitlines()]
            records_by[(task, arm)] = records
            summary[f"{task} / {arm}"] = analyze(records)
            s = summary[f"{task} / {arm}"]
            print(f"{task:8s} {arm:15s} proposals {s['proposals']:3d} | final=parent {s['final_round_is_parent']:3d} "
                  f"| returned from round 20: {s['returned_from_round_20']}/{s['returned_a_child']} "
                  f"| chosen median round {s['chosen_round_median']} | returns to parent/proposal "
                  f"{s['returns_to_parent_per_proposal_mean']:.2f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1, 1.25]})
    # (a) which round the returned candidate came from (fair beam arm, both tasks)
    ax = axes[0]
    # Everything here is the beam ADMM proposer (violet); tasks differ by shade.
    for shade, task, offset in ((COLOR["beam"], "AG News", -0.2), (BEAM_LIGHT, "Emotion", 0.2)):
        hist = summary[f"{task} / beam (fair S')"]["chosen_round_histogram"]
        xs = list(range(1, 21))
        ax.bar([x + offset for x in xs], [hist.get(str(x), 0) for x in xs], width=0.4, color=shade,
               label=task)
    ax.set_xlabel("round the returned candidate came from")
    ax.set_ylabel("proposals")
    ax.set_title("(a) Round of the returned candidate", fontsize=10)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(loc="upper left")
    # (b) status of the round-20 iterate
    ax = axes[1]
    statuses = ["parent (no change)", "revisit of an earlier candidate", "new candidate in round 20"]
    tasks = ["AG News", "Emotion"]
    left = [0.0, 0.0]
    shades = [BEAM_PALE, BEAM_LIGHT, COLOR["beam"]]
    for status, color in zip(statuses, shades):
        values = []
        for task in tasks:
            s = summary[f"{task} / beam (fair S')"]
            values.append(100 * s["final_round_status"].get(status, 0) / s["proposals"])
        ax.barh(tasks, values, left=left, color=color, height=0.55, label=status, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(values):
            if v >= 8:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if color == COLOR["beam"] else TEXT)
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of proposals (%)")
    ax.set_title("(b) The round-20 iterate is", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    ax.grid(axis="y", visible=False)
    # (c) one typical trajectory
    ax = axes[2]
    example = example_trajectory(records_by[("AG News", "beam (fair S')")])
    if example:
        rows = example["iterations"]
        rounds = [row["iteration"] for row in rows]
        changed = [row["changed_token_count_from_parent"] for row in rows]
        ax.plot(rounds, changed, color=COLOR["beam"], lw=1.5, drawstyle="steps-mid")
        new = [row for row in rows if "new_projection" in row
               and row["new_projection"].get("selection_loss") is not None]
        ax.scatter([row["iteration"] for row in new], [row["changed_token_count_from_parent"] for row in new],
                   s=34, color=COLOR["beam"], edgecolor=SURFACE, zorder=3, label="new candidate (scored on S)")
        c = example["chosen"]["iteration"]
        ax.scatter([c], [changed[c - 1]], s=150, marker="*", color=TEXT, zorder=4, label="returned (lowest S-loss)")
        ax.annotate(f"returned: S-loss {example['chosen']['selection_loss']:.2f}", (c, changed[c - 1]),
                    xytext=(-6, 10), textcoords="offset points", fontsize=7.5, ha="right")
        ax.annotate(f"round 20 = parent (S-loss {example['parent_selection_loss']:.2f})", (20, 0),
                    xytext=(0, -14), textcoords="offset points", fontsize=7.5, ha="right", va="top",
                    color=TEXT_2)
        ax.set_ylim(-1.1, max(changed) + 1.0)
        ax.set_yticks(range(0, max(changed) + 1))
        ax.legend(loc="upper left", fontsize=7.5)
        EXAMPLE.update(example)
    ax.set_xlabel("ADMM round")
    ax.set_ylabel("tokens changed from the parent")
    ax.set_title("(c) One proposal's trajectory", fontsize=10)
    ax.set_xticks([1, 5, 10, 15, 20])
    fig.subplots_adjust(wspace=0.38, bottom=0.27)
    path = Path("artifacts/figures/fig7_admm_iterates.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"WROTE {path}")
    if EXAMPLE:
        print("EXAMPLE parent:", EXAMPLE["parent_instruction"])
        for row in EXAMPLE["iterations"]:
            if "new_projection" in row:
                p = row["new_projection"]
                print(f"  round {row['iteration']:2d} S-loss {p['selection_loss']}: {p['text']}")


if __name__ == "__main__":
    main()
