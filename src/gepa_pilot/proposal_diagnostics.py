"""Two diagnostics over every GEPA proposal (parent -> child), from saved runs only.

1. Text distance vs. behavior distance, per proposer (reflection, Euclidean / top-p / beam ADMM,
   reflect-then-polish):
   - text: token-level edit distance normalized by the longer slot, word-level edit distance,
     change in token length;
   - behavior: share of the 512 validation examples whose predicted label differs between child
     and parent ("flip rate"), split into fixes (wrong -> right) and breaks (right -> wrong).
2. Acceptance reliability, per acceptance rule: how often an accepted / rejected child is
   actually better than its parent on the 512 validation examples. No acceptance rule ever sees
   validation data (all rules use training examples), so validation is an independent check.

Uses the post-hoc audit in each run's gepa_result.json (every parent and child was scored on all
512 validation examples). Writes artifacts/proposal_diagnostics/{summary.json,proposals.jsonl} and
figures fig9_text_vs_behavior.png, fig10_acceptance_reliability.png.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .make_figures import COLOR, GRID, REFERENCE, SURFACE, TEXT, TEXT_2  # noqa: F401  (shared style)

MODEL = ("/home/xiaoyuto/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/"
         "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")

# (task, proposer, acceptance rule, run directories)
GROUPS = [
    ("agnews", "reflection", "minibatch (GEPA default)", [f"artifacts/phase2/reflective_s{i}" for i in range(10)]),
    ("agnews", "reflection", "held-out judge", [f"artifacts/phase3/reflective_heldout_p30_s{i}" for i in range(10)]),
    ("agnews", "Euclidean ADMM", "minibatch (GEPA default)",
     [f"artifacts/phase1/admm_rho{r}_s{i}" for r in ("0.3", "1") for i in range(3)]),
    ("agnews", "Euclidean ADMM", "held-out on S (S used twice)",
     [f"artifacts/phase1_heldout/admm_rho{r}_s{i}" for r in ("0.3", "1") for i in range(3)]),
    ("agnews", "top-p ADMM", "held-out on S (S used twice)", [f"artifacts/phase2/admm_topp_s{i}" for i in range(10)]),
    ("agnews", "beam ADMM", "held-out on S (S used twice)", [f"artifacts/phase2/beam_lam0.3_s{i}" for i in range(10)]),
    ("agnews", "beam ADMM", "held-out on independent S'",
     [f"artifacts/phase6_independent_acceptance/agnews_beam_indep_s{i}" for i in range(10)]),
    ("agnews", "reflect-then-polish", "minibatch (GEPA default)", [f"artifacts/phase2/polish_s{i}" for i in range(3)]),
    ("emotion", "reflection", "minibatch (GEPA default)", [f"artifacts/phase4_emotion/reflective_p12_s{i}" for i in range(10)]),
    ("emotion", "reflection", "held-out judge", [f"artifacts/phase4_emotion/reflective_heldout_p30_s{i}" for i in range(10)]),
    ("emotion", "top-p ADMM", "held-out on S (S used twice)", [f"artifacts/phase4_emotion/admm_topp_s{i}" for i in range(10)]),
    ("emotion", "beam ADMM", "held-out on S (S used twice)", [f"artifacts/phase4_emotion/beam_lam0.3_s{i}" for i in range(10)]),
    ("emotion", "beam ADMM", "held-out on independent S'",
     [f"artifacts/phase6_independent_acceptance/emotion_beam_indep_s{i}" for i in range(10)]),
]


def edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, y in enumerate(b, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y))
        previous = current
    return previous[-1]


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows: list[dict] = []
    for task, proposer, rule, runs in GROUPS:
        for run in runs:
            result = json.loads(Path(run, "gepa_result.json").read_text(encoding="utf-8"))
            audit = result["audit_by_instruction"]
            for proposal in result["candidate_table"]:
                parent, child = proposal["parent_instruction"], proposal["instruction"]
                if not parent or not child or parent == child or parent not in audit or child not in audit:
                    continue
                pa, ch = audit[parent], audit[child]
                ids = list(pa["predicted_by_id"])
                flips = sum(pa["predicted_by_id"][i] != ch["predicted_by_id"][i] for i in ids)
                fixes = sum(ch["correct_by_id"][i] and not pa["correct_by_id"][i] for i in ids)
                breaks = sum(pa["correct_by_id"][i] and not ch["correct_by_id"][i] for i in ids)
                tp, tc = tokenizer.encode(parent), tokenizer.encode(child)
                wp, wc = parent.split(), child.split()
                rows.append({
                    "task": task, "proposer": proposer, "rule": rule, "run": run,
                    "status": proposal["status"],
                    "token_edit_fraction": edit_distance(tp, tc) / max(len(tp), len(tc)),
                    "word_edit_fraction": edit_distance(wp, wc) / max(len(wp), len(wc)),
                    "token_length_change": len(tc) - len(tp),
                    "flip_rate": flips / len(ids),
                    "fixes": fixes, "breaks": breaks,
                    "accuracy_change_pp": 100 * (ch["accuracy"] - pa["accuracy"]),
                    "parent": parent, "child": child,
                })
    out = Path("artifacts/proposal_diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    (out / "proposals.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                         encoding="utf-8")

    def med(values: list[float]) -> float:
        return statistics.median(values) if values else float("nan")

    summary: dict = {"behavior": {}, "acceptance": {}}
    for task in ("agnews", "emotion"):
        for proposer in dict.fromkeys(r["proposer"] for r in rows if r["task"] == task):
            sub = [r for r in rows if r["task"] == task and r["proposer"] == proposer]
            summary["behavior"][f"{task} / {proposer}"] = {
                "proposals": len(sub),
                "token_edit_fraction_median": med([r["token_edit_fraction"] for r in sub]),
                "word_edit_fraction_median": med([r["word_edit_fraction"] for r in sub]),
                "abs_length_change_median": med([abs(r["token_length_change"]) for r in sub]),
                "flip_rate_median": med([r["flip_rate"] for r in sub]),
                "flips_per_edited_token_median": med([r["flip_rate"] / max(r["token_edit_fraction"], 1e-9) for r in sub]),
                "fix_share_of_flips": (sum(r["fixes"] for r in sub) / max(1, sum(r["fixes"] + r["breaks"] for r in sub))),
                "accuracy_change_median_pp": med([r["accuracy_change_pp"] for r in sub]),
            }
        for _, proposer, rule, _ in [g for g in GROUPS if g[0] == task]:
            sub = [r for r in rows if r["task"] == task and r["proposer"] == proposer and r["rule"] == rule]
            acc = [r for r in sub if r["status"] == "accepted"]
            rej = [r for r in sub if r["status"] == "rejected"]
            better = lambda rs: sum(r["accuracy_change_pp"] > 0 for r in rs) / len(rs) if rs else float("nan")  # noqa: E731
            summary["acceptance"][f"{task} / {proposer} / {rule}"] = {
                "accepted": len(acc), "rejected": len(rej),
                "share_accepted_better": better(acc), "share_rejected_better": better(rej),
                "mean_change_accepted_pp": statistics.mean(r["accuracy_change_pp"] for r in acc) if acc else None,
                "mean_change_rejected_pp": statistics.mean(r["accuracy_change_pp"] for r in rej) if rej else None,
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for k, v in summary["behavior"].items():
        print(f"BEHAVIOR {k:32s} n={v['proposals']:3d} text-edit {v['token_edit_fraction_median']:.2f} "
              f"|len| {v['abs_length_change_median']:.0f} flip {100 * v['flip_rate_median']:.1f}% "
              f"fix-share {100 * v['fix_share_of_flips']:.0f}% dAcc {v['accuracy_change_median_pp']:+.2f}")
    for k, v in summary["acceptance"].items():
        print(f"ACCEPT {k:60s} acc {v['accepted']:3d} ({100 * v['share_accepted_better']:.0f}% better) "
              f"rej {v['rejected']:3d} ({100 * v['share_rejected_better']:.0f}% better)")
    figures(rows, summary)


def figures(rows: list[dict], summary: dict) -> None:
    # Figure 9: text vs behavior, small multiples (one proposer per panel, AG News).
    panels = [("reflection", "reflective", "o", "reflection (GEPA default)"),
              ("Euclidean ADMM", "euclidean", "s", "Euclidean ADMM proposer"),
              ("top-p ADMM", "topp", "^", "top-p ADMM proposer"),
              ("beam ADMM", "beam", "D", "beam ADMM proposer")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), sharex=True, sharey=True)
    for ax, (proposer, key, marker, title) in zip(axes, panels):
        sub = [r for r in rows if r["task"] == "agnews" and r["proposer"] == proposer]
        ax.scatter([100 * r["token_edit_fraction"] for r in sub], [100 * r["flip_rate"] for r in sub],
                   s=16, marker=marker, color=COLOR[key], alpha=0.55, edgecolor="none")
        s = summary["behavior"][f"agnews / {proposer}"]
        ax.scatter([100 * s["token_edit_fraction_median"]], [100 * s["flip_rate_median"]], s=110, marker="*",
                   color=TEXT, zorder=4)
        ax.set_title(f"{title}\n(n = {len(sub)}; ★ = median)", fontsize=9.5)
        ax.set_xlabel("text changed (% of tokens edited)")
    axes[0].set_ylabel("predictions changed (% of 512 examples)")
    fig.suptitle("AG News: how much each proposer changes the text vs. the model's predictions", fontsize=11,
                 fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(top=0.8, wspace=0.08)
    fig.savefig("artifacts/figures/fig9_text_vs_behavior.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("WROTE artifacts/figures/fig9_text_vs_behavior.png")

    # Figure 10: acceptance reliability.
    items = [(k, v) for k, v in summary["acceptance"].items() if v["accepted"] and v["rejected"]]
    labels = [k.replace("agnews / ", "AG News: ").replace("emotion / ", "Emotion: ") for k, _ in items]
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(items) + 1.4))
    ys = list(range(len(items)))[::-1]
    acc = [100 * v["share_accepted_better"] for _, v in items]
    rej = [100 * v["share_rejected_better"] for _, v in items]
    ax.barh([y + 0.18 for y in ys], acc, height=0.34, color=TEXT_2, label="accepted children")
    ax.barh([y - 0.18 for y in ys], rej, height=0.34, color="#c9c8c3", label="rejected children")
    for y, a, r, (_, v) in zip(ys, acc, rej, items):
        ax.text(a + 1, y + 0.18, f"{a:.0f}% (n={v['accepted']})", va="center", fontsize=7.5)
        ax.text(r + 1, y - 0.18, f"{r:.0f}% (n={v['rejected']})", va="center", fontsize=7.5, color=TEXT_2)
    ax.axvline(50, color=REFERENCE, lw=1, ls=":")
    ax.set_yticks(ys, labels, fontsize=8)
    ax.set_xlim(0, 112)
    ax.set_xlabel("children actually better than their parent on 512 validation examples (%)")
    ax.set_title("How well each acceptance rule separates better from worse children", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.savefig("artifacts/figures/fig10_acceptance_reliability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("WROTE artifacts/figures/fig10_acceptance_reliability.png")


if __name__ == "__main__":
    main()
