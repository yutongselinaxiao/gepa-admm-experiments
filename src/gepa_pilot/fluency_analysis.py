"""Fluency of prompts and ADMM edits (what the perplexity constraint buys).

For one task:
1. Every selected prompt of every arm (from artifacts/classification_results/summary.json),
   plus the seed instruction: total slot NLL under the frozen model and per-token perplexity.
2. Every ADMM proposal that produced a child, per projection (Euclidean = no fluency
   constraint, top-p = hard per-token nucleus constraint, beam = soft NLL penalty, polish):
   - NLL of the most surprising edited token in context ("edit NLL", as in E5),
   - change in total slot NLL, child minus parent,
   - embedding-norm percentile of each new token,
   - change in validation accuracy (full-validation audit), when both texts were audited.

NLL is measured on the slot given the fixed prompt prefix (system prompt and the start of the
user turn). The label mapping comes after the slot, so task words such as "capital" are
surprising here; compare NLLs, do not read them as absolute fluency.

Writes artifacts/fluency_analysis/<task>.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from .admm import build_allowed_token_ids
from .config import PilotConfig
from .data import load_examples_json
from .evaluator import CommonEvaluator

PROPOSAL_ARMS = {
    "agnews": {
        "euclidean (no constraint)": [f"artifacts/phase1/admm_rho{r}_s{i}" for r in ("0.3", "1") for i in range(3)]
        + [f"artifacts/phase1_heldout/admm_rho{r}_s{i}" for r in ("0.3", "1") for i in range(3)],
        "top-p (hard constraint)": [f"artifacts/phase2/admm_topp_s{i}" for i in range(10)],
        "beam λ=0.3 (soft penalty)": [f"artifacts/phase2/beam_lam0.3_s{i}" for i in range(10)]
        + [f"artifacts/phase6_independent_acceptance/agnews_beam_indep_s{i}" for i in range(10)],
        "beam λ=1.0": [f"artifacts/phase2/beam_lam1.0_s{i}" for i in range(3)],
        "reflect-then-polish (top-p)": [f"artifacts/phase2/polish_s{i}" for i in range(3)],
    },
    "emotion": {
        "top-p (hard constraint)": [f"artifacts/phase4_emotion/admm_topp_s{i}" for i in range(10)],
        "beam λ=0.3 (soft penalty)": [f"artifacts/phase4_emotion/beam_lam0.3_s{i}" for i in range(10)]
        + [f"artifacts/phase6_independent_acceptance/emotion_beam_indep_s{i}" for i in range(10)],
    },
}

TASKS = {
    "agnews": ("configs/pilot.yaml", "artifacts/data/train.json"),
    "emotion": ("configs/emotion.yaml", "artifacts/data_emotion/train.json"),
}


class SlotScorer:
    def __init__(self, evaluator: CommonEvaluator, reference_article: str) -> None:
        self.evaluator = evaluator
        self.reference_article = reference_article
        weight = evaluator.model.get_input_embeddings().weight.detach()
        allowed = build_allowed_token_ids(evaluator.tokenizer, weight.shape[0]).to(weight.device)
        self.weight = weight
        self.sorted_norms = weight.index_select(0, allowed).float().norm(dim=1).sort().values

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        encoding = self.evaluator.prompt_builder.encode(self.reference_article, text)
        return list(encoding.prefix_ids), list(encoding.slot_ids)

    @torch.inference_mode()
    def token_nll(self, prefix: list[int], slot: list[int]) -> list[float]:
        ids = torch.tensor([prefix + slot], device=self.evaluator.device)
        log_probs = self.evaluator.model(input_ids=ids).logits[0].float().log_softmax(dim=-1)
        return [float(-log_probs[len(prefix) + i - 1, token]) for i, token in enumerate(slot)]

    def norm_percentile(self, token_id: int) -> float:
        value = self.weight[token_id].float().norm()
        return float((self.sorted_norms < value).float().mean() * 100)

    def prompt_stats(self, text: str) -> dict[str, Any]:
        prefix, slot = self.encode(text)
        nll = self.token_nll(prefix, slot)
        return {"slot_tokens": len(slot), "slot_nll": sum(nll),
                "per_token_perplexity": math.exp(sum(nll) / len(slot))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    args = parser.parse_args()
    config_path, train_path = TASKS[args.task]
    config = PilotConfig.from_yaml(config_path)
    evaluator = CommonEvaluator.from_pretrained(
        name_or_path=config.model.resolved_name_or_path,
        revision=None if config.model.local_snapshot_path else config.model.revision,
        local_files_only=config.model.local_files_only,
        dtype=config.model.dtype,
        device=config.model.device,
        system_prompt=config.prompt.system_prompt,
        label_names=config.prompt.label_names,
        enable_thinking=config.prompt.enable_thinking,
        input_heading=config.task.input_heading,
    )
    scorer = SlotScorer(evaluator, load_examples_json(train_path)[0].article)
    decode = lambda token: evaluator.tokenizer.decode([token])

    # 1. Selected prompts.
    summary = json.loads(Path("artifacts/classification_results/summary.json").read_text(encoding="utf-8"))
    output: dict[str, Any] = {"task": args.task, "seed": {"text": config.prompt.seed_instruction,
                                                          **scorer.prompt_stats(config.prompt.seed_instruction)},
                              "arms": {}, "proposals": {}}
    for name, arm in summary["arms"].items():
        if arm["task"] != args.task:
            continue
        rows = []
        for run in arm["runs"]:
            rows.append({"run": run["run"], "report_accuracy": run["report_accuracy"],
                         "text": run["selected_instruction"], **scorer.prompt_stats(run["selected_instruction"])})
        output["arms"][name] = rows
        print(f"ARM {name}: median perplexity {sorted(r['per_token_perplexity'] for r in rows)[len(rows)//2]:.1f}")

    # 2. ADMM edits.
    for projection, runs in PROPOSAL_ARMS[args.task].items():
        edits = []
        for run in runs:
            log = Path(run, "admm_proposals.jsonl")
            if not log.exists():
                continue
            audit = json.loads(Path(run, "gepa_result.json").read_text(encoding="utf-8"))["audit_by_instruction"]
            for record in map(json.loads, log.read_text(encoding="utf-8").splitlines()):
                if record.get("outcome") != "proposed":
                    continue
                parent_ids, child_ids = record["parent_slot_ids"], record["chosen"]["slot_ids"]
                positions = [i for i, (a, b) in enumerate(zip(parent_ids, child_ids)) if a != b]
                prefix, _ = scorer.encode(record["parent_instruction"])
                parent_nll = scorer.token_nll(prefix, parent_ids)
                child_nll = scorer.token_nll(prefix, child_ids)
                child_text = record["child_instruction"]
                parent_text = record["parent_instruction"]
                gain = None
                if child_text in audit and parent_text in audit:
                    gain = 100 * (audit[child_text]["accuracy"] - audit[parent_text]["accuracy"])
                edits.append({
                    "run": run,
                    "edit_nll": max(child_nll[i] for i in positions),
                    "parent_token_nll": max(parent_nll[i] for i in positions),
                    "slot_nll_change": sum(child_nll) - sum(parent_nll),
                    "new_token_norm_percentiles": [scorer.norm_percentile(child_ids[i]) for i in positions],
                    "changes": [[decode(parent_ids[i]), decode(child_ids[i])] for i in positions],
                    "validation_gain_pp": gain,
                    "parent_text": parent_text,
                    "child_text": child_text,
                })
        output["proposals"][projection] = edits
        if edits:
            median = sorted(e["edit_nll"] for e in edits)[len(edits) // 2]
            print(f"EDITS {projection}: n={len(edits)} median edit NLL {median:.1f}")

    path = Path(f"artifacts/fluency_analysis/{args.task}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
