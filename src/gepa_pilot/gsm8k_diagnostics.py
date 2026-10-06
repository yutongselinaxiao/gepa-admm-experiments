"""Surrogate and gradient diagnostics for a generative task (GSM8K) before building generative ADMM.

Stage "rank": for a pool of instructions, measure greedy exact-match accuracy on a fixed set
of GSM8K *training* problems and score each instruction under candidate surrogates. Then
report how well each surrogate orders the instructions by accuracy.

Stage "grad": for a few instructions, compute gradients of each objective with respect to the
instruction-slot embeddings on disjoint problem batches. Report cross-batch consistency (mean
pairwise cosine) and alignment with a high-sample policy-gradient estimate. These decide which
objective the ADMM E-step could use.

The official GSM8K test split is never loaded.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .config import PilotConfig
from .prompting import PromptBuilder

SYSTEM_PROMPT = "You are a careful math problem solver."
RESPONSE_RULE = (
    "Show your reasoning, then give the final answer on the last line in the form "
    "'#### <number>'."
)
SEED_INSTRUCTION = (
    "Carefully solve the problem step by step, check each calculation, and state the final "
    "numeric answer."
)

# Deliberately spans helpful, neutral and harmful instructions so accuracy varies.
INSTRUCTION_POOL = [
    SEED_INSTRUCTION,
    "Read the problem twice, list the known quantities, solve step by step, and verify the answer before writing it.",
    "Break the problem into small steps, compute each intermediate value explicitly, and double-check the arithmetic.",
    "Think about what the question is really asking, set up the calculation, and solve it carefully.",
    "Write an equation for the unknown quantity, solve it, and check that the answer makes sense.",
    "Solve the problem. Keep the reasoning short and focused on the necessary calculations.",
    "Explain your reasoning clearly as if teaching a student, then give the answer.",
    "Use careful arithmetic and track units at every step.",
    "Work backwards from what is asked to the given information, then compute forward to the answer.",
    "First estimate the answer roughly, then compute it exactly and compare with the estimate.",
    "Solve the problem.",
    "Answer the question.",
    "Be concise.",
    "Respond in a friendly and encouraging tone.",
    "Make sure your answer is formatted correctly.",
    "Do not overthink; go with your first instinct.",
    "Answer immediately with only the final number and no reasoning.",
    "Give only a rough estimate of the answer, rounded to the nearest ten.",
    "Ignore the numbers in the problem and guess a plausible answer.",
    "Always add all the numbers in the problem together to get the answer.",
    "Reply with the answer 0 unless you are completely certain.",
    "Write a short poem about the problem before answering.",
    "Solve the problem using only mental math and skip writing intermediate steps.",
    "Repeat the problem in your own words, then solve it step by step.",
    "Consider two different solution approaches and give the answer they agree on.",
    "Focus on the last sentence of the problem, which contains the actual question.",
    "Convert all quantities to the same units before calculating.",
    "Multiply every number in the problem by two before solving.",
    "Solve step by step, but report the answer in words instead of digits.",
    "Carefully solve the problem step by step, check each calculation, and state the final numeric answer in the required format.",
]

ANSWER_PATTERN = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
NUMBER_PATTERN = re.compile(r"-?[\d,]*\.?\d+")
CALCULATOR_PATTERN = re.compile(r"<<[^>]*>>")


@dataclass(frozen=True)
class MathProblem:
    problem_id: str
    question: str
    reference_reasoning: str
    gold_answer: str


def load_problems(seed: int, eval_size: int, gradient_size: int) -> tuple[list[MathProblem], list[MathProblem]]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    problems = []
    for index in indices[: eval_size + gradient_size]:
        row = dataset[index]
        reasoning, _, answer = row["answer"].rpartition("####")
        problems.append(
            MathProblem(
                problem_id=f"gsm8k-train:{index}",
                question=row["question"],
                reference_reasoning=CALCULATOR_PATTERN.sub("", reasoning).strip(),
                gold_answer=answer.strip().replace(",", ""),
            )
        )
    return problems[:eval_size], problems[eval_size:]


def extract_answer(text: str) -> str | None:
    matches = ANSWER_PATTERN.findall(text)
    if not matches:
        matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def is_correct(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    try:
        return math.isclose(float(predicted), float(gold), rel_tol=0.0, abs_tol=1e-6)
    except ValueError:
        return False


def answer_line(answer: str) -> str:
    return f"#### {answer}"


def reasoning_before_answer(text: str) -> str:
    """Model output up to (not including) its '####' line; the whole output if absent."""
    head, separator, _ = text.partition("####")
    return head.rstrip() if separator else text.rstrip()


class GenerativePromptBuilder(PromptBuilder):
    """Same editable-slot mechanics as the classifier prompt, with a math-problem body."""

    def __init__(self, tokenizer: Any, *, enable_thinking: bool = False) -> None:
        super().__init__(
            tokenizer,
            SYSTEM_PROMPT,
            {"A": "unused", "B": "unused"},
            enable_thinking=enable_thinking,
            input_heading="Problem",
        )

    def _messages(self, article: str, instruction: str) -> list[dict[str, str]]:
        user_content = (
            "### Editable instruction\n"
            f"{instruction}"
            "### End editable instruction\n\n"
            f"### {self.input_heading}\n"
            f"{article}\n\n"
            f"{RESPONSE_RULE}"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]


class GenerativeScorer:
    def __init__(self, model: Any, tokenizer: Any, builder: GenerativePromptBuilder, device: str) -> None:
        self.model, self.tokenizer, self.builder, self.device = model, tokenizer, builder, device
        self.embedding = model.get_input_embeddings()

    # ----- generation (discrete prompts) -----
    @torch.inference_mode()
    def generate(
        self,
        problems: Sequence[MathProblem],
        instruction: str,
        *,
        batch_size: int,
        max_new_tokens: int,
        samples: int = 1,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> list[list[str]]:
        """Return `samples` completions per problem (greedy when temperature == 0)."""
        outputs: list[list[str]] = []
        pad_id = int(self.tokenizer.pad_token_id)
        for start in range(0, len(problems), batch_size):
            batch = problems[start : start + batch_size]
            prompts = [self.builder.encode(p.question, instruction).input_ids for p in batch]
            width = max(map(len, prompts))
            input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
            attention = torch.zeros_like(input_ids)
            for row, ids in enumerate(prompts):  # left padding for generation
                input_ids[row, width - len(ids) :] = torch.tensor(ids)
                attention[row, width - len(ids) :] = 1
            kwargs: dict[str, Any] = {"do_sample": False}
            if temperature > 0:
                torch.manual_seed(seed + start)
                kwargs = {"do_sample": True, "temperature": temperature, "top_p": 0.95}
            generated = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention.to(self.device),
                max_new_tokens=max_new_tokens,
                num_return_sequences=samples,
                pad_token_id=pad_id,
                **kwargs,
            )[:, width:]
            texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(texts[row * samples : (row + 1) * samples] for row in range(len(batch)))
        return outputs

    # ----- teacher-forced continuation log-likelihood -----
    def continuation_logprobs(
        self,
        problems: Sequence[MathProblem],
        instruction: str,
        continuations: Sequence[tuple[str, str]],
        slot_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probabilities of (context, scored) continuation pairs after each prompt.

        `context` is teacher-forced but not scored; `scored` is scored. Returns per-example
        (sum log-prob of scored tokens, scored token count, sum log-prob of all continuation
        tokens, all continuation token count). With slot_embeddings, the instruction slot is
        replaced by those (differentiable) embeddings; otherwise the token embeddings are used.
        """
        rows, scored_masks, all_masks = [], [], []
        for problem, (context, scored) in zip(problems, continuations, strict=True):
            encoding = self.builder.encode(problem.question, instruction)
            if not encoding.compositional:
                raise ValueError("Prompt is not compositional; cannot splice slot embeddings.")
            context_ids = self.tokenizer.encode(context, add_special_tokens=False) if context else []
            scored_ids = self.tokenizer.encode(scored, add_special_tokens=False)
            prompt_ids = list(encoding.input_ids)
            ids = prompt_ids + context_ids + scored_ids
            rows.append((encoding, ids))
            start = len(prompt_ids)
            all_mask = [False] * start + [True] * (len(context_ids) + len(scored_ids))
            scored_mask = [False] * (start + len(context_ids)) + [True] * len(scored_ids)
            all_masks.append(all_mask)
            scored_masks.append(scored_mask)

        width = max(len(ids) for _, ids in rows)
        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=self.device)
        attention = torch.zeros_like(input_ids)
        scored_mask = torch.zeros(len(rows), width, dtype=torch.bool, device=self.device)
        all_mask = torch.zeros_like(scored_mask)
        for row, (_, ids) in enumerate(rows):  # right padding for teacher forcing
            input_ids[row, : len(ids)] = torch.tensor(ids, device=self.device)
            attention[row, : len(ids)] = 1
            scored_mask[row, : len(ids)] = torch.tensor(scored_masks[row], device=self.device)
            all_mask[row, : len(ids)] = torch.tensor(all_masks[row], device=self.device)

        embeds = self.embedding(input_ids)
        if slot_embeddings is not None:
            embeds = embeds.clone()
            for row, (encoding, _) in enumerate(rows):
                embeds[row, encoding.slot_start : encoding.slot_end] = slot_embeddings.to(embeds.dtype)
        logits = self.model(inputs_embeds=embeds, attention_mask=attention, use_cache=False).logits
        log_probs = logits[:, :-1].float().log_softmax(dim=-1)
        token_log_probs = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        scored = scored_mask[:, 1:].float()
        everything = all_mask[:, 1:].float()
        return (
            (token_log_probs * scored).sum(dim=1),
            scored.sum(dim=1),
            (token_log_probs * everything).sum(dim=1),
            everything.sum(dim=1),
        )


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                result[order[k]] = (i + j) / 2
            i = j + 1
        return result

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return cov / var if var else float("nan")


def pairwise_disagreement(surrogate: Sequence[float], accuracy: Sequence[float]) -> tuple[int, int]:
    """Pairs with distinct accuracy where the surrogate (higher = better) orders them the other way."""
    disagree = total = 0
    for i, j in itertools.combinations(range(len(accuracy)), 2):
        if accuracy[i] == accuracy[j]:
            continue
        total += 1
        if (surrogate[i] - surrogate[j]) * (accuracy[i] - accuracy[j]) < 0:
            disagree += 1
    return disagree, total


def run_rank_stage(scorer: GenerativeScorer, problems: list[MathProblem], args: argparse.Namespace,
                   instructions: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for index, instruction in enumerate(instructions):
        started = time.perf_counter()
        greedy = [outs[0] for outs in scorer.generate(
            problems, instruction, batch_size=args.generation_batch_size,
            max_new_tokens=args.max_new_tokens)]
        sampled = scorer.generate(
            problems, instruction, batch_size=args.generation_batch_size,
            max_new_tokens=args.max_new_tokens, samples=args.samples,
            temperature=args.temperature, seed=args.seed + index)
        predictions = [extract_answer(text) for text in greedy]
        correct = [is_correct(pred, p.gold_answer) for pred, p in zip(predictions, problems)]
        sample_correct = [[is_correct(extract_answer(t), p.gold_answer) for t in outs]
                          for outs, p in zip(sampled, problems)]
        agreement = []
        for outs in sampled:
            answers = [extract_answer(t) for t in outs]
            counts = {a: answers.count(a) for a in set(answers) if a is not None}
            agreement.append(max(counts.values()) / len(answers) if counts else 0.0)

        sums: dict[str, float] = {"ref_full": 0.0, "ref_answer": 0.0, "own_answer": 0.0, "own_confidence": 0.0}
        counts_: dict[str, float] = {key: 0.0 for key in sums}
        with torch.inference_mode():
            for start in range(0, len(problems), args.score_batch_size):
                batch = problems[start : start + args.score_batch_size]
                outs = greedy[start : start + args.score_batch_size]
                pairs = {
                    "ref_answer": [(p.reference_reasoning + "\n", answer_line(p.gold_answer)) for p in batch],
                    "own_answer": [(reasoning_before_answer(o) + "\n", answer_line(p.gold_answer)) for o, p in zip(outs, batch)],
                    "own_confidence": [(reasoning_before_answer(o) + "\n", answer_line(pred or "0"))
                                       for o, pred in zip(outs, predictions[start : start + args.score_batch_size])],
                }
                for key, cont in pairs.items():
                    lp, n, all_lp, all_n = scorer.continuation_logprobs(batch, instruction, cont)
                    sums[key] += float((-lp / n.clamp_min(1)).sum())
                    counts_[key] += len(batch)
                    if key == "ref_answer":  # the same pass gives the full reference NLL
                        sums["ref_full"] += float((-all_lp / all_n.clamp_min(1)).sum())
                        counts_["ref_full"] += len(batch)
        row = {
            "instruction": instruction,
            "greedy_accuracy": sum(correct) / len(correct),
            "sampled_accuracy": sum(map(sum, sample_correct)) / (len(problems) * args.samples),
            "self_consistency": sum(agreement) / len(agreement),
            "format_rate": sum("####" in t for t in greedy) / len(greedy),
            "mean_output_chars": sum(map(len, greedy)) / len(greedy),
            **{f"nll_{key}": sums[key] / counts_[key] for key in sums},
            "wall_time_seconds": time.perf_counter() - started,
            "greedy_correct_by_id": {p.problem_id: c for p, c in zip(problems, correct)},
        }
        rows.append(row)
        print(f"RANK {index + 1}/{len(instructions)} acc={row['greedy_accuracy']:.3f} "
              f"ref_full={row['nll_ref_full']:.3f} own_answer={row['nll_own_answer']:.3f} "
              f"t={row['wall_time_seconds']:.0f}s :: {instruction[:60]}")
    return rows


def summarize_rank(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy = [r["greedy_accuracy"] for r in rows]
    # Every surrogate oriented so that higher = predicted better.
    surrogates = {
        "neg_nll_ref_full": [-r["nll_ref_full"] for r in rows],
        "neg_nll_ref_answer": [-r["nll_ref_answer"] for r in rows],
        "neg_nll_own_answer": [-r["nll_own_answer"] for r in rows],
        "neg_nll_own_confidence (label-free)": [-r["nll_own_confidence"] for r in rows],
        "self_consistency (label-free)": [r["self_consistency"] for r in rows],
        "sampled_accuracy (RL objective)": [r["sampled_accuracy"] for r in rows],
    }
    summary = {}
    for name, values in surrogates.items():
        disagree, total = pairwise_disagreement(values, accuracy)
        summary[name] = {"spearman": spearman(values, accuracy), "pairwise_disagreements": disagree, "pairs": total}
    return summary


def run_grad_stage(scorer: GenerativeScorer, problems: list[MathProblem], args: argparse.Namespace,
                   instructions: Sequence[str]) -> list[dict[str, Any]]:
    """Gradients w.r.t. slot embeddings for each objective on disjoint batches."""
    results = []
    batches = [problems[i : i + args.grad_batch_size]
               for i in range(0, args.grad_batches * args.grad_batch_size, args.grad_batch_size)]
    for instruction in instructions:
        encoding = scorer.builder.encode(problems[0].question, instruction)
        slot_ids = torch.tensor(encoding.slot_ids, device=scorer.device)
        base = scorer.embedding(slot_ids).detach().float()

        def chunks(items: Sequence[Any]) -> list[range]:
            return [range(i, min(i + args.score_batch_size, len(items)))
                    for i in range(0, len(items), args.score_batch_size)]

        def gradient(chunk_losses) -> torch.Tensor:
            """Accumulate d(objective)/d(slot) chunk by chunk to bound activation memory."""
            e = base.clone().requires_grad_(True)
            for loss in chunk_losses(e):
                loss.backward()
            if e.grad is None:  # every weight was zero (e.g. RLOO when all rewards tie)
                return torch.zeros_like(base).flatten()
            return e.grad.detach().flatten()

        # Samples for the policy-gradient objectives come from the discrete prompt (on-policy).
        def samples_for(batch: Sequence[MathProblem], k: int, seed: int) -> tuple[list[list[str]], torch.Tensor]:
            outs = scorer.generate(batch, instruction, batch_size=args.generation_batch_size,
                                   max_new_tokens=args.max_new_tokens, samples=k,
                                   temperature=args.temperature, seed=seed)
            rewards = torch.tensor([[float(is_correct(extract_answer(t), p.gold_answer)) for t in o]
                                    for o, p in zip(outs, batch)])
            return outs, rewards

        def policy_losses(batch, outs, rewards, mode: str):
            """-(1/|batch|) sum_i sum_j w_ij * mean-token log p(y_ij); RLOO or RWR weights."""
            k = rewards.shape[1]
            if mode == "rloo":
                weights = rewards - (rewards.sum(dim=1, keepdim=True) - rewards) / max(k - 1, 1)
            else:  # reward-weighted regression
                weights = torch.softmax(rewards / args.rwr_temperature, dim=1)

            def losses(e: torch.Tensor):
                for j in range(k):
                    for rows in chunks(batch):
                        if not weights[list(rows), j].abs().sum():
                            continue  # zero weight (e.g. RLOO with equal rewards): no gradient
                        sub = [batch[i] for i in rows]
                        _, _, lp, n = scorer.continuation_logprobs(
                            sub, instruction, [("", outs[i][j]) for i in rows], e)
                        w = weights[list(rows), j].to(lp.device)
                        yield -(w * lp / n.clamp_min(1)).sum() / len(batch)
            return losses

        def nll_losses(batch, key: str):
            def losses(e: torch.Tensor):
                for rows in chunks(batch):
                    sub = [batch[i] for i in rows]
                    if key == "ref_full":
                        pairs = [("", p.reference_reasoning + "\n" + answer_line(p.gold_answer)) for p in sub]
                    else:
                        pairs = [(p.reference_reasoning + "\n", answer_line(p.gold_answer)) for p in sub]
                    lp, n, _, _ = scorer.continuation_logprobs(sub, instruction, pairs, e)
                    yield (-lp / n.clamp_min(1)).sum() / len(batch)
            return losses

        objectives: dict[str, list[torch.Tensor]] = {name: [] for name in ("nll_ref_full", "nll_ref_answer", "rloo", "rwr")}
        degenerate = []
        for b, batch in enumerate(batches):
            objectives["nll_ref_full"].append(gradient(nll_losses(batch, "ref_full")))
            objectives["nll_ref_answer"].append(gradient(nll_losses(batch, "ref_answer")))
            outs, rewards = samples_for(batch, args.samples, args.seed + 100 + b)
            degenerate.append(float((rewards.max(dim=1).values == rewards.min(dim=1).values).float().mean()))
            objectives["rloo"].append(gradient(policy_losses(batch, outs, rewards, "rloo")))
            objectives["rwr"].append(gradient(policy_losses(batch, outs, rewards, "rwr")))
            torch.cuda.empty_cache()

        all_problems = [p for batch in batches for p in batch]
        ref_outs, ref_rewards = samples_for(all_problems, args.reference_samples, args.seed + 999)
        reference = gradient(policy_losses(all_problems, ref_outs, ref_rewards, "rloo"))

        row: dict[str, Any] = {"instruction": instruction, "slot_tokens": len(slot_ids),
                               "fraction_problems_with_equal_rewards": sum(degenerate) / len(degenerate)}
        for name, grads in objectives.items():
            cosines = [float(F.cosine_similarity(a, b, dim=0)) for a, b in itertools.combinations(grads, 2)]
            zero_batches = sum(1 for g in grads if not g.abs().sum())
            alignment = [float(F.cosine_similarity(g, reference, dim=0)) for g in grads]
            row[name] = {
                "cross_batch_cosine_mean": sum(cosines) / len(cosines),
                "alignment_with_reference_pg_mean": sum(alignment) / len(alignment),
                "gradient_norm_mean": sum(float(g.norm()) for g in grads) / len(grads),
                "zero_gradient_batches": zero_batches,
            }
        results.append(row)
        print(f"GRAD {instruction[:60]} :: " + ", ".join(
            f"{k}: cons={row[k]['cross_batch_cosine_mean']:.3f} align={row[k]['alignment_with_reference_pg_mean']:.3f}"
            for k in objectives))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/pilot.yaml", help="Only the model block is used.")
    parser.add_argument("--stage", choices=("rank", "grad", "both"), default="both")
    parser.add_argument("--output-dir", default="artifacts/gsm8k_diagnostics")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--gradient-problems", type=int, default=64)
    parser.add_argument("--instruction-indices", type=int, nargs="*", default=None,
                        help="Subset of INSTRUCTION_POOL for the rank stage (for sharding across GPUs).")
    parser.add_argument("--grad-instruction-indices", type=int, nargs="*", default=[0, 1, 16])
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--reference-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--rwr-temperature", type=float, default=0.5)
    parser.add_argument("--grad-batches", type=int, default=4)
    parser.add_argument("--grad-batch-size", type=int, default=16)
    parser.add_argument("--shard-name", default="all")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = PilotConfig.from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.model.resolved_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.resolved_name_or_path, dtype=torch.bfloat16).to(config.model.device)
    model.requires_grad_(False)
    model.eval()
    builder = GenerativePromptBuilder(tokenizer, enable_thinking=config.prompt.enable_thinking)
    scorer = GenerativeScorer(model, tokenizer, builder, config.model.device)
    eval_problems, gradient_problems = load_problems(args.seed, args.eval_size, args.gradient_problems)
    builder.assert_seed_gates(SEED_INSTRUCTION, [p.question for p in eval_problems[:8]])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"args": vars(args), "eval_problem_ids": [p.problem_id for p in eval_problems],
                "gradient_problem_ids": [p.problem_id for p in gradient_problems],
                "system_prompt": SYSTEM_PROMPT, "response_rule": RESPONSE_RULE,
                "official_test_split_used": False}
    if args.stage in ("rank", "both"):
        indices = args.instruction_indices or list(range(len(INSTRUCTION_POOL)))
        rows = run_rank_stage(scorer, eval_problems, args, [INSTRUCTION_POOL[i] for i in indices])
        (output_dir / f"rank_{args.shard_name}.json").write_text(
            json.dumps({"manifest": manifest, "rows": rows}, indent=2), encoding="utf-8")
    if args.stage in ("grad", "both"):
        instructions = [INSTRUCTION_POOL[i] for i in args.grad_instruction_indices]
        rows = run_grad_stage(scorer, gradient_problems, args, instructions)
        (output_dir / "grad.json").write_text(
            json.dumps({"manifest": manifest, "rows": rows}, indent=2), encoding="utf-8")


def merge_rank(output_dir: str = "artifacts/gsm8k_diagnostics") -> dict[str, Any]:
    """Combine rank shards and compute the surrogate audit."""
    rows = []
    for path in sorted(Path(output_dir).glob("rank_*.json")):
        if path.name == "rank_summary.json":
            continue
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    summary = summarize_rank(rows)
    Path(output_dir, "rank_summary.json").write_text(
        json.dumps({"instructions": len(rows), "summary": summary}, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
