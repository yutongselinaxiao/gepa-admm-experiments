"""Variable-length ADMM: every slot carries a keep-gate, and the projection may leave a slot blank.

Layout: the parent's slot tokens plus `spare_slots` extra slots inserted before the positions where
the task loss is most sensitive (gradient norm). Each slot i has an embedding e_i and a gate logit
a_i with g_i = sigmoid(a_i). Scoring uses CommonEvaluator.score_gated_slot_embeddings, where g -> 0
is exactly the same as deleting the token. Original slots start open (g ~ 0.98), spare slots start
almost closed (g ~ 0.12), so the gradient decides where to delete and where to insert.

Scaled ADMM over the pair (embeddings, gates):
- E-step (Adam): CE(e, g) + rho/2 sum_i k_i ||e_i - E(z_i) + u_i||^2 + gate_rho/2 sum_i (g_i - k_i + v_i)^2,
  where k_i = 1 if slot i currently holds a token and 0 if it is blank.
- Z-step (beam search, left to right): each slot is either blank, at cost gate_weight * t_i^2, or a
  token w from the LM nucleus (plus the parent's token at original slots), at cost
  d_i(w) - min_w' d_i(w') + gate_weight * (1 - t_i)^2 + fluency_weight * NLL(w | prefix, chosen tokens),
  with d_i(w) = ||s_i - E(w)||^2 measured relative to the closest candidate (a blank has no embedding),
  with s_i = e_i + u_i and t_i = g_i + v_i. The final slot token (trailing newline) is never blank.
- Dual step: u_i += e_i - E(z_i) for slots holding a token (reset to 0 when blank); v_i += g_i - k_i.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .admm import ADMMRefiner, gold_indices
from .data import NewsExample


@dataclass(frozen=True)
class GatedConfig:
    spare_slots: int = 4
    gate_rho: float = 1.0
    gate_weight: float = 10.0
    gate_learning_rate: float = 0.1
    kept_logit: float = 4.0
    spare_logit: float = -2.0


@dataclass
class GatedState:
    reference_instruction: str  # supplies the prompt prefix/suffix only
    prefix_ids: tuple[int, ...]
    initial: list[int | None]  # the parent's token at original slots, None at spare slots
    e: torch.Tensor
    a: torch.Tensor
    z: list[int | None]  # None = blank
    u: torch.Tensor
    v: torch.Tensor
    optimizer: torch.optim.Optimizer
    forward_calls: int = 0
    backward_calls: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class GatedADMMRefiner:
    def __init__(self, refiner: ADMMRefiner, config: GatedConfig) -> None:
        self.refiner = refiner
        self.config = config
        self.evaluator = refiner.evaluator
        self.hp = refiner.hp
        self.embedding_weight = refiner.embedding_weight

    def init_state(
        self, parent: str, parent_ids: torch.Tensor, prefix_ids: Sequence[int], batch: Sequence[NewsExample]
    ) -> GatedState:
        """Insert spare slots before the most loss-sensitive positions (never after the final token)."""
        evaluator = self.evaluator
        e = self.embedding_weight.index_select(0, parent_ids).detach().float().clone().requires_grad_(True)
        targets = gold_indices(batch, evaluator.labels, evaluator.device)
        F.cross_entropy(evaluator.score_slot_embeddings(batch, parent, e).float(), targets).backward()
        saliency = e.grad.norm(dim=1)[:-1].cpu()
        spare_before = set(saliency.argsort(descending=True)[: self.config.spare_slots].tolist())
        ids = parent_ids.cpu().tolist()
        initial: list[int | None] = []
        tokens: list[int] = []
        for position, token in enumerate(ids):
            if position in spare_before:  # the LM's most likely next token seeds the spare embedding
                context = torch.tensor([list(prefix_ids) + ids[:position]], device=evaluator.device)
                with torch.inference_mode():
                    logits = evaluator.model(input_ids=context).logits[0, -1].float()
                    seed = int(logits.masked_fill(~self.refiner.allowed_mask, -math.inf).argmax())
                initial.append(None)
                tokens.append(seed)
            initial.append(token)
            tokens.append(token)
        ids_t = torch.tensor(tokens, device=evaluator.device)
        e = self.embedding_weight.index_select(0, ids_t).detach().float().clone().requires_grad_(True)
        a = torch.tensor(
            [self.config.kept_logit if t is not None else self.config.spare_logit for t in initial],
            device=evaluator.device,
        ).requires_grad_(True)
        optimizer = torch.optim.Adam(
            [{"params": [e], "lr": self.hp.learning_rate},
             {"params": [a], "lr": self.config.gate_learning_rate}]
        )
        return GatedState(
            reference_instruction=parent, prefix_ids=tuple(int(i) for i in prefix_ids), initial=initial,
            e=e, a=a, z=list(initial), u=torch.zeros_like(e), v=torch.zeros_like(a), optimizer=optimizer,
            forward_calls=len(spare_before) + 1, backward_calls=1,
        )

    def token_ids(self, state: GatedState) -> torch.Tensor:
        return torch.tensor([t for t in state.z if t is not None], dtype=torch.long, device=self.evaluator.device)

    def step(self, state: GatedState, batch: Sequence[NewsExample]) -> float:
        cfg, device = self.config, self.evaluator.device
        targets = gold_indices(batch, self.evaluator.labels, device)
        kept = torch.tensor([t is not None for t in state.z], dtype=torch.float32, device=device)
        ez = self.embedding_weight.index_select(
            0, torch.tensor([t if t is not None else 0 for t in state.z], device=device)
        ).detach().float()
        last = math.nan
        for _ in range(self.hp.e_steps):
            state.optimizer.zero_grad(set_to_none=True)
            scores = self.evaluator.score_gated_slot_embeddings(
                batch, state.reference_instruction, state.e, F.logsigmoid(state.a)
            )
            task = F.cross_entropy(scores.float(), targets)
            penalty = 0.5 * self.hp.rho * (kept[:, None] * (state.e - ez + state.u)).square().sum()
            gate_penalty = 0.5 * cfg.gate_rho * (state.a.sigmoid() - kept + state.v).square().sum()
            (task + penalty + gate_penalty).backward()
            state.optimizer.step()
            state.forward_calls += 1
            state.backward_calls += 1
            last = float(task.detach())
        with torch.no_grad():
            gates = state.a.sigmoid()
            state.z = self.project(state, state.e.detach() + state.u, gates + state.v)
            kept_new = torch.tensor([t is not None for t in state.z], dtype=torch.float32, device=device)
            ez_new = self.embedding_weight.index_select(
                0, torch.tensor([t if t is not None else 0 for t in state.z], device=device)
            ).float()
            state.u = kept_new[:, None] * (state.u + state.e.detach() - ez_new)
            state.v = state.v + gates - kept_new
        return last

    def _candidates(self, logits: torch.Tensor, initial: int | None) -> torch.Tensor:
        probabilities = logits.float().softmax(dim=-1)
        sorted_p, sorted_ids = probabilities.sort(descending=True)
        candidates = sorted_ids[(sorted_p.cumsum(0) - sorted_p) < self.hp.top_p]
        candidates = candidates[self.refiner.allowed_mask[candidates]]
        if initial is not None:
            candidates = torch.cat((candidates, torch.tensor([initial], device=candidates.device)))
        return torch.unique(candidates)

    @torch.inference_mode()
    def _next_logits(self, state: GatedState, sequences: list[tuple[int, ...]]) -> torch.Tensor:
        """Next-token logits after prefix + each sequence (right-padded batch, no cache)."""
        device = self.evaluator.device
        rows = [list(state.prefix_ids) + list(seq) for seq in sequences]
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), int(self.evaluator.tokenizer.pad_token_id), device=device)
        mask = torch.zeros_like(ids)
        for r, row in enumerate(rows):
            ids[r, : len(row)] = torch.tensor(row, device=device)
            mask[r, : len(row)] = 1
        logits = self.evaluator.model(input_ids=ids, attention_mask=mask).logits
        state.forward_calls += 1
        return logits[torch.arange(len(rows), device=device), torch.tensor([len(r) - 1 for r in rows], device=device)]

    @torch.inference_mode()
    def project(self, state: GatedState, target_e: torch.Tensor, target_g: torch.Tensor) -> list[int | None]:
        cfg, lam = self.config, self.hp.fluency_weight
        beams: list[tuple[float, tuple[int, ...], tuple[int | None, ...]]] = [(0.0, (), ())]
        final = len(state.z) - 1
        for i in range(len(state.z)):
            sequences = list(dict.fromkeys(tokens for _, tokens, _ in beams))
            logits = self._next_logits(state, sequences)
            by_sequence = {seq: logits[k] for k, seq in enumerate(sequences)}
            t = float(target_g[i])
            options: list[tuple[float, tuple[int, ...], tuple[int | None, ...]]] = []
            for score, tokens, choices in beams:
                if i != final:
                    options.append((score + cfg.gate_weight * t * t, tokens, choices + (None,)))
                row = by_sequence[tokens]
                candidates = self._candidates(row, state.initial[i])
                vectors = self.embedding_weight.index_select(0, candidates).float()
                distance = (vectors - target_e[i].float()).square().sum(1)
                # Relative to the closest candidate: the ADMM target drifts in absolute terms, and a
                # blank has no embedding to compare, so keep-vs-blank is left to the gates.
                cost = distance - distance.min() - lam * row.float().log_softmax(-1)[candidates]
                cost = cost + cfg.gate_weight * (1 - t) ** 2 + score
                best = cost.topk(min(self.hp.beam_width, len(cost)), largest=False)
                for c, k in zip(best.values.tolist(), best.indices.tolist()):
                    w = int(candidates[k])
                    options.append((c, tokens + (w,), choices + (w,)))
            options.sort(key=lambda o: o[0])
            seen, beams = set(), []
            for option in options:  # keep distinct token sequences
                if option[1] not in seen:
                    seen.add(option[1])
                    beams.append(option)
                if len(beams) == self.hp.beam_width:
                    break
        return list(beams[0][2])
