"""Model harness: load M1-M4 checkpoints, run the latent phase, capture trajectories.

The Coconut protocol (facebookresearch/coconut, bmarti44 curriculum checkpoints):
  input = tokenize(question + "\n") + [<|start-latent|>] + 6*[<|latent|>] + [<|end-latent|>]
  M2 (continuous): the embedding fed at latent slot t is the last-layer hidden state of the
  previous position (hidden-state recycling).  M3/M4 (pause): a learned pause embedding is
  fed at every latent slot; at inference the multi-pass variant (M4) is numerically
  identical to a single causal pass, which we verify in tests.

Trajectory convention (fixed in PLAN.md):
  c_t (t=1..6) is the vector FED at latent slot t.  c_1 = last hidden at the
  <|start-latent|> position; for M2, c_{t+1} = last hidden at slot t.  The local step map
  F_t : c_t -> c_{t+1} holds the prefix KV cache (positions before slot t) fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.cache_utils import DynamicCache

from .paths import CHECKPOINTS, FEEDBACK_MODE
from .prosqa import Problem

N_LATENT = 6
MAX_NEW_TOKENS = 16


def _build_tokenizer() -> GPT2Tokenizer:
    tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tok.add_special_tokens(
        {"additional_special_tokens": ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]}
    )
    return tok


@dataclass
class LatentRun:
    """Captured latent phase of one problem."""

    problem_idx: int
    input_ids: list[int]          # question + start + 6 latent + end
    slot_pos: list[int]           # absolute positions of the 6 latent slots
    start_pos: int                # position of <|start-latent|>
    end_pos: int                  # position of <|end-latent|>
    hs: torch.Tensor              # (7, 768): last hidden at [start-latent, slot 1..6]
    kv: list[tuple[torch.Tensor, torch.Tensor]]  # full-length per-layer (k, v), detached
    # For M2 (continuous), the fed thought c_t (t=1..6) equals hs[t-1]; hs[6] is the
    # hidden at slot 6 (never fed back).  For pause modes hs records the same hidden
    # trajectory but the fed vector is always the pause embedding.


class Harness:
    def __init__(self, model_key: str, device: str = "cuda", checkpoint_path=None):
        assert model_key in CHECKPOINTS
        self.model_key = model_key
        self.mode = FEEDBACK_MODE[model_key]
        self.device = device
        self.tok = _build_tokenizer()
        self.latent_id = self.tok.convert_tokens_to_ids("<|latent|>")
        self.start_id = self.tok.convert_tokens_to_ids("<|start-latent|>")
        self.end_id = self.tok.convert_tokens_to_ids("<|end-latent|>")
        self.eos_id = self.tok.eos_token_id

        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        model.resize_token_embeddings(len(self.tok))
        sd = torch.load(checkpoint_path or CHECKPOINTS[model_key], map_location="cpu",
                        weights_only=True)
        if any(k.startswith("base_causallm.") for k in sd):
            base_sd = {k[len("base_causallm."):]: v for k, v in sd.items()
                       if k.startswith("base_causallm.")}
        else:
            # M1 (CoT) was trained as a bare GPT2LMHeadModel, no Coconut wrapper
            base_sd = dict(sd)
        missing, unexpected = model.load_state_dict(base_sd, strict=False)
        # lm_head is tied to wte; "missing" entries must be none, unexpected none
        assert not missing and not unexpected, (missing, unexpected)
        self.pause_embedding = None
        if "pause_embedding" in sd:
            self.pause_embedding = sd["pause_embedding"].to(device)
        if self.mode == "pause":
            assert self.pause_embedding is not None, f"{model_key}: no pause_embedding"

        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.emb = model.transformer.get_input_embeddings()

    # ---------- tokenization ----------

    def encode(self, problem: Problem) -> tuple[list[int], list[int], int, int]:
        q_ids = self.tok.encode(problem.question + "\n", add_special_tokens=True)
        ids = q_ids + [self.start_id] + [self.latent_id] * N_LATENT + [self.end_id]
        start_pos = len(q_ids)
        slot_pos = [start_pos + 1 + i for i in range(N_LATENT)]
        end_pos = slot_pos[-1] + 1
        return ids, slot_pos, start_pos, end_pos

    def answer_ids(self, problem: Problem, symbol_idx: int) -> list[int]:
        root_name = problem.idx_to_symbol[problem.root]
        sym = problem.idx_to_symbol[symbol_idx]
        text = f"### {root_name} is a {sym}."
        return self.tok.encode(text, add_special_tokens=False) + [self.eos_id]

    # ---------- cache plumbing ----------

    @staticmethod
    def _cache_from(kv, upto: int) -> DynamicCache:
        """DynamicCache over positions [0, upto) from stored per-layer (k, v)."""
        legacy = tuple(
            (k[:, :, :upto, :], v[:, :, :upto, :]) for k, v in kv
        )
        return DynamicCache.from_legacy_cache(legacy)

    @staticmethod
    def _kv_of(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
        legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
        return [(k.detach(), v.detach()) for k, v in legacy]

    def _fwd_one(self, embed: torch.Tensor, pos: int, cache: DynamicCache):
        """Single-token forward. Returns (last hidden (768,), logits (V,), cache)."""
        out = self.model(
            inputs_embeds=embed.view(1, 1, -1),
            position_ids=torch.tensor([[pos]], device=self.device),
            attention_mask=torch.ones(1, pos + 1, device=self.device, dtype=torch.long),
            past_key_values=cache,
            output_hidden_states=True,
            use_cache=True,
        )
        return out.hidden_states[-1][0, -1], out.logits[0, -1], out.past_key_values

    # ---------- latent phase ----------

    @torch.no_grad()
    def run_latent(self, problem: Problem) -> LatentRun:
        ids, slot_pos, start_pos, end_pos = self.encode(problem)
        prefix = torch.tensor([ids[: start_pos + 1]], device=self.device)
        out = self.model(
            input_ids=prefix,
            position_ids=torch.arange(prefix.shape[1], device=self.device).view(1, -1),
            output_hidden_states=True,
            use_cache=True,
        )
        cache = out.past_key_values
        h = out.hidden_states[-1][0, -1]  # hidden at <|start-latent|> (= c_1 for M2)
        hs = [h]
        for t in range(1, N_LATENT + 1):
            fed = hs[t - 1] if self.mode == "continuous" else self.pause_embedding
            h, _, cache = self._fwd_one(fed, slot_pos[t - 1], cache)
            hs.append(h)
        # feed <|end-latent|> as a normal token
        h, _, cache = self._fwd_one(self.emb.weight[self.end_id], end_pos, cache)
        return LatentRun(
            problem_idx=problem.idx,
            input_ids=ids,
            slot_pos=slot_pos,
            start_pos=start_pos,
            end_pos=end_pos,
            hs=torch.stack(hs),
            kv=self._kv_of(cache),
        )

    # For M2 the fed vector IS the previous hidden; for pause modes the fed vector is the
    # pause embedding at every slot.  fed_vector(t) returns what was fed at slot t (1-based).
    def fed_vector(self, run: LatentRun, t: int) -> torch.Tensor:
        if self.mode == "continuous":
            return run.hs[t - 1]
        return self.pause_embedding

    # ---------- step map + Jacobian ----------

    def step_function(self, run: LatentRun, t: int):
        """f: R^768 -> R^768, embedding fed at slot t -> last hidden at slot t.

        Prefix KV (positions < slot_t) frozen.  For M2 this is the local step map
        F_t : c_t -> c_{t+1} of the recurrence (PLAN.md sec. 3); for M3/M4 it is the
        matched-control map at the identical position.
        """
        pos = run.slot_pos[t - 1]
        kv = run.kv

        def f(c: torch.Tensor) -> torch.Tensor:
            cache = self._cache_from(kv, pos)
            out = self.model(
                inputs_embeds=c.view(1, 1, -1),
                position_ids=torch.tensor([[pos]], device=self.device),
                attention_mask=torch.ones(1, pos + 1, device=self.device,
                                          dtype=torch.long),
                past_key_values=cache,
                output_hidden_states=True,
                use_cache=False,
            )
            return out.hidden_states[-1][0, -1]

        return f

    def jacobian(self, run: LatentRun, t: int) -> torch.Tensor:
        """J_t = d c_{t+1} / d c_t (768x768) at the realized fed vector."""
        f = self.step_function(run, t)
        c = self.fed_vector(run, t).detach().requires_grad_(True)
        J = torch.autograd.functional.jacobian(f, c, vectorize=True)
        return J.detach()

    def unrolled_function(self, run: LatentRun, t: int, k: int):
        """g: c_t -> c_{t+k}, differentiating through ALL paths (incl. KV written at
        intermediate latent slots).  M2 only.  Prefix KV before slot t frozen.
        """
        assert self.mode == "continuous"
        assert 1 <= t and t + k <= N_LATENT + 1
        pos0 = run.slot_pos[t - 1]
        kv = run.kv

        def g(c: torch.Tensor) -> torch.Tensor:
            embeds = [c]
            cur = None
            for i in range(k):
                pos_hi = run.slot_pos[t - 1 + i] + 1
                cache = self._cache_from(kv, pos0)  # fresh graph-free prefix each pass
                seq = torch.stack(embeds).view(1, len(embeds), -1)
                out = self.model(
                    inputs_embeds=seq,
                    position_ids=torch.arange(pos0, pos_hi, device=self.device).view(1, -1),
                    attention_mask=torch.ones(1, pos_hi, device=self.device,
                                              dtype=torch.long),
                    past_key_values=cache,
                    output_hidden_states=True,
                    use_cache=False,
                )
                cur = out.hidden_states[-1][0, -1]
                embeds.append(cur)
            return cur

        return g

    def influence_jacobian(self, run: LatentRun, t: int, k: int) -> torch.Tensor:
        """G_{t->t+k} = d c_{t+k} / d c_t through all paths (E2)."""
        g = self.unrolled_function(run, t, k)
        c = self.fed_vector(run, t).detach().requires_grad_(True)
        G = torch.autograd.functional.jacobian(g, c, vectorize=True)
        return G.detach()

    # ---------- answer readout ----------

    @torch.no_grad()
    def _teacher_forced_logprob(self, cache_kv, first_logits: torch.Tensor,
                                start_pos: int, token_ids: list[int]) -> float:
        """Sum log P(token_ids) continuing after the cached prefix.

        first_logits: logits at the last cached position (predicting token_ids[0]).
        """
        logp = 0.0
        logits = first_logits
        cache = self._cache_from(cache_kv, start_pos)
        for i, tid in enumerate(token_ids):
            logp += torch.log_softmax(logits.float(), dim=-1)[tid].item()
            if i == len(token_ids) - 1:
                break
            _, logits, cache = self._fwd_one(
                self.emb.weight[tid], start_pos + i, cache
            )
        return logp

    @torch.no_grad()
    def readout(self, run: LatentRun, problem: Problem) -> dict:
        """Greedy answer + teacher-forced margin, continuing after <|end-latent|>."""
        n_ctx = run.end_pos + 1  # cached positions incl. end-latent
        # logits at end-latent position: recompute from cache
        cache = self._cache_from(run.kv, run.end_pos)
        _, logits_end, _ = self._fwd_one(
            self.emb.weight[self.end_id], run.end_pos, cache
        )
        tgt = self.answer_ids(problem, problem.target)
        neg = self.answer_ids(problem, problem.neg_target)
        lp_t = self._teacher_forced_logprob(run.kv, logits_end, n_ctx, tgt)
        lp_n = self._teacher_forced_logprob(run.kv, logits_end, n_ctx, neg)

        # greedy decode
        toks = []
        logits = logits_end
        cache = self._cache_from(run.kv, n_ctx)
        for i in range(MAX_NEW_TOKENS):
            nxt = int(torch.argmax(logits).item())
            if nxt == self.eos_id:
                break
            toks.append(nxt)
            _, logits, cache = self._fwd_one(
                self.emb.weight[nxt], n_ctx + i, cache
            )
        text = self.tok.decode(toks)
        pred_sym = None
        m = text.strip().split()
        if m and m[-1].rstrip(".").isalpha():
            pred_sym = m[-1].rstrip(".")
        correct = pred_sym == problem.idx_to_symbol[problem.target]
        return {
            "margin": lp_t - lp_n,
            "logp_target": lp_t,
            "logp_neg": lp_n,
            "greedy_text": text,
            "correct": bool(correct),
        }

    # ---------- counterfactual rerun (ablations / interventions) ----------

    @torch.no_grad()
    def rerun_from(self, run: LatentRun, problem: Problem, t: int,
                   c_t_new: torch.Tensor, greedy: bool = True) -> dict:
        """Feed c_t_new at slot t, recompute the downstream recurrence + readout (M2).

        With greedy=False, only the teacher-forced margin is computed (fast path for
        intervention sweeps; answer flips are then judged by the margin's sign).
        """
        assert self.mode == "continuous"
        pos = run.slot_pos[t - 1]
        cache = self._cache_from(run.kv, pos)
        c = c_t_new
        for i in range(t, N_LATENT + 1):
            h, _, cache = self._fwd_one(c, run.slot_pos[i - 1], cache)
            c = h
        h, logits_end, cache = self._fwd_one(
            self.emb.weight[self.end_id], run.end_pos, cache
        )
        kv = self._kv_of(cache)
        n_ctx = run.end_pos + 1
        tgt = self.answer_ids(problem, problem.target)
        neg = self.answer_ids(problem, problem.neg_target)
        lp_t = self._teacher_forced_logprob(kv, logits_end, n_ctx, tgt)
        lp_n = self._teacher_forced_logprob(kv, logits_end, n_ctx, neg)
        out = {"margin": lp_t - lp_n}
        if not greedy:
            return out

        toks = []
        logits = logits_end
        cache2 = self._cache_from(kv, n_ctx)
        for i in range(MAX_NEW_TOKENS):
            nxt = int(torch.argmax(logits).item())
            if nxt == self.eos_id:
                break
            toks.append(nxt)
            _, logits, cache2 = self._fwd_one(self.emb.weight[nxt], n_ctx + i, cache2)
        text = self.tok.decode(toks)
        m = text.strip().split()
        pred_sym = m[-1].rstrip(".") if m else ""
        out.update({
            "greedy_text": text,
            "correct": bool(pred_sym == problem.idx_to_symbol[problem.target]),
            "pred_neg": bool(pred_sym == problem.idx_to_symbol[problem.neg_target]),
        })
        return out
