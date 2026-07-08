# Project status: IN PROGRESS (pipeline running)
Updated: 2026-07-08

## Done
- PLAN.md pre-registered and committed BEFORE any results (kill conditions K1-K3;
  one amendment, also pre-results: paired pruned-real linear-chain null, since natural
  linear chains are near-absent in ProsQA: 0/800 in test+valid, 5/17,886 in train).
- lrspec package + 11 passing tests (step-map self-consistency exact; Jacobian vs finite
  differences; M4 multipass == single pass; unrolled influence map consistency).
- Assets: 4 best checkpoints (M1-M4) + curriculum epochs, ProsQA all splits.
- Sanity gate: M2 98.0% (pub. 97.0), M3 95.4% (pub. 96.6) on ProsQA test under our
  decode protocol — harness validated. M4/M1 in flight.
- Recency sweep 2026-07-08 (lit_notes.md): cell still open; all load-bearing citations
  verified against live arXiv.

## Running
- E0-E3 pipeline (scripts/run_all.py) on the 3080, sharing the GPU with other
  sessions' jobs (one model per subprocess keeps our footprint ~3 GB).

## Next
- E0 analyze -> K1 verdict (go/no-go), then E1 (K2), E2 (K3), E3.
- make_figures.py -> paper/figs + numbers.tex; paper writing after verdicts.

## Notes / gotchas
- GPU shared with concurrent qwen7b + other experiment processes; expect slowdown, not
  incorrectness. All stages are per-process and checkpointed every 20-25 problems.
- transformers 4.55 GPT-2 still returns legacy tuple KV caches; harness handles both.
- Published-vs-ours accuracy deltas (±1.2pp) trace to decode protocol (max_new_tokens,
  extraction); fine for a sanity gate, don't quote our numbers as reproductions of theirs
  without the protocol caveat.
