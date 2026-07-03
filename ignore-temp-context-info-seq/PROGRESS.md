# Project state / handoff notes

**Goal:** arXiv-ready paper on SemRF (Semantic Reference Frames) — attention bias
representing context/time relative to learned semantic anchors instead of
absolute positions. See `info.txt` (original brief), `README.md` (repo guide),
`paper/main.tex` (draft, compiles with `tools/tectonic.exe`).

**Scope decisions (user-approved):** synthetic diagnostics + enwik8 char-LM,
3 seeds synthetic (2 for ablations), autonomous execution with milestone
check-ins.

## Status at last update (2026-07-02 morning)

- [x] Library, 7 schemes, 3 tasks, pipelines, drivers, analysis, paper skeleton.
- [x] CHAR-LM DONE (14/14 runs in `results/charlm/enwik8/`). Headline results:
      test bpc @512: SemRF 1.4245±0.0006 (best) < RoPE 1.4276 < ALiBi 1.4348.
      Extrapolation bpc @4096 (8x train): SemRF 1.341 < ALiBi 1.365 (margin
      GROWS with length); RoPE 4.62, sinusoidal 35.3 (catastrophic).
- [x] PROBE3 DONE — double dissociation confirmed:
      TR-easy: alibi 1.000, semrf 1.000 (rope was 0.39 — decay prior needed).
      AR-locked: alibi 0.26 (rope 1.00 — decay hurts long-range retrieval).
- [x] Anchor analysis done (charlm semrf seed0 ckpt): anchors = character
      classes (lowercase/digits/uppercase/punctuation). Slow-decay frames:
      '|', punctuation, whitespace (document structure); fast-decay: lowercase
      (word-local). Within-head relative slope spread up to 88%.
      Figures: fig_anchors, fig_frame_time. `results/anchor_clusters.json`.
- [x] SYNTHETIC SWEEP DONE (99/99). TR: semrf/alibi 1.000 at ALL lengths;
      learned 1.0->0.36@4x (saturation story); rope bistable 0.60±0.28;
      no_time ablation 0.997->0.545@4x (causal evidence for time term).
      SC: t5 0.95 / nope 0.86 / semrf 0.73 > alibi 0.70 / rope+learned <0.10
      @4x; no_time 0.88 -> honest trade-off (decay costs pure copying).
      AR@gap(0,32) was TOO HARD (all baselines ~0.26-0.33 guessing; only
      semrf seed0 solved 1.000) -> preserved as results/synthetic/
      assoc_recall_hard (task field retagged), reported as secondary finding.
- [x] AR RERUN DONE (gap 0-8): SemRF 1.000±0.000 (3/3 seeds — ONLY method
      that always forms the circuit); alibi 2/3 (0.75±0.35, scale-free extrap
      when formed); all others 0/3 (~0.26). no_time ablation 0.33 -> time term
      causally necessary for formation. SemRF extrap: 0.87@gap64, 0.39@gap160+
      (learned slopes adapt to train gap scale — honest limitation, in paper).
- [x] ALL figures/tables final (per-task ablation chart, fixed distance fig,
      stacked anchors fig); significance tests in results/.
- [x] PAPER COMPLETE: all prose final, hard-regime appendix, claims verified
      16/16 against result files by automated check. 13 pages, compiles clean.
- [x] Reproducibility polish done; PDF delivered (v1).
- [x] CABLE ADDED AS 8TH BASELINE (2026-07-02, user-requested pre-submission):
      implemented faithfully from github.com/axiomlab/cable (per-layer
      Linear(d->H), -softplus cumsum bias S_i-S_j, per-layer recompute,
      default init; released code omits paper's g_theta — matched code).
      Results: enwik8 bpc 1.4409±0.0019 (4th; SemRF still best at ALL
      lengths, 1.340@4096 vs CABLE 1.352); AR 1.000 3/3 + PERFECT gap
      extrapolation (CABLE wins AR — near-zero increments on filler);
      TR 1.000 all lengths; SC FAILS training (0.174±0.042 — no ordinal
      position signal). NEW headline: SemRF = only scheme of 8 solving all
      three diagnostics at train length. Abstract now leads with
      interpretability hook. All claims re-verified (9/9 CABLE + 16/16
      original). Paper 14 pp, compiles clean. READY FOR SUBMISSION.

## Novelty assessment (2026-07-02, user-prompted)

Challenge: "Moschella et al. 2023 already did this." VERDICT: NO — that paper
(Relative representations, ICLR 2023) is latent-space communication / model
stitching via sample-anchored similarity vectors; zero content on position,
attention, time, or extrapolation (verified via abstract + full-text checks).
It IS the inspiration for the anchor-relative principle → now cited prominently
(intro + dedicated related-work paragraph).
REAL adjacent work found in the same search: CABLE (Veisi & Mansourian, arXiv
2503.08067) — content-conditioned per-token additive biases generalizing ALiBi
via cumulative sums. Differentiation: SemRF uses structured anchor-frame
factorization (frame membership + residual + FRAME-conditioned decay), yields
interpretability (character-class frames, coherent time constants), and adds
the double-dissociation diagnostics. Claims scoped accordingly: we do NOT claim
"first content-conditioned bias." Also added FIRE (Li et al. 2023) citation.
User was told; task continues unless they override.

## Key experimental findings so far (from debugging probes)

1. **SemRF init fix (implemented):** content biases at init drowned positional
   signal (selective_copy 0.12). Now initializes exactly at ALiBi's operating
   point (per-frame slopes = ALiBi head slopes, content gates ≈ 0.05).
   Fix verified: selective_copy 0.998.
2. **Induction phase transition:** assoc_recall (vocab 16/16, 8 pairs, 8 queries)
   needs d=256/8h AND ~15k steps: acc 0.26 @6k -> 1.00 @15k (rope). d=128 never
   forms the circuit (even 15k). Consistent with Zoology/MQAR capacity results.
3. **Locked task configs** (in `experiments/run_synthetic.py` TASK_SPECS):
   - assoc_recall: vocab 16/16, n_pairs=8, n_queries=8, train gap=(0,32),
     eval gaps 16/64/160/352, steps=15000
   - temporal_recency: PENDING probe3 verdict (rope fails even easy config 0.39;
     alibi/semrf may succeed via recency decay — if all fail, drop TR from paper)
   - selective_copy: n_data=16, context 128->512 eval, steps=5000
   - model: d=256, 4L, 8H, lr=1e-3, bs=64
4. **Emerging trade-off story:** SemRF/decay methods fail AR-with-gap (0.35)
   where RoPE succeeds (1.0); RoPE fails temporal recency. If probe3 confirms
   (alibi fails AR too / solves TR), report honestly as extrapolation-vs-retrieval
   trade-off; SemRF's learnable per-frame slopes are the escape hatch in theory.
5. Char-LM runs are independent of all this and form the headline evidence
   (bpc + length extrapolation 512->4096).

## How to resume in a fresh session

1. Read this file, `README.md`, `info.txt`.
2. Check `results/charlm/enwik8/*.json` — if <14 files, relaunch the charlm
   command above (it skips completed runs).
3. Read probe3 results from the task output path above (or rerun
   `python -u scripts/probe3.py`, ~40 min) -> decide TR in/out -> set
   temporal_recency `steps` in TASK_SPECS (10000) or drop it from the sweep.
4. Launch synthetic sweep, then analysis/figures/anchor scripts.
5. Fill `paper/results_macros.tex` from `results/*.csv` + write results prose;
   recompile paper (`cd paper && ../tools/tectonic.exe main.tex`).
6. Reproducibility polish (task #8), final consistency check paper vs results.
