# PLAN.md — Pre-registration

**Project:** Spectral Invariants of Latent Reasoning: Predicting and Causally Explaining
Anchor/Branch Structure in Continuous Chain-of-Thought

**Author:** Gunner Levi Howe (gunnerlevihowe@gmail.com)
**Date of pre-registration:** 2026-07-08 (committed BEFORE any experimental results exist in this repository)
**Hardware:** single RTX 3080 (10 GB); no training — public checkpoints only.

---

## 1. Claim under test

A Coconut-family latent chain-of-thought reasons by feeding its last hidden state back as the
next input embedding, producing a trajectory of continuous thoughts c_1..c_n — a discrete-time
dynamical system c_{t+1} = F_t(c_t). **Claim:** the interpretable structure of the reasoning —
high-influence "anchor" thoughts and branch points — is *predicted* by the dynamical invariants
of F:

- **Branch points** ↔ transient-expansion / non-normal-amplification signatures of the local
  step Jacobian J_t = ∂c_{t+1}/∂c_t (top singular value σ₁, count of expanding singular
  directions, Henrici departure from normality, σ₁/ρ non-normality ratio).
- **Anchor thoughts** ↔ dominant/slow modes of J_t (spectral radius ρ, near-unit spectral
  mass — the RKSP-style diagnostic — and alignment of the trajectory step with the dominant
  eigen/singular directions).

The phenomenon itself (anchors exist; step effects are heterogeneous and routed) is
**established background** (arXiv:2602.08783, arXiv:2606.12689). We do NOT claim it. Our
contribution is the spectral *predictor* and *causal explanation* of it, validated against
ProsQA's known ground-truth DAG with matched null controls.

## 2. Substrate (fixed before results)

- Checkpoints: `bmarti44/coconut-curriculum-checkpoints` (HuggingFace), GPT-2 124M base,
  7-stage curriculum, seed 0. Variants: M1 `cot-baseline` (best = epoch 44, 83.0%),
  M2 `coconut` (best = epoch 49, 97.0%), M3 `pause-curriculum` (best = epoch 43, 96.6%),
  M4 `pause-multipass` (best = epoch 30, 94.8%). `checkpoint_best` used throughout.
- Data: `facebookresearch/coconut` `data/prosqa_test.json` (500 problems) and
  `prosqa_valid.json`. Each record carries the full DAG (`edges`), the ground-truth path
  (`steps`, `root`, `target`) and the distractor (`neg_target`).
- Eval input (Coconut protocol, c_thought=1, max_latent_stage=6, pad_latent_to_max=True):
  `tokenize(question+"\n") + [<|start-latent|>] + 6×[<|latent|>] + [<|end-latent|>]`,
  then greedy decode `### {root} is a {concept}.`.

## 3. Definitions (fixed before results)

- **Trajectory.** c_t (t = 1..6) is the vector *fed* at latent slot t; c_1 = last hidden at
  the `<|start-latent|>` position; c_{t+1} = last hidden at slot-t position. Local step map
  F_t: c_t ↦ c_{t+1} holds the prefix KV cache fixed; J_t = ∂c_{t+1}/∂c_t ∈ R^{768×768}
  (torch.func.jacrev), t = 1..5.
- **Matched-control Jacobian (M3/M4).** The identical architectural object at the identical
  positions: J̃_t = ∂h(slot_t)/∂e(slot_t) evaluated at the pause embedding. For M2 this map's
  Jacobian IS the step map of the recurrence; for M3/M4 the recurrence is absent, so any
  spectral structure appearing equally there is architectural, not dynamical.
- **Step ↔ DAG alignment.** Coconut's curriculum replaces reasoning step t with latent
  thought t. Ground-truth path nodes v_0=root, v_1, ..., v_L from `steps` (L = 3 or 4 hops).
  Thought step t (1 ≤ t ≤ L) executes the hop v_{t-1} → v_t.
- **Branch label (per step t).** branch(t) = 1 iff out-degree of v_{t-1} in the problem's DAG
  is > 1 (a real choice existed when leaving v_{t-1}). Steps t > L (padding thoughts beyond
  the path) are excluded from branch AUROC.
- **Graded branch label.** out-degree of v_{t-1} (for secondary rank correlation).
- **Anchor ground truth (per step t).** Causal influence I_t measured by our own step
  ablation: replace c_t with the dataset-mean thought vector (and, robustness: zero vector,
  and a random other-problem thought at the same slot), recompute downstream latents +
  answer; I_t = |Δ answer margin| where margin = log P(target answer) − log P(neg_target
  answer), teacher-forced. Anchor(t) = 1 iff I_t is in the top tercile within-problem
  (primary), with the continuous I_t used for rank statistics.
- **Linear-chain null instances.** Test/valid problems whose ground-truth path has NO
  branch step (out-degree of every v_0..v_{L-1} equals 1). If < 20 such instances exist in
  test+valid, mine additional ones from prosqa_train.json (inference only; no training).
  **AMENDMENT (2026-07-08, before E0 was run):** mining found 0 linear-chain instances in
  test+valid and only 5 in the 17,886-problem train split (branching is intrinsic to the
  ProsQA generator). The instance-level null is therefore constructed as a PAIRED
  PRUNED-REAL control: for ≥100 test problems, delete from the question exactly those
  off-path out-edges of ground-truth path nodes (so every path node has out-degree 1),
  keeping all other statements, ordering, phrasing, and the final question intact; skip
  problems where the pruning removes every mention of neg_target. This yields a
  linear-chain twin of each real problem and enables paired (within-problem)
  branch-signature comparisons. The 5 natural train linear chains are run as a secondary
  check. Model accuracy on the pruned set is reported (distribution-shift guard).
- **Answer margin.** log-prob difference of the full tokenized correct answer
  `### {root} is a {target}.` vs `### {root} is a {neg_target}.`, teacher-forced after
  `<|end-latent|>`.

## 4. Experiment ladder (gated)

### E0 — Week-1 go/no-go: spectral prediction + null controls
1. For M2 on ProsQA test (all 500): capture c_1..c_6, J_1..J_5, record eigenspectrum,
   ρ, σ₁, Henrici dep_F(J)=√(‖J‖_F²−Σ|λ_i|²) (also normalized by ‖J‖_F), σ₁/ρ, number of
   expanding singular values (σ_i > 1), near-unit spectral mass Σ_{|λ_i|∈[0.9,1.1]} |λ_i|,
   top singular vectors.
2. **Branch test:** AUROC / average precision of expansion & non-normality invariants vs
   branch(t), pooled over problems (primary: pooled AUROC; secondary: within-problem rank).
3. **Anchor test:** AUROC / Spearman of slow-mode invariants vs I_t (ablation influence).
4. **Null controls (make-or-break):**
   a. Linear-chain instances: branch-signature distribution must collapse toward the
      non-branch distribution (no spurious "branch" spectral structure).
   b. M3/M4 matched Jacobians at identical positions: the branch-vs-non-branch spectral
      separation present in M2 must be ABSENT (or drastically reduced) in M3/M4.
   Paired per-problem comparisons (Wilcoxon), plus AUROC computed *in the control* as if it
   were the treatment.

### E1 — Causal validation (mandatory; the 2606.12689 discipline)
Interventions on M2 at spectrally-identified steps/directions:
- Perturb c_t by ε·d with d = top expansion direction (right singular vector v₁ of J_t) vs
  ε·d_rand (random unit vectors, magnitude-matched, ≥5 seeds); measure downstream |Δ margin|
  and answer flip rate as functions of ε.
- Ablate the dominant slow-mode subspace (project out top-k eigendirections) vs random
  k-dim subspaces; same metrics.
- Success criterion: spectral-direction effects exceed random-direction effects (paired
  Wilcoxon p < 0.01 and effect ratio ≥ 2× at matched ε across the tested ε grid), and the
  effect is largest at spectrally-flagged steps.

### E2 — Local vs routed dynamics (the 2602.08783 threat)
- Full influence Jacobians G_{t→t+k} = ∂c_{t+k}/∂c_t differentiated through ALL paths
  (including KV written at intermediate latent slots), vs the chained local product
  Π J. Routedness index R_{t,k} = ‖G_{t→t+k} − Π J‖_F / ‖G_{t→t+k}‖_F.
- If R is large (> 0.5 typical), local-Jacobian story is incomplete: escalate to orbit-level
  operator — pooled EDMD over (c_t, c_{t+1}) pairs (linear dictionary + delay embedding 2),
  Koopman spectrum, mode participation per step; re-run E0's AUROC with Koopman-mode
  predictors and full-influence predictors; report which level of description carries the
  anchor/branch signal.

### E3 — Scale / robustness
- Test + valid sets; ≥5 analysis seeds (bootstrap resampling of problems + random-direction
  seeds) for all CIs (95% percentile bootstrap, 10,000 resamples).
- M1 CoT accuracy context; curriculum-stage sweep (M2 checkpoints at epochs 10/20/30/40/best)
  for emergence of the spectral signature (secondary, budget permitting).
- Honest scope statement: GPT-2-124M, toy synthetic ProsQA, one seed of one architecture.

## 5. Pre-registered kill conditions (verbatim commitments)

- **K1 — NULL-CONTROL FAILURE.** The pause-token M3/M4 models (or linear-chain instances)
  show the SAME branch/expansion spectral structure as Coconut → the spectral objects are
  epiphenomenal (2606.12689 pre-refutes) → honest negative; project dead, or reframed as
  "spectral structure is a property of the architecture, not the reasoning."
  Operationalization: M2 branch AUROC's 95% CI overlaps the M3/M4 matched-control AUROC CI
  AND the paired branch-vs-non-branch spectral separation in M3/M4 is ≥ 50% of M2's.
- **K2 — CAUSAL FAILURE (E1).** Intervening on spectrally-identified directions does NOT
  produce the predicted downstream change beyond random-direction baselines
  (effect ratio < 2× or p ≥ 0.01) → lens is descriptive-only → demote to incremental;
  do not claim "explains."
- **K3 — ROUTED-EFFECT FAILURE (E2).** Neither local Jacobian NOR orbit-level Koopman
  captures the anchor/branch structure (all spectral predictors' AUROC 95% CIs include 0.5,
  or are dominated by trivial baselines like step index) because reasoning routes through
  attention in a way state-to-state operators miss → report the honest limitation of the
  dynamical-systems lens for routed computation.

**Trivial-baseline discipline:** every AUROC is compared against (i) step index t alone,
(ii) ‖c_t‖ alone, (iii) ‖c_{t+1} − c_t‖ alone. A spectral predictor that does not beat these
is reported as not beating them.

**Multiple-comparisons note:** the invariant families are fixed above; primary branch
predictor = σ₁(J_t) [transient expansion]; primary anchor predictor = near-unit spectral
mass. All others are reported as secondary/exploratory.

## 6. Deliverables

- `src/lrspec/` (harness, prosqa, spectra, causal, koopman) + `tests/`
- `experiments/exp0_gate.py .. exp3_robustness.py`, `make_figures.py`
- `results/*.json` committed
- `paper/main.tex` + auto-generated `paper/numbers.tex` + `paper/verify_regen.py`
- Recency re-sweep (live web) logged in `lit_notes.md` before submission.

## 7. What we will NOT claim

- Not claiming discovery of anchors/branch structure (established: 2602.08783, 2606.12689).
- Not claiming mechanism without E1 causal validation.
- Not claiming generality beyond GPT-2-124M/ProsQA/Coconut without evidence.
- A descriptive "eigenvalues along a trajectory" result will not be shipped as the headline.
