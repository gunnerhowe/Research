# Phase 3 working notes (for paper #3 assembly)

## Status
- Kill-check: survived (8 angles). Occupants to cite+disambiguate: DECT 2310.07630
  (complexes), SECT/WECT/ERT (shape statistics), Minkowski Image Loss 2604.11422
  (pixel grids), STITCH 2412.18696 + CAD-2022 PH (complexes), NeurCADRecon 2404.13420
  (pointwise curvature geometry), GKF/Adler-Taylor (probability theory).
- GATE V passed: chi-hat accuracy 1-3% at N=2e5, 3 seeds. Exact numbers:
  one bump 0.998±0.002/0.997±0.012/1.021±0.007 (want 1); two bumps 2.017±0.018/
  2.005±0.016/1.006±0.026 (want 2,2,1); annulus 0.018±0.015/0.019±0.040/0.011±0.033
  (want 0); 3D ball 1.012±0.019/0.987±0.014 (want 1); solid torus 0.016±0.042/
  0.026±0.071 (want 0). Tests: tests/test_phase3.py (7 pass).
- GATE S passed via three mechanism findings (all paper-section-worthy):
  1. LEVEL LADDER: chi is a.e. flat in theta; smoothed estimator gradients live
     only within ~eps of transitions; dense levels (spacing ~eps) spanning the
     value range = extinction-gradient ratchet. (Failed run receipt: cap at
     levels<=0.4 with blob peak 0.7 -> cap_loss=0 forever.)
  2. C^2 BACKBONE: ReLU-PE nets hide curvature in kinks invisible to MC sampling
     (chi-hat underreads; cap never fired). SIREN/tanh mandatory.
  3. VECTOR ANTI-GOODHART: chi alone gamed by debris (4 comps + 3 holes = chi 1);
     M1 (perimeter) budget prices out cancellation -> honest deletion.
- exp9 (3 seeds) FINAL: anchor_only 0/3 repaired; smooth_null 2/3 but fidelity
  0.053-0.062 in its successes (~11-17x (per-seed: 17.0, 11.5) worse); chi_only 0/3 (Goodhart); vector_cap
  3/3, spur<=0.110, mse 0.003-0.005. JSON: results/phase3/exp9.json.
- exp10 RUNNING (GPU, tag _full, 3000 iters, ph_loss at 1500): smoke numbers:
  topo cost ours 2.8-3.1 ms/it vs PH (gudhi 48^3) 926-1072 ms/it => ~340x.
  Report matched-iterations AND matched-wallclock. Eval: voxel_betti at 96^3
  (verified exact on analytic sphere/torus/genus2), Chamfer vs clean surface.
- SDF chi must use MATCH mode (handles push chi down, components up).

## Paper #3 skeleton ("Differentiable Minkowski Functionals for Neural Fields")
1. Intro: topology control for neural fields without persistence machinery;
   the PH literature's own gap statement ("Betti numbers not easily used...").
2. Related: the six-family disambiguation table.
3. Method: M0/M1/M2 estimators (co-area + Gauss-Bonnet), EulerProfileLoss
   (match/cap), design rules = ladder + C^2 + vector. Formulas in
   src/kacrice/minkowski.py docstring (sign conventions verified on disc).
4. Validation (GATE V table/figure).
5. exp9 controlled quartet (table above).
6. exp10 SDF benchmark vs PH + cost axis (pending).
7. Limitations: boundary terms dodged (fields decay at domain edge); levels
   fixed; chi conflates features (vector mitigates); PH remains stronger
   per-iteration signal.
- Env pins: Python 3.13.6, torch 2.7.1, numpy 2.4.4, scipy 1.17.0, gudhi 3.13.0,
  skimage 0.26.0. Author: Gunner Levi Howe. Same repro-appendix pattern as papers 1-2.
- Figures: make_figures_phase3.py TO BE WRITTEN (paper3/figs): validation bar/profile
  fig, exp9 quartet fig (2x4 fields+profiles), exp10 table + cost bar, mechanism
  figure (ladder ratchet schematic from the three GATE S receipts in
  results/p3_gate_s*.log).

## exp10 FULL RESULT (2026-07-04): ALL 36 RUNS FAIL topology (incl. PH baseline)
- vector_match at 3000 iters: sub-sampling debris explosion (~75k voxel-scale
  cavities; MC spacing 0.06 vs voxel 0.02). MECHANISM FINDING #4: MC topology
  estimators are blind below sampling density; network bandwidth must be matched
  to sampling (SIREN omega=15 in 3D too high for 8k points). 2D exp9 worked
  because sampling ~ eval resolution.
- PH baseline: 0/9 exact Betti too (closest qualitatively; best Chamfer 0.09-0.25
  -- its grid loss is also a global regularizer). Benchmark as designed (2k noisy
  pts + 5% outliers, 3k iters) is too hard for every method.
- Cost axis STANDS: ours 2.7-4.0 ms/it vs PH 670-977 ms/it (~250x, GPU net + CPU gudhi).
- NEXT: exp10b = bandwidth-matched regime applied to ALL configs equally:
  SIREN omega=8, eikonal weight 1.0 (was 0.1), n_dom 16k (was 8k), iters 3000.
  If repairs emerge -> paper reports both regimes (hard: all fail + mechanism;
  matched: comparison valid). If not -> kill condition #4 fires honestly: paper
  = 2D training success + 3D measurement validation + 3D training failure w/
  mechanism (sampling-resolution gap). Either way publishable, honest.

## exp10b RESULT (omega=8, eik=1.0, n_dom=16k, tag _bw): KILL #4 FIRES, with mechanism
- PH: 4/9 exact (sphere 3/3, torus 1/3, genus2 0/3 near-misses) -> matched regime IS
  solvable; PH is the honest incumbent line.
- vector_match: 7/9 debris-exploded (~75k cavities), 2/9 close-not-correct.
  REFINED MECHANISM (#4): debris is ADVERSARIAL, not bandwidth: gradient descent
  tunnels into the estimator null space below sampling density; M1 guard equally
  blind there (2D worked because sampling ~ eval scale). eikonal_only shows only
  5-16 comps / 57-210 handles -> debris is topology-LOSS-INDUCED, not noise.
- PAPER 3 FINAL SHAPE: validated estimator (GATE V 1-3%) + 2D training success
  (exp9 3/3 vector) + design rules (ladder, C2, vector, sampling-scale) + 3D
  adversarial failure precisely diagnosed + PH comparison (both regimes, both
  iteration protocols) + cost axis (~250x) + the honest boundary statement:
  MC topology losses are trustworthy iff sampling density covers the feature/eval
  scale (cubic cost in 3D). Title candidate: "Differentiable Minkowski Functionals
  for Neural Fields: Validation, Design Rules, and an Adversarial Failure Mode".
- Scaling receipt running (tag _scal): sphere/vector_match, 1 seed,
  n_dom in {8k, 32k, 128k} -> debris count vs sampling density curve.

## SCALING PROBE FINAL (tag _scal32, seed0, n_dom=32k): DATA COLLECTION COMPLETE
- vector_match debris at n_dom 8k/16k/32k (spacing .08/.063/.05): ~75k/~75k/51-75k
  features -- INVARIANT to 4x sampling. Closing the null window to voxel scale (.02)
  needs N~1e6 (cubic) => erases the ~250x cost advantage. That IS the measured
  boundary statement; 128k run SKIPPED as uninformative (justify in paper).
- PH at 32k unchanged (near-misses), eikonal/smooth cleaner with more samples but
  never correct.
- ASSEMBLY NOW: make_figures_phase3.py + paper3/main.tex per skeleton above.
  All JSONs final: results/phase3/{exp9,exp10_full,exp10_bw,exp10_scal32}.json
  (glob exp9*.json / exp10*.json; dedup by (shape,config,seed,iters,omega,n_dom)).
  Regime labels: _full=omega15/eik0.1/8k, _bw=omega8/eik1.0/16k, _scal32=32k.

