# P5 lit kill-test: forecasting emergence timing from internal signals

Question: can internal (representational) probes FORECAST delayed generalization / capability
emergence before behavior moves — and do they beat cheap scalar training signals?

Every adjacent work checked 2026-07-14. Verdict per item: what it establishes, what it KILLS
in our design, what survives.

## Direct predecessors (must cite, must beat or absorb as baselines)

1. **Notsawo et al., "Predicting Grokking Long Before it Happens" (arXiv:2306.13253, ICLR'24)**
   Predicts WHETHER grokking will occur from low-frequency oscillations in the first few
   epochs' loss curve (spectral signature of the learning curve). Scalar signal, binary
   whether-classification, no time-to-event forecast, no calibration, no false-alarm
   accounting, no probe-vs-scalar comparison.
   KILLS: "we are the first to predict grokking." Nobody may claim that.
   SURVIVES: when-forecasting (calibrated time-to-event); their oscillation feature becomes
   one of OUR scalar baselines (osc(train_ce)).

2. **"From Density Matrices to Phase Transitions in Deep Learning: Spectral Early Warnings
   and Interpretability" (arXiv:2603.29805, Mar 2026)** — CLOSEST WORK.
   2-datapoint reduced density matrix; spectral heat capacity claimed to give "early warning"
   of second-order transitions via critical slowing down; validated on deep linear, induction
   heads, grokking, emergent misalignment. Per abstract: no quantified lead time, no
   calibration, no false-alarm rate, no scalar-baseline comparison — early warning as a
   qualitative indicator, not a scored forecast.
   KILLS: "internal spectral signals show precursors of transitions" as a novelty claim.
   SURVIVES: forecasting-as-forecasting (lead time at controlled false-alarm rate,
   prospective validation); the baseline horse race; interventional-shift robustness. Their
   2RDM heat capacity is a candidate signal we can ADD to the battery later (R5).

3. **Aoyama & Wilcox, "Predicting the Formation of Induction Heads" (arXiv:2511.16893)**
   Closed-form prediction of IH formation timing from PRE-TRAINING quantities: batch size,
   context size, bigram repetition statistics of the data. Not within-run, not from internal
   state.
   KILLS: our planned R-induction headline ("predict induction head formation"). Downgraded
   to stretch rung with narrowed delta: config-based prediction cannot explain SEED-level
   timing variance at fixed config; can within-run internal precursors (previous-token-head
   strength) forecast per-seed residuals? Only run if R1-R4 land.

4. **Hoogland et al., "Loss Landscape Degeneracy and Stagewise Development in Transformers"
   (arXiv:2402.02364) + devinterp/LLC program**
   LLC segments training into developmental stages, incl. stages hidden from loss.
   Retrospective/concurrent stage DETECTION; no advance forecasting, no lead time, no false
   alarms. KILLS: "hidden progress is visible in geometry" as novelty. SURVIVES: everything
   forecasting-specific. LLC is another candidate battery signal (R5, compute permitting).

5. **Thilak et al., "The Slingshot Mechanism" (arXiv:2206.04817)**
   Last-layer norm spikes/cycles co-occur with grokking onset under adaptive optimizers.
   Concurrent marker, not advance forecast. Becomes a scalar-baseline feature candidate.

6. **Scale-axis emergence prediction — different axis, cite and distinguish:**
   PassUntil (arXiv:2310.03262) predicts task performance ACROSS MODEL SCALE with
   high-resolution eval; Snell et al. (arXiv:2411.16035) shift emergence points via
   finetuning to fit "emergence laws" across scale. Both predict across the scaling axis
   before/without the target run; we forecast WITHIN a training run over time. Complementary.

7. **Context/premise (cite):** Nanda et al. progress measures (retrodictive mechanistic
   progress metrics); Barak et al. hidden progress; Schaeffer et al. "mirage" (metric-induced
   discontinuity — supports internal-continuity premise); Olsson et al. induction-head phase
   change; International AI Safety Report 2026 states capability-emergence forecastability
   is an OPEN question — the gap we target, in writing, in 2026.
   In-house: the weight-norm delay law (arXiv:2606.13753) makes norm the strongest known
   scalar clock in grokking regimes — the baseline any probe must beat; our Papers 1/3/4:
   probes lead behavior (premise), and interventions (priors/clamps) shift t_gen up to ~100x
   while DECOUPLING the norm clock from the outcome (supcon arms grok at frozen norm ~92;
   label prior at growing norm ~104; MNIST baseline probe TRAILS behavior).

## The surviving lane (novel portions)

N1. **Forecasting discipline where the field has indicators**: calibrated time-to-event
    forecasts with quantified LEAD TIME at CONTROLLED FALSE-ALARM RATE, scored on held-out
    seeds. Nobody in items 1-5 scores a forecast.
N2. **The horse race**: representational probes vs scalar training signals (norm, train/test
    CE, logit stats, Notsawo oscillation feature) under identical protocol. "Does
    interpretability buy warning time?" is unanswered in print.
N3. **Interventional-shift robustness**: train forecasters on control-family runs, test on
    intervention arms where the norm clock is decoupled by construction (our grids are the
    only corpus with this property). Pre-registered prediction: scalar forecasters break,
    content probes survive. If probes ALSO break -> honest negative for the robustness claim
    of the early-warning program.
N4. **Negative-control accounting**: never-generalizing runs (wrong-structure, stalls,
    censored pins) as the false-alarm test set. Early warning without false-alarm statistics
    is not a warning system.
N5. **Prospective pre-registered validation**: frozen forecaster + committed predictions
    BEFORE outcome runs execute (commit-hash discipline). First in this literature to our
    knowledge.
N6. **Corpus**: 466 runs / ~210k synchronized probe records across 5 condition families,
    2 domains, with 100x causal spread in t_gen. Released as a forecasting benchmark.

## Re-derivation risks accepted and mitigated
- If a 2026 paper we missed already scores lead time vs scalar baselines: closest candidates
  checked above do not; International AI Safety Report 2026 treats it as open. Residual risk
  acknowledged; a final targeted search precedes submission.
- Our probes are task-aware (cos_gap/fourier use labels/structure). Framed honestly:
  capability-specific probes = the realistic monitoring setting (you know which capability
  you watch); label-free battery is R5.
