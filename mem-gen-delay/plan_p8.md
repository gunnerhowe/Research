# P8 — Fingerprint-library re-emergence monitor ("Mythos"): a toy lab-safety experiment

Started 2026-07-20. QUESTION: can a library of capability fingerprints, captured from an
UNGUARDED specimen while each capability is present, be used to WATCH a guarded
continual-learning model and alarm BEFORE a removed capability's behavior returns — and
does that fingerprint MISS a capability that re-forms through a DIFFERENT circuit
(disguised return)? Designed by a 9-agent design panel (4 designs + 4 critiques +
synthesis); full synthesis archived in the session record. User forks locked 2026-07-20:
10 slots / honest effective-count framing; 3 layers / 8 heads; PILOT FIRST.

## Frozen design (reference; full-campaign predictions locked AFTER the pilot)

SUBSTRATE: TinyLM d_model=256, n_layers=3, n_heads=8, ctx=128, vocab≈128. Marker-triggered
"capability zoo": normal Markov filler with episodes spliced in exactly as the current
`rep` block overwrites seq (data path stays bit-identical on the default). Episode form:
`... M_k  <inputs>  SEP  <answer>`; opcode M_k is the SOLE trigger selecting the
deterministic map f_k. Loss = next-token CE over the whole sequence; per-skill behavioral
signal read only on the post-SEP span of M_k episodes.

10 SLOTS = 6 distinct mechanism families + 4 within-family specificity siblings (HONEST:
this is ~6 distinct circuits, not 10; effective count decided by the P1 separability
matrix, reported as measured):
- F1 content-match copy: M0 INDUCTION(+1), M1 SKIP-INDUCTION(+2, sibling)
- F2 positional k-back: M2 2-BACK(-2), M3 3-BACK(-3, sibling)
- F3 key-value recall: M4 KV-RECALL, M5 REV-RECALL(sibling)
- F4 bracket-depth: M6 DYCK (linear-probe subspace mechanism)
- F5 set-membership: M7 MEMBERSHIP
- F6 comparison: M8 MAX, M9 MIN(sibling)
GUARDED (5, mechanism-diverse): M0, M2, M4, M6, M7. RETAINED (must never alarm): M1, M3,
M5, M8, M9.

FINGERPRINT CAPTURE (from the FROZEN converged specimen — forward-pass / one-shot fit /
ablation only, NO training trajectory; on a fixed per-skill probe bank P_k, N=256 held-out
episodes + matched foils):
- (A) ATTENTION-ALIGNMENT (M0,M1,M2,M3,M4,M5,M7): the run_probes gather idiom with the
  alignment index swapped per rule; fingerprint = home head (argmax layer,head), its mass
  m0, the per-head mass vector v0, and (M4 vs raw copy) the OV write direction. Sibling
  fingerprint = CONJUNCTION of family read AND its distinguishing offset/role.
- (B) LINEAR-PROBE SUBSPACE (M6 Dyck; readout legs of M7/M8/M9): rank-r ridge/LEACE
  subspace U_k of the block[1] interior residual predicting the skill latent. The
  specimen-IDIOSYNCRATIC capture used for the disguise test.
- (C) SAE-lite CORROBORATOR (dict 2048, top-k 32; causal-atom filtered): REPORTED, never
  gates the sealed alarm.
- (D) CAUSAL SITE: per-head mean-ablation sweep → lesion/relocation reference.

WATCH (inline every eval on the FIXED P_k — NEVER the streaming batch; this firewall is
what exposed the ESN data-mix reader). Per skill: frozen-home-head mass (relocation-
FRAGILE), max-over-heads mass (relocation-ROBUST), cosine(v_t,v0); for M6 frozen-subspace
energy + a re-fit-R2 DIAGNOSTIC. All vs the specimen's ABSOLUTE m0/U_k, never a running max.
COMPOSED ANCHOR (the shipped 0/33-FA rule): alarm iff [structural proximity ≥ tau_struct]
AND [behavioral ramp ≥ 0.5·tau_k], sustain 2. tau_struct ABSOLUTE = guarded+matched-blocked
baseline proximity + 5σ of its fluctuation; set BEFORE any positive; never %-of-max, never
calibrated on positives. Bare-structural and gate-only alarms reported as pre-registered
ablations (bare expected to false-alarm on siblings/N2 → shows the gate is load-bearing).

MANUFACTURED NEGATIVES (matched-data capability-blocked; the ESN lesson institutionalized):
N1 omission (retained only); N2 MARKER-SCRAMBLE (guarded marker present at same freq +
input dist but targets scrambled — the acid test / trap-one analog); N3 cross-family (watch
guarded skill while re-teaching an adjacent RETAINED sibling); N4 architectural (onelayer);
N5 dist-shift (new filler lang_seed, no guarded episodes).
DISGUISE TIERS: RELOCATION (burn home head → re-forms elsewhere; frozen-home misses,
max-over-heads catches); IDIOSYNCRATIC-SUBSPACE (M6, LEACE-erase U_k → return orthogonal,
honest miss); UNCATALOGUED-MECHANISM (M0 headline: copy re-routed through a path no
attention fingerprint covers). Every disguise miss is scored ONLY after the circuit change
is verified (home-head argmax moved / low cosine / low subspace overlap), with a matched
no-reteach VALIDITY CONTROL (0 event AND 0 alarm) proving a miss is a genuine blind spot.

EVENT rule (frozen, all skills): absolute metric ≥ tau_k on 2 consecutive evals (never
%-of-max). tau_k pinned once from the specimen, strictly between guarded-baseline and
specimen value; binary skills tau=0.85–0.90. Behavioral ramp = ≥ 0.5·tau_k.

## PHASE 0 PILOT (prereg; EXPLORATORY / de-risking — gates are GO-NO-GO, not science)
Skills: M0 induction, M1 skip (retained sibling of M0), M2 2-back, M3 3-back (retained
sibling of M2), M6 Dyck. 3 mechanisms across 5 slots. Guard {M0, M2, M6}; retain {M1, M3}.
Specimen: 1 primary seed (+1 replication if the first is admissible). Streams: P-FAITHFUL;
D-RELOC on M0 and M2; D-IDIO on M6; D-UNCAT on M0 built BOTH ways (Fork 3: (A) fixed
positional-offset/MLP shortcut, (B) all-attention-heads-frozen so an MLP/residual route
must carry it); negatives N1, N2. ~15–20 runs, ~5–8 GPU-h. Sealed pilot scorer
analysis/score_zoo_pilot.py (refuses partial/existing).

GO-NO-GO GATES (a fired gate reshapes the full campaign BEFORE it spends compute):
- G-P1 CONVERGENCE: induction (M0) reaches copy_adv ≥ 2.0 at 3L/8H within budget, AND
  ≥4/5 pilot skills fire their event. (Confirms the depth change didn't break the
  validated substrate.)
- G-P2 SEPARABILITY (resolves Fork 1 guard partition): the 5×5 CONJUNCTION-fingerprint
  confusion matrix is diagonally dominant (diagonal > 3× max off-diagonal) on ≥3/5 slots,
  AND the M0/M1 and M2/M3 sibling pairs are individually separated. If a pair fails →
  full campaign uses the PAIRED-guard partition for that family; if it passes → the
  straddle partition stands.
- G-P3 FIREWALL (the ESN lesson, as a pilot gate): the composed anchor gives 0 false
  alarms on N2 marker-scramble for the guarded skills, WHILE the bare-structural alarm
  fires on N2 (confirming the behavioral gate is load-bearing, not decorative).
- G-P4 DISGUISE CONSTRUCTIBILITY (resolves Fork 3): ≥1 disguise is circuit-verified
  (D-RELOC home-head argmax moved, and/or D-IDIO return orthogonal to U_k, and/or D-UNCAT
  library-bypassed) AND re-teaches to its behavioral event. Keep whichever D-UNCAT
  construction (A or B) both converges and is verified library-bypassing.
- K0 GUARD VALIDITY (manipulation check, outcomes never consulted; fix-and-relaunch
  allowed): guarded events silent + guarded fingerprints below band on a marker-specific
  read; retained events intact. If a guard zeros accuracy but leaves the fingerprint up,
  strengthen the guard before any watch scoring.

After the pilot: lock the full-campaign predictions P1–P6 + kills (K1 separability,
K-FA marker-scramble, K-DISGUISE-A/B, K6 disguise-vs-blocked, K-GATE gate-dominance,
P-REFIT fallback, global lead bar ≥ max(200, 0.25·D)) in a plan_p8.md amendment, then
Phases 1–3 + sealed score.

## Build plan (files)
- src/train_lm.py: +--n_layers arg; generalize Block scaffold from head-0-only to
  --scaffold_head; expose/hook the block[1] interior residual. (Default path stays
  bit-identical; guard with the existing K-C2b-style smoke.)
- src/data_zoo.py: marker-episode generators (10 maps) + per-skill probe banks P_k + foils
  + stream samplers (mixture, P-FAITHFUL, D-RELOC, D-IDIO, D-UNCAT-A/B, N1–N5, validity).
- src/fingerprint_zoo.py: capture_attention/subspace/sae + inline proximity(model,skill,fp).
- src/guard_zoo.py: G1 data-ablation, G2 LEACE-erase, causal-site sweep + head-burn lever.
- scripts/run_zoo_pilot.py (idempotent) + analysis/score_zoo_pilot.py (sealed one-shot).

## PILOT AMENDMENT (2026-07-20, disclosed before capture/watch scoring)
Pilot convergence finding (specimen training, seed 1, 3L/8H d256): the copy family
(M0 induction, M1 skip) and M6 Dyck form; the POSITIONAL k-back family (M2/M3) does NOT
hold in-mixture even up-weighted 3x (M2 0.09, M3 0.27 at 30k), and forcing it stole
capacity from induction (M0 0.73 equal-weight -> 0.48 up-weighted). Diagnosed as
capacity contention + k-back's marker-conditional-offset awkwardness (2-back vs 3-back
interfere). PILOT NARROWED to M0, M1, M6 (guard M0, M6; retain M1 as M0's sibling). This
still exercises every machinery piece: attention-alignment fingerprint (M0,M1) +
separability of the M0/M1 sibling pair (Fork 1, copy family); subspace fingerprint (M6);
D-RELOC + D-UNCAT (M0) and D-IDIO (M6); N1/N2/N3 negatives. The k-back partition question
and the "how many mechanisms fit one model" question defer to the full campaign, where
they become a Fork-4 decision (leaner zoo vs bigger model) informed by this pilot. This
is exactly the pre-registered role of the pilot: reshape scope before spending fleet
compute. Substrate-capacity is now itself a reportable pilot result.

## PILOT RESULT (2026-07-22) — ran end-to-end; did NOT cleanly demonstrate fingerprint
## early-warning; surfaced two fixable issues + one candidate finding. Honest interim.
Specimen {M0 induction 0.855, M6 depth 0.943} trained (after fixing the loss-dilution bug:
focus loss on answer spans; and the mechanism-interference finding: multiple attention-
matching skills do not coexist at this scale, only one attention skill + one counting skill).
Fingerprints captured clean: M0 attn home (L4,H0) m0=0.701; M6 subspace layer3 r2=1.000.
Guard by data-ablation, then 8 watch streams with inline proximity.

FINDINGS:
1. GUARD INCOMPLETE (K0 partial): data-ablation (omit the skill, continue-train 8k @ 5e-4)
   only PARTIALLY removed skills — M0 0.855->0.520, M6 0.943->0.390 (not driven to chance).
   So the watch starts from a half-present skill; re-emergence is fast (behavior snaps back
   by ~step 1000 in faithful) with no slow ramp to forecast. A complete guard (gradient-
   ascent unlearning or much longer ablation) is required for a clean re-emergence test.
2. STRUCTURAL FINGERPRINTS DECOUPLE FROM BEHAVIOR (candidate finding, needs a clean re-run):
   - M0 attention max_mass on the M0 probe bank: faithful ~0.65, N1 omission ~0.60,
     N2 scramble ~0.15. The attention alignment PERSISTS through omission-decay (N1: acc
     -> 0 while max_mass stays ~0.60), so a STRUCTURAL-ONLY alarm cannot distinguish a
     genuine return (faithful) from mere omission (N1) — both ~0.6. Only the behavioral gate
     separates them (faithful acc 0.85 vs N1 acc 0). This RE-CONFIRMS the composed-anchor
     thesis (behavioral gate is load-bearing) in the re-emergence setting, and is a
     safety-relevant caution for the "capture fingerprint, watch for its return" idea: the
     fingerprint can be present without the capability. BUT the "fingerprint LEADS behavior"
     claim FAILS here — the structure is persistently high, not an advance signal.
   - M6 subspace ENERGY metric is FLAT (~0.15-0.20) regardless of depth accuracy (0.39->0.93)
     -> the energy readout is the WRONG proximity metric. The right metric is frozen-subspace
     DECODE quality (r2 of U decoding depth), which was not logged; requires a re-run.
3. N2 marker-scramble: composed anchor would stay silent (behavior low AND structure low
   for M0); the firewall holds for the composed rule (bare structural is the false-alarmer).
4. DISGUISE (D-RELOC, D-IDIO): behavior returns in both, but because the guard was
   incomplete and the proximity metrics are flawed, the disguise-miss question is NOT
   cleanly answerable from this run — deferred to a clean re-run.

VERDICT: pilot RAN (full pipeline works) but is INCONCLUSIVE on the headline (fingerprint
early-warning + disguise kill) due to (1) incomplete guard and (2) proximity-metric design.
A clean re-run needs: complete guard (gradient-ascent unlearn to chance), subspace proximity
= decode-r2 (not energy), and a slow re-emergence (low reteach rate) so there is lead to
measure. The structural-behavioral DECOUPLING is the one real candidate result and it
supports the program's composed-anchor thesis. NEXT STEP IS THE USER'S CALL (effort already
large): iterate to a clean re-run, or bank the decoupling caution and stop.

## PILOT CLEAN RE-RUN VERDICT (2026-07-22) [analysis/out8/zoo_pilot_scored.json, sealed]
All three fixes verified in-run: guard-by-scramble drove behavior AND structure to chance
(M0 acc 0.027 / struct 0.152; M6 acc 0.000) -> K0 PASS both. Subspace metric = frozen-w
decode-r2 (specimen 1.000; erased 0.000 while behavior survives).

RESULTS (composed anchor = struct >= negbase+0.10 AND acc >= 0.5*tau_k, sustain 2):
1. G-P3 FIREWALL: PASS — composed alarm SILENT on all 4 capability-blocked streams
   (N1 omission + N2 marker-scramble, both skills). Certified 0 FA on this fleet.
2. FAITHFUL LEAD: **FAILED — re-emergence is too fast to forecast.** M0: event step 400,
   alarm step 400, lead 0 (behavior back above tau by the FIRST/second eval even at 0.4x
   reteach weight). M6: event 400, alarm 1800, lead -1400 (the frozen fingerprint LAGS:
   behavior snaps back while the specimen subspace re-forms slowly). Re-acquisition from a
   truly guarded state is near-instantaneous relative to any feasible eval cadence —
   relearning >> first learning (first learning took 500-1500 steps at 2.5x the data
   rate). NO fingerprint early-warning exists in this regime; the monitor is a DETECTOR
   (with certified FA), not a FORECASTER, for RE-emergence. The forecastable warning phase
   demonstrated in P5-P7 belongs to FIRST emergence.
3. DISGUISE (the pre-registered headline):
   - M6 SUBSPACE-ERASE: **KILL FIRES CLEANLY.** Behavior fully returns (0.948) while the
     frozen-subspace decode-r2 stays at 0.002 — the composed alarm NEVER fires. With the
     firewall passing (negatives silent, faithful alarm does fire at 1800), this is a
     verified blind spot, not a blind monitor: a specimen-captured fingerprint MISSES a
     capability that re-forms through an orthogonal representation. "Same output,
     different process" (the d-bar finding) operationalized in the monitoring direction.
   - M0 RELOCATION: partial relocation verified (home_mass 0.335 vs 0.654 faithful) and
     MAX-OVER-HEADS still catches (0.666, alarm fires) — the cheap disguise is defeated
     by the broad read, as designed.
4. Fast-relearning is itself the safety-relevant hazard surfaced: a guarded (unlearned-to-
   chance) capability returns to full strength within ~400 steps of re-exposure at 0.15-0.4x
   data rate — behavioral suppression left rapid re-acquisition intact.

STANDING: the Mythos pilot DELIVERED its pre-registered adjudication: fingerprint
libraries with a composed anchor give certified-FA DETECTION of capability returns;
relocation is catchable by broad reads; orthogonal re-formation defeats the captured
fingerprint (K-DISGUISE-B analog FIRED); and re-emergence at toy scale has NO warning
phase to forecast. Scope: 1 seed per stream, 2 mechanisms, toy scale — pilot-grade, not
fleet-certified. Full-campaign decision (multi-seed fleets, refit-on-canary fallback,
slower-re-emergence protocols) rests with the user.

## P8-R2 PREREG (2026-07-22) — "Unlearning removes the behavior, not the scaffold"
## Frozen BEFORE any R2 code exists. Substrate: the VALIDATED 2L fleet world
## (src/train_lm.py, bigram lang_seed 777, p_rep 0.75, d256/4H/ctx256, lr 1e-3,
## event = copy_adv >= 2.0 nats sustain 2; first-emergence T0 = 6,300 median, n=30).

MOTIVATION: P8 pilot found near-instant re-emergence at 6L — but 6L first-learning is also
fast (resolution-limited comparison, disclosed). The 6L guarded model retained its
prev-token scaffold at 0.49 (specimen 0.71) while behavior sat at chance — suggesting
fast relearn = P7's scaffold-supplied regime (which emerges at ~3,600 vs 6,300). R2 tests
this ON THE SUBSTRATE WHERE TIMING IS UNDERSTOOD, plus the compensating-law bet: a
disguised return must REBUILD the scaffold and should therefore be slow — restoring
warning time exactly where the fingerprint is blind.

DESIGN (runs/grid8r2/; n=3 seeds per arm 501-503; pilot-grade, disclosed):
1. SPECIMENS: rep condition, 16k steps, NEW --save_ckpt flag (default-off; bit-identity
   guarded by a K-C2b-style smoke vs grid6r2/rep_s1 before the fleet).
2. GUARD (active unlearning): continue-train specimen (--init_from, new flag) on NEW
   condition "shufrep" — repeats present at the same rate/offsets but the copied span is
   per-sequence PERMUTED, so match-then-copy actively mispredicts. 8k steps. GUARD
   VALIDITY (manipulation check, outcomes never consulted): copy_adv <= 0.25 AND
   indist_adv <= 0.05 over the last 5 evals; else re-tune guard before watch. Secondary
   comparison arm: norep guard (omission), reported descriptively.
3. RETEACH-FAITHFUL: continue guarded ckpt on rep data (matched rate), 10k budget.
4. RETEACH-BURNED (the disguise): same, but the guarded model's layer-0 argmax prevtok
   head is burned with the P7 sink bias (beta 8) — the surviving scaffold is denied, so
   the return must re-grow a precursor elsewhere. --log_heads on.
5. NEGATIVES: guard-hold x3 (continue shufrep, no re-teach) — anchor must stay silent.

FROZEN MEASURES: t_reevent = copy_adv >= 2.0 sustain 2 (as everywhere). Composed anchor =
first eval with prevtok_by_layer[0] >= 0.10 AND indist_adv >= 0.10 (the campaign's frozen
conjunction). Lead = t_reevent - t_anchor. Scaffold read = layer-0 max-over-heads prevtok.

PREDICTIONS / KILLS (frozen now):
- P-R2a (scaffold survives active unlearning): guard-end prevtok >= 0.5 in >=2/3 shufrep
  guards while behavior is at chance. K-R2a fires if prevtok < 0.10 in >=2/3 (premise
  dead; proceed but the P7 link is cut).
- P-R2b (fast relearn on the slow substrate): faithful median t_reevent <= 3,150 (0.5*T0).
  K-R2b fires if >= 5,040 (0.8*T0): relearn NOT fast at 2L -> the 6L instant-relearn was a
  substrate-speed artifact; "unlearning is effective at slow scale" = the honest negative.
- P-R2c (THE COMPENSATING LAW, headline): burned median t_reevent >= 2.0x faithful median
  AND burned median >= 3,800 (0.6*T0). K-R2c fires if burned <= 1.25x faithful: disguise
  costs nothing -> uncovered-cell negative for fingerprint monitoring.
- P-R2d (warning returns on the disguised path): burned-arm composed anchor fires with
  median lead >= 300 steps pre-reevent, AND 0 composed alarms on the 3 guard-hold
  negatives. (In guarded models prevtok stays high, so the conjunction reduces to the
  indist gate — the FA control is exactly that gate on the negatives.)
- Faithful-arm lead reported as measured (no bet; expected small — the scaffold-supplied
  regime has little warning, per P7).
Scoring: sealed one-shot analysis/score_p8r2.py after ALL 15 summaries exist; refuses on
partial fleet or existing scorefile. Disclosures: T0 reused from grid6r2; reteach lr 1e-3
(matched); beta=8 burn; n=3/arm.
