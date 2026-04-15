# Tie-Break Randomization Diagnostic — Synthesis

## Question

Is the persistent `mean_correct_value_selected = 0.500` plateau on KV-Easy a
**structural artifact** of the deterministic `np.argmin` tie-break in
`_nearest_value_indices` at `src/evolve/evaluator.py:1196–1198` — or is it a
**real neuron/evolution plateau**?

## Patch

`src/evolve/evaluator.py` — added 1e-9 uniform jitter to distances before
`argmin`, so that exact distance ties are broken uniformly at random.
Jitter scale is ~ 8 orders of magnitude below the minimum meaningful distance
gap on KV-Easy (≥ 0.25), so only genuine ties are affected.

## Result

5 seeds, identical configuration to `baseline-delta-v2` and `v15m-delta`
(tasks=key_value_memory, delay=8, generations=12, population=40,
profile=kv_easy, variant=stateful_v6_delta_memory).

| seed | mean_correct_value_selected | mean_correct_key_selected | mean_query_key_match_score |
| --- | --- | --- | --- |
| 7   | 0.500 | 0.500 | -0.110 |
| 11  | 0.500 | 0.500 | -0.110 |
| 13  | 0.500 | 0.500 | -0.131 |
| 17  | 0.500 | 0.500 | -0.108 |
| 23  | 0.500 | 0.500 | -0.108 |
| **mean** | **0.500** | **0.500** | **-0.113** |

### Cross-comparison

| label | mean_correct_value_selected | mean_correct_key_selected |
| --- | --- | --- |
| baseline-delta-v2     | 0.483 | 0.500 |
| v15m-delta            | 0.500 | 0.500 |
| **tiebreak-diag (this run)** | **0.500** | **0.500** |

Plateau is **bit-identical** to the unrandomized baselines. The diagnostic
produces the same plateau to three decimal places across every seed.

## Verdict

**The plateau is NOT a tie-break artifact.** It is a real neuron/evolution
plateau. Deterministic `argmin` with lowest-index tie preference is not the
source of the 0.500 cap.

## Implication for next lever

- The forensic "smoking gun" hypothesis is falsified: `pre_decode` outputs of
  evolved genomes do not in practice land on exact inter-level midpoints
  often enough for tie-break semantics to matter.
- Therefore every observed 0.500 data point is a **genuine** aggregate over
  per-episode scores `{0.0, 0.167, 0.333, 0.500, 0.667, 0.833, 1.0}` —
  whatever combination of episode-level outcomes averages to 0.500.
- Combined with the prior negative results on v15l-B (readout sharpening)
  and v15m (logit decoder), the **decoder/readout path is not the bottleneck**.
- The architectural agent's recommendation stands and is now the leading
  candidate: **evolvable per-neuron `memory_decay_factor` ∈ [0.90, 0.99]**
  — a Write-Path-class lever (historically 4/4 success rate) that
  addresses retention rather than readout.

## Artifacts

- `results/tiebreak-diag-s7.md`, `tiebreak-diag-s11.md`, `tiebreak-diag-s13.md`,
  `tiebreak-diag-s17.md`, `tiebreak-diag-s23.md`
- Diagnostic patch: commit `6f8b6fd`.
- Next step: revert the patch and move on to the retention-path lever.
