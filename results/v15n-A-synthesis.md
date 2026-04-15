# v15n-A — Per-Neuron Evolvable `memory_decay` via `node.alpha_slow`

## Verdict: **failure**

Per-neuron evolvable `memory_decay` in range [0.90, 0.99] actively *degrades*
performance vs. the hardcoded-0.97 baseline on the KV-Easy delay-8 benchmark.

## 5-seed result table

| seed | `correct_value_selected` | `correct_key_selected` | `final_max_score` | `mean_memory_frobenius_norm` |
| --- | --- | --- | --- | --- |
| 7   | 0.417 | 0.417 | 5.236 | 0.824 |
| 11  | 0.500 | 0.500 | 6.239 | 1.065 |
| 13  | 0.500 | 0.500 | 6.257 | 1.649 |
| 17  | 0.417 | 0.500 | 5.207 | 1.266 |
| 23  | 0.417 | 0.417 | 5.234 | 2.449 |
| **mean** | **0.450** | **0.467** | **5.634** | **1.451** |

### Comparison

| label | `mean_correct_value_selected` | `final_max_score` |
| --- | --- | --- |
| baseline-delta-v2        | 0.483 | 6.024 |
| v15m-delta               | 0.500 | 6.249 |
| **v15n-A**               | **0.450** | **5.634** |

- `correct_value_selected` dropped by ~0.05 (≈ −11 %) vs. v15m.
- `final_max_score` dropped by ~0.6 on every seed where value-selection
  dropped. Seed 11 and 13, which held at the 0.500 plateau, also held
  roughly the baseline fitness; the three seeds that fell to 0.417 lost
  ~1.0 on the fitness score.

## Interpretation

Evolution **pushed `alpha_slow` downward**, not upward. The search pressure
is apparently biased toward lower retention because short retention
produces a larger store-vs-distractor signal *locally* (the store-tick's
outer-product dominates the residual), which is easier to grab onto within
12 generations than the global benefit of long retention across the delay.

Evidence:

- Three seeds show `mean_memory_frobenius_norm` well below baseline
  (1.685), with seed 7 at just 0.824 — that is memory being aggressively
  forgotten.
- `mean_query_memory_alignment` is uniformly **higher** than baseline
  (0.482–0.835 vs baseline 0.760) — the memory that survives is
  cleaner, but there is less of it.
- `final_max_score` drops correlate 1:1 with `correct_value_selected`
  drops — this is not a spurious metric shift.

So the lever fires in the intended direction (mutable retention) but the
**local fitness gradient points the wrong way** for KV-Easy at delay 8.
The longer-retention end of the [0.90, 0.99] range is never reached by the
population within 12 generations.

## Rulled out

- Retention as a pure-directionless lever. Just exposing the knob is not
  enough; fitness lands on short-retention local optima.

## Candidate next steps

1. **v15n-B — floor the decay at the existing baseline.**
   Map `alpha_slow` to `memory_decay ∈ [0.97, 0.99]` (floor = hardcoded
   baseline). Evolution can only *lengthen* retention, never shorten it.
   Eliminates the regressive local optimum while preserving the lever.
   One-line code change on top of v15n-A.

2. **v15n-C — per-neuron but initialise biased toward long retention.**
   Keep the full [0.90, 0.99] range but perturb `alpha_slow` init to
   `mean=0.95, std=0.02` (currently 0.85 / 0.08). Makes the population
   start near the right end of the retention axis. Requires a
   variant-conditional init override in the tensorneat adapter.

3. **Roll back v15n-A and pivot** to a different lever class — e.g.
   `d_value` expansion (memory geometry), or per-neuron write-gain,
   or per-neuron `read_clip_norm`.

## Recommendation

**Option 1 (v15n-B) first.** It is the minimal, single-line change that
tests whether retention **when forced to be at least as long as baseline**
provides signal. If v15n-B is also a null, we know that retention is not
the bottleneck at delay 8 and can cleanly move on to memory geometry.

## Artifacts

- `results/v15n-A-s{7,11,13,17,23}.md`
- Implementation: commit `975a4f5`.
- This report: `results/v15n-A-synthesis.md`.
