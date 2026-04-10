# V14v vs V14u / V14s-B / V14r / V14t / V14h Comparison

## Focus Metrics (key_value_memory, delay=8, seed=7)

| version | query_key_match_score | correct_key_selected | correct_value_selected | store_vs_distractor_beta_gap | query_value_read_strength | key_query_cosine_mean | key_query_cosine_at_query | key_variance_mean | query_variance_mean | mean_memory_frobenius_norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v14v | -0.109 | 0.500 | 0.500 | 0.012 | 0.149 | 0.930 | 0.937 | 0.002 | 0.001 | 1.027 |
| v14u | -0.114 | 0.500 | 0.500 | 0.010 | 0.153 | n/a | n/a | n/a | n/a | 1.392 |
| v14s-b | -0.109 | 0.583 | 0.500 | 0.191 | 0.211 | n/a | n/a | n/a | n/a | n/a |
| v14r | -0.108 | 0.583 | 0.500 | 0.191 | 0.211 | 0.931 | 0.938 | 0.002 | 0.001 | 1.641 |
| v14t | -0.109 | 0.500 | 0.500 | 0.013 | 0.083 | 0.887 | 0.901 | 0.004 | 0.002 | 0.594 |
| v14h | -0.114 | 0.417 | 0.417 | 0.015 | 0.149 | n/a | n/a | n/a | n/a | 2.678 |

## Honest Assessment

- **Vs V14u:** slight improvement in `query_key_match_score` (-0.109 vs -0.114), same `correct_key_selected`/`correct_value_selected`, tiny beta-gap increase (0.012 vs 0.010), but lower `query_value_read_strength` (0.149 vs 0.153).
- **Vs V14s-B / V14r:** still clearly below on write-selectivity and key selection (`store_vs_distractor_beta_gap` 0.012 vs 0.191; `correct_key_selected` 0.500 vs 0.583).
- **Vs V14t:** mostly lateral on retrieval metrics; better read strength (0.149 vs 0.083), similar key-match score, slightly lower beta gap.
- **Geometry/variance behavior:** V14v remains very close to V14r (`key_query_cosine_*` and variances nearly identical), indicating no major geometric shift; this is consistent with the intended minimal local beta-only change.

## Verdict

V14v is **partial**: it improves over V14u on one retrieval-alignment metric (`query_key_match_score`) while preserving stable geometry, but it does **not** reach V14s-B/V14r key-selection/write-selectivity performance.

## Data Sources

- V14v metrics and diagnostics: `results/v14v-delta.md`, `results/v14v-delta.candidate-features.jsonl`.
- V14u/V14s-B/V14r/V14t/V14h comparison carry-over: `results/v14u-vs-v14s-b-v14r-v14t-v14h-comparison.md`.
- V14t/V14r geometry carry-over: `results/v14t-vs-v14s-b-v14r-v14i-v14h-v14g-v14ff-v14e-v14d-comparison.md`.
