# V14x vs V14w / V14v / V14u / V14t / V14s-B / V14r Comparison

## Focus Metrics (key_value_memory, delay=8, seed=7)

| version | query_key_match_score | correct_key_selected | correct_value_selected | store_vs_distractor_beta_gap | mean_memory_frobenius_norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| v14x | -0.109 | 0.500 | 0.500 | 0.012 | 1.027 |
| v14w | n/a | n/a | n/a | n/a | n/a |
| v14v | -0.109 | 0.500 | 0.500 | 0.012 | 1.027 |
| v14u | -0.114 | 0.500 | 0.500 | 0.010 | 1.392 |
| v14t | -0.109 | 0.500 | 0.500 | 0.013 | 0.594 |
| v14s-b | -0.109 | 0.583 | 0.500 | 0.191 | n/a |
| v14r | -0.108 | 0.583 | 0.500 | 0.191 | 1.641 |

## Regression Check

- V14x does not improve `correct_key_selected` over V14s-B/V14r (0.500 vs 0.583), so it remains below the established conservative readout baseline on key retrieval.
- V14x is effectively tied with V14v on all tracked retrieval metrics in this run, indicating the query-only sharpening is conservative but not yet beneficial.
- V14x improves over V14u on `query_key_match_score` and memory norm while preserving `correct_value_selected`.
- Relative to V14t, V14x is neutral on retrieval metrics with a slightly lower `store_vs_distractor_beta_gap` and higher memory norm.

## Notes

- No `v14w` benchmark artifacts were found in `results/` in this workspace snapshot; v14w entries are marked `n/a` pending a reproducible v14w delta run.
- Per the run criteria, this iteration should be treated as a regression relative to V14s-B/V14r because key-selection performance did not recover to those baselines.

## Data Sources

- V14x: `results/v14x-delta.md`
- V14v/V14u/V14t/V14s-B/V14r carry-over: `results/v14v-vs-v14u-v14s-b-v14r-v14t-v14h-comparison.md`
