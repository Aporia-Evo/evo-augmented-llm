# V14y vs V14x / V14w / V14v / V14u / V14t / V14s-B / V14r Comparison

## Focus Metrics (key_value_memory, delay=8, seed=7)

| version | query_key_match_score | correct_key_selected | correct_value_selected | store_vs_distractor_beta_gap | mean_memory_frobenius_norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| v14y | -0.109 | 0.500 | 0.500 | 0.012 | 1.027 |
| v14x | -0.109 | 0.500 | 0.500 | 0.012 | 1.027 |
| v14w | n/a | n/a | n/a | n/a | n/a |
| v14v | -0.109 | 0.500 | 0.500 | 0.012 | 1.027 |
| v14u | -0.114 | 0.500 | 0.500 | 0.010 | 1.392 |
| v14t | -0.109 | 0.500 | 0.500 | 0.013 | 0.594 |
| v14s-b | -0.109 | 0.583 | 0.500 | 0.191 | n/a |
| v14r | -0.108 | 0.583 | 0.500 | 0.191 | 1.641 |

## Regression Check

- V14y does not improve `correct_key_selected` over V14x/V14w (where available) and remains below V14s-B/V14r (0.500 vs 0.583).
- V14y matches V14x/V14v on the two primary retrieval-selection metrics (`correct_key_selected`, `query_key_match_score`) in this run.
- Guardrail metrics remain unchanged relative to V14x (`correct_value_selected`, `store_vs_distractor_beta_gap`, `mean_memory_frobenius_norm`).
- Per success criteria, V14y should be treated as neutral/failure: the coupled bounded geometry tweak did not unlock retrieval selection in this seed/delay run.

## Notes

- No `v14w` benchmark artifacts were found in `results/` in this workspace snapshot; v14w entries are marked `n/a` pending a reproducible v14w delta run.

## Data Sources

- V14y: `results/v14y-delta.md`
- V14x baseline: `results/v14x-delta.md`
- Prior carry-over table for V14w/V14v/V14u/V14t/V14s-B/V14r: `results/v14x-vs-v14w-v14v-v14u-v14t-v14s-b-v14r-comparison.md`
