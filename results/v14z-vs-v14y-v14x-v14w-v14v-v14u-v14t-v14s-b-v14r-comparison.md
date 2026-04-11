# V14z comparison vs prior baselines

Compared using existing `results/*.csv` artifacts only. Missing artifacts are marked `n/a`.

| label | artifact_status | correct_value_selected | query_key_match_score | store_vs_distractor_beta_gap | correct_key_selected (secondary) | mean_memory_frobenius_norm | delta_retrieval_selection_pressure_bonus |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v14z-delta | ok | 0.5 | -0.11183234800895055 | 0.04719791243222519 | 0.5 | 1.473684481412304 | n/a |
| v14y-delta | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| v14x-delta | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| v14w-delta | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| v14v-delta | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| v14u-delta | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| v14t-delta | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| v14s-b-delta | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| v14r-delta | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Notes

- `v14s-b-delta.csv` was not found; row is `n/a`.
- The benchmark-suite CSV schema in these artifacts does not include `delta_retrieval_selection_pressure_bonus`; reported as `n/a`.
