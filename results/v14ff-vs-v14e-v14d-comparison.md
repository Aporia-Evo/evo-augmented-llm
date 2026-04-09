# V14ff vs V14e vs V14d (stateful_v6_delta_memory, key_value_memory/kv_easy, seed=7)

## Metric comparison

| metric | v14d-delta | v14e-delta | v14ff-delta | delta (v14ff-v14e) | delta (v14ff-v14d) |
| --- | ---: | ---: | ---: | ---: | ---: |
| final_max_score | 3.766667 | 3.750000 | 3.766667 | +0.016667 | +0.000000 |
| retrieval_score | 0.573440 | 0.573445 | 0.573195 | -0.000250 | -0.000245 |
| query_accuracy | 0.313889 | 0.312500 | 0.313889 | +0.001389 | +0.000000 |
| correct_key_selected | 0.355556 | 0.354167 | 0.354167 | +0.000000 | -0.001389 |
| correct_value_selected | 0.313889 | 0.312500 | 0.313889 | +0.001389 | +0.000000 |
| query_key_match_score | -0.188304 | -0.188308 | -0.188242 | +0.000066 | +0.000062 |
| value_margin | -0.295683 | -0.295675 | -0.296642 | -0.000967 | -0.000959 |
| distractor_competition_score | 0.761744 | 0.761753 | 0.761437 | -0.000316 | -0.000307 |
| beta_at_store | n/a | n/a | 0.394357 | n/a | n/a |
| beta_at_distractor | n/a | n/a | 0.392561 | n/a | n/a |
| beta_at_query | n/a | n/a | 0.407943 | n/a | n/a |
| store_vs_distractor_beta_gap | -0.012280 | -0.012280 | 0.001796 | +0.014076 | +0.014076 |
| query_memory_alignment | 0.957013 | 0.957014 | 0.957008 | -0.000006 | -0.000005 |
| memory_read_strength | n/a | n/a | 0.700571 | n/a | n/a |
| store_memory_update_strength | n/a | n/a | 0.210213 | n/a | n/a |
| delta_correction_magnitude | n/a | 1.817000 | 1.313693 | -0.503307 | n/a |
| readout_selectivity | n/a | n/a | 0.090915 | n/a | n/a |
| mean_memory_frobenius_norm | 1.402266 | 1.402260 | 2.155334 | +0.753074 | +0.753068 |

## Notes

- `v14d-delta`/`v14e-delta` values are taken from the existing historical reports in `results/v14e-vs-v14d-v14c-comparison.md` and `results/v14e-delta.md`.
- Several metrics were not persisted in those historical artifacts and are therefore reported as `n/a` instead of backfilling pseudo-precise values.
- `v14ff-delta` values come from the new run artifacts (`results/v14ff-delta.*`) and search-space summary output.

## Verdict

**neutral**

- The primary write-selection signal improved directionally: `store_vs_distractor_beta_gap` moved from negative in V14d/V14e to slightly positive in V14ff.
- Core retrieval metrics remained effectively flat (small changes in the 1e-4 to 1e-3 range), with slight regression on `retrieval_score` and `value_margin`.
- The instrumentation is clearer for V6: `beta_at_store/distractor/query`, `memory_read_strength`, `store_memory_update_strength`, and calibrated `readout_selectivity` are now explicitly populated.
