# V14g vs V14ff vs V14e vs V14d (stateful_v6_delta_memory, key_value_memory/kv_easy, seed=7)

## Metric comparison

| metric | v14d-delta | v14e-delta | v14ff-delta | v14g-delta | delta (v14g-v14ff) | delta (v14g-v14e) | delta (v14g-v14d) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final_max_score | 3.766667 | 3.750000 | 3.766667 | 3.766667 | -0.000000 | +0.016667 | -0.000000 |
| retrieval_score | 0.573440 | 0.573445 | 0.573195 | 0.573208 | +0.000013 | -0.000237 | -0.000232 |
| query_accuracy | 0.313889 | 0.312500 | 0.313889 | 0.313889 | -0.000000 | +0.001389 | -0.000000 |
| correct_key_selected | 0.355556 | 0.354167 | 0.354167 | 0.354167 | -0.000000 | -0.000000 | -0.001389 |
| correct_value_selected | 0.313889 | 0.312500 | 0.313889 | 0.313889 | -0.000000 | +0.001389 | -0.000000 |
| query_key_match_score | -0.188304 | -0.188308 | -0.188242 | -0.188237 | +0.000005 | +0.000071 | +0.000067 |
| value_margin | -0.295683 | -0.295675 | -0.296642 | -0.296651 | -0.000009 | -0.000976 | -0.000968 |
| distractor_competition_score | 0.761744 | 0.761753 | 0.761437 | 0.761445 | +0.000008 | -0.000308 | -0.000299 |
| store_vs_distractor_beta_gap | -0.012280 | -0.012280 | 0.001796 | 0.001888 | +0.000092 | +0.014168 | +0.014168 |
| query_memory_alignment | 0.957013 | 0.957014 | 0.957008 | 0.954431 | -0.002577 | -0.002583 | -0.002582 |
| mean_memory_frobenius_norm | 1.402266 | 1.402260 | 2.155334 | 2.158717 | +0.003383 | +0.756457 | +0.756451 |
| memory_read_strength | n/a | n/a | 0.700571 | 0.701991 | +0.001420 | n/a | n/a |
| readout_selectivity | n/a | n/a | 0.090915 | 0.011844 | -0.079071 | n/a | n/a |
| key_query_cosine_mean | n/a | n/a | n/a | 0.962661 | n/a | n/a | n/a |
| key_query_cosine_at_query | n/a | n/a | n/a | 0.954431 | n/a | n/a | n/a |

## Notes

- V14d/V14e/V14ff values are sourced from `results/v14ff-vs-v14e-v14d-comparison.md` and earlier V14 comparison artifacts.
- V14g values are means computed from `results/v14g-delta.candidate-features.jsonl` (60 candidates across the requested run).
- Historical metrics that were not persisted are reported as `n/a` without backfilling.
