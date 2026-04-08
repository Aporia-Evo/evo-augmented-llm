# V14e vs V14d vs V14c (stateful_v6_delta_memory, key_value_memory/kv_easy, seed=7)

| metric | v14c-baseline | v14d-delta | v14e-delta | delta (v14e-v14d) | delta (v14e-v14c) |
| --- | ---: | ---: | ---: | ---: | ---: |
| final_max_score | 3.766667 | 3.766667 | 3.750000 | -0.016667 | -0.016667 |
| retrieval_score | 0.573380 | 0.573440 | 0.573445 | 0.000005 | 0.000065 |
| query_accuracy | 0.313889 | 0.313889 | 0.312500 | -0.001389 | -0.001389 |
| correct_key_selected | 0.354167 | 0.355556 | 0.354167 | -0.001389 | 0.000000 |
| correct_value_selected | 0.313889 | 0.313889 | 0.312500 | -0.001389 | -0.001389 |
| query_key_match_score | -0.188377 | -0.188304 | -0.188308 | -0.000004 | 0.000069 |
| value_margin | -0.295783 | -0.295683 | -0.295675 | 0.000008 | 0.000108 |
| distractor_competition_score | 0.761758 | 0.761744 | 0.761753 | 0.000009 | -0.000005 |
| store_vs_distractor_beta_gap | -0.012280 | -0.012280 | -0.012280 | 0.000000 | 0.000000 |
| query_memory_alignment | 0.971097 | 0.957013 | 0.957014 | 0.000001 | -0.014083 |
| mean_memory_frobenius_norm | 1.395105 | 1.402266 | 1.402260 | -0.000006 | 0.007155 |
| archive_occupied_cells | 4.000000 | 5.000000 | 5.000000 | 0.000000 | 1.000000 |
| strategy_diversity_proxy | 1.000000 | 2.000000 | 2.000000 | 0.000000 | 1.000000 |

## Verdict

**neutral** — V14e keeps the delta-memory dynamics stable and preserves the V14d-level archive diversity, but it does not produce a meaningful gain on the target readout-selectivity retrieval metrics (query_key_match_score, correct_key_selected, correct_value_selected). Small uplifts in retrieval_score/value_margin are marginal and accompanied by flat-to-slightly-worse key/value selection rates.
