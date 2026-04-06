# Generation Benchmark Suite: v9b-kv-mid-smoke

- tasks: key_value_memory
- seeds: 7, 11, 13

| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 5 | stateful | 3 | 0.000 | 5.000000 | n/a | n/a | 9.00 | 8.00 |
| key_value_memory | 5 | stateful_plastic_hebb | 3 | 0.000 | 5.000000 | n/a | n/a | 9.00 | 8.00 |
| key_value_memory | 5 | stateful_v2 | 3 | 0.000 | 5.666667 | n/a | n/a | 9.00 | 6.33 |

## Retrieval Metrics

| task | delay | variant | mean_query_accuracy | mean_retrieval_score | mean_query_distance | mean_distractor_load |
| --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 5 | stateful | 0.556 | 0.730 | 4.667 | 3.111 |
| key_value_memory | 5 | stateful_plastic_hebb | 0.556 | 0.730 | 4.667 | 3.111 |
| key_value_memory | 5 | stateful_v2 | 0.630 | 0.764 | 4.667 | 3.111 |

## Retrieval Diagnostics

| task | delay | variant | mean_correct_key_selected | mean_correct_value_selected | mean_query_key_match_score | mean_value_margin | mean_distractor_competition_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 5 | stateful | 0.741 | 0.556 | 0.354 | 0.006 | 0.376 |
| key_value_memory | 5 | stateful_plastic_hebb | 0.741 | 0.556 | 0.356 | 0.007 | 0.374 |
| key_value_memory | 5 | stateful_v2 | 0.741 | 0.630 | 0.390 | 0.073 | 0.374 |
