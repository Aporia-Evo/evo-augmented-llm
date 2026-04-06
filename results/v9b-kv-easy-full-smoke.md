# Generation Benchmark Suite: v9b-kv-easy-full-smoke

- tasks: key_value_memory
- seeds: 7, 11, 13

| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful | 3 | 0.000 | 3.666667 | n/a | n/a | 9.67 | 7.00 |
| key_value_memory | 3 | stateful_plastic_hebb | 3 | 0.000 | 3.000000 | n/a | n/a | 9.33 | 8.33 |
| key_value_memory | 3 | stateful_v2 | 3 | 0.000 | 4.666667 | n/a | n/a | 10.00 | 7.33 |

## Retrieval Metrics

| task | delay | variant | mean_query_accuracy | mean_retrieval_score | mean_query_distance | mean_distractor_load |
| --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful | 0.611 | 0.723 | 2.000 | 1.000 |
| key_value_memory | 3 | stateful_plastic_hebb | 0.500 | 0.584 | 2.000 | 1.000 |
| key_value_memory | 3 | stateful_v2 | 0.778 | 0.758 | 2.000 | 1.000 |

## Retrieval Diagnostics

| task | delay | variant | mean_correct_key_selected | mean_correct_value_selected | mean_query_key_match_score | mean_value_margin | mean_distractor_competition_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful | 1.000 | 0.611 | 0.723 | 0.023 | 0.000 |
| key_value_memory | 3 | stateful_plastic_hebb | 1.000 | 0.500 | 0.584 | -0.206 | 0.000 |
| key_value_memory | 3 | stateful_v2 | 1.000 | 0.778 | 0.758 | 0.098 | 0.000 |
