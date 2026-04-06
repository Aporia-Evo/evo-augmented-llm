# Generation Benchmark Suite: v9a-smoke-test

- tasks: key_value_memory
- seeds: 7

| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 8 | stateful | 1 | 0.000 | 6.000000 | n/a | n/a | 8.00 | 7.00 |
| key_value_memory | 8 | stateful_plastic_hebb | 1 | 0.000 | 7.000000 | n/a | n/a | 9.00 | 8.00 |
| key_value_memory | 8 | stateful_v2 | 1 | 0.000 | 7.000000 | n/a | n/a | 8.00 | 7.00 |

## Retrieval Metrics

| task | delay | variant | mean_query_accuracy | mean_retrieval_score | mean_query_distance | mean_distractor_load |
| --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 8 | stateful | 0.500 | 0.754 | 6.333 | 5.333 |
| key_value_memory | 8 | stateful_plastic_hebb | 0.583 | 0.752 | 6.333 | 5.333 |
| key_value_memory | 8 | stateful_v2 | 0.583 | 0.781 | 6.333 | 5.333 |
