# Generation Benchmark Suite: v9a-curriculum

- tasks: key_value_memory
- seeds: 7, 11, 13, 17, 19
- curriculum: 3->8@g6


| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3->8@g6 | stateful | 5 | 0.000 | 6.200000 | n/a | n/a | 8.60 | 5.80 |
| key_value_memory | 3->8@g6 | stateful_plastic_hebb | 5 | 0.000 | 6.400000 | n/a | n/a | 8.60 | 5.80 |
| key_value_memory | 3->8@g6 | stateful_v2 | 5 | 0.000 | 7.400000 | n/a | n/a | 9.00 | 6.00 |

## Retrieval Metrics

| task | delay | variant | mean_query_accuracy | mean_retrieval_score | mean_query_distance | mean_distractor_load |
| --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3->8@g6 | stateful | 0.517 | 0.754 | 6.333 | 5.333 |
| key_value_memory | 3->8@g6 | stateful_plastic_hebb | 0.533 | 0.767 | 6.333 | 5.333 |
| key_value_memory | 3->8@g6 | stateful_v2 | 0.617 | 0.789 | 6.333 | 5.333 |
