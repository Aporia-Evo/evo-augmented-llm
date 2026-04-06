# Generation Benchmark Suite: v8c-boundary4

- tasks: bit_memory
- seeds: 7, 11, 13, 17, 19
- curriculum: 5->5,8@g4


| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bit_memory | 5->5,8@g4 | stateful | 5 | 0.600 | 3.757856 | 7.00 | 6.00 | 4.60 | 3.20 |
| bit_memory | 5->5,8@g4 | stateful_plastic_hebb | 5 | 0.800 | 3.691023 | 6.00 | 6.00 | 4.80 | 4.00 |
| bit_memory | 5->5,8@g4 | stateful_v2 | 5 | 0.600 | 3.800100 | 6.67 | 7.00 | 4.80 | 4.40 |
