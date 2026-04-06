# Generation Benchmark Suite: v8c-boundary6

- tasks: bit_memory
- seeds: 7, 11, 13, 17, 19
- curriculum: 5->5,8@g6


| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bit_memory | 5->5,8@g6 | stateful | 5 | 0.600 | 3.788794 | 7.33 | 7.00 | 4.40 | 3.20 |
| bit_memory | 5->5,8@g6 | stateful_plastic_hebb | 5 | 0.800 | 3.710019 | 5.50 | 5.50 | 4.60 | 3.80 |
| bit_memory | 5->5,8@g6 | stateful_v2 | 5 | 0.800 | 3.799423 | 6.00 | 6.00 | 4.20 | 3.40 |
