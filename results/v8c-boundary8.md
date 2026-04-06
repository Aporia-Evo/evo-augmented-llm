# Generation Benchmark Suite: v8c-boundary8

- tasks: bit_memory
- seeds: 7, 11, 13, 17, 19
- curriculum: 5->5,8@g8


| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bit_memory | 5->5,8@g8 | stateful | 5 | 0.400 | 3.686770 | 6.50 | 6.50 | 4.60 | 3.20 |
| bit_memory | 5->5,8@g8 | stateful_plastic_hebb | 5 | 1.000 | 3.702632 | 6.00 | 6.00 | 4.40 | 3.80 |
| bit_memory | 5->5,8@g8 | stateful_v2 | 5 | 0.800 | 3.899213 | 5.75 | 5.50 | 4.40 | 3.80 |
