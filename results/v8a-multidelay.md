# Generation Benchmark Suite: v8a-multidelay

- tasks: bit_memory
- seeds: 7, 11, 13, 17, 19

| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bit_memory | 5,8 | stateful | 5 | 0.600 | 3.790500 | 5.67 | 6.00 | 4.60 | 3.80 |
| bit_memory | 5,8 | stateful_plastic_hebb | 5 | 0.800 | 3.792447 | 5.25 | 5.50 | 4.60 | 4.00 |
| bit_memory | 5,8 | stateful_v2 | 5 | 0.400 | 3.578509 | 4.50 | 4.50 | 4.60 | 4.20 |
