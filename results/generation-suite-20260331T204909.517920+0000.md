# Generation Benchmark Suite: generation-suite-20260331T204909.517920+0000

- tasks: delayed_xor, bit_memory
- seeds: 7, 11, 13, 17, 19

| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bit_memory | 1 | stateful | 5 | 1.000 | 3.999875 | 6.60 | 8.00 | 5.20 | 5.00 |
| bit_memory | 1 | stateless | 5 | 0.400 | 3.174569 | 7.00 | 7.00 | 5.60 | 5.40 |
| delayed_xor | 1 | stateful | 5 | 0.000 | 14.000580 | n/a | n/a | 4.80 | 3.80 |
| delayed_xor | 1 | stateless | 5 | 0.000 | 13.999994 | n/a | n/a | 5.80 | 5.20 |
