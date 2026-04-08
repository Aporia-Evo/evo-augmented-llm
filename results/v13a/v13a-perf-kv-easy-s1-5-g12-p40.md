# Generation Benchmark Suite: v13a-perf-kv-easy-s1-5-g12-p40

- tasks: key_value_memory
- seeds: 1, 2, 3, 4, 5

| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 8 | stateful_v2 | 5 | 0.000 | 6.600000 | n/a | n/a | 9.40 | 5.80 |
| key_value_memory | 8 | stateful_v4_slots | 5 | 0.000 | 6.200000 | n/a | n/a | 9.20 | 7.60 |
| key_value_memory | 8 | stateful_v5_addressed_slots | 5 | 0.000 | 5.000000 | n/a | n/a | 9.00 | 7.60 |

## Retrieval Metrics

| task | delay | variant | mean_query_accuracy | mean_retrieval_score | mean_query_distance | mean_distractor_load |
| --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 8 | stateful_v2 | 0.550 | 0.656 | 7.333 | 5.333 |
| key_value_memory | 8 | stateful_v4_slots | 0.517 | 0.655 | 7.333 | 5.333 |
| key_value_memory | 8 | stateful_v5_addressed_slots | 0.417 | 0.643 | 7.333 | 5.333 |

## Retrieval Diagnostics

| task | delay | variant | mean_correct_key_selected | mean_correct_value_selected | mean_query_key_match_score | mean_value_margin | mean_distractor_competition_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 8 | stateful_v2 | 0.550 | 0.550 | -0.078 | -0.177 | 0.734 |
| key_value_memory | 8 | stateful_v4_slots | 0.533 | 0.517 | -0.094 | -0.181 | 0.749 |
| key_value_memory | 8 | stateful_v5_addressed_slots | 0.433 | 0.417 | -0.120 | -0.205 | 0.762 |

## KV Selectivity Diagnostics

| task | delay | variant | mean_store_vs_distractor_write_gap | mean_query_value_read_strength |
| --- | --- | --- | --- | --- |
| key_value_memory | 8 | stateful_v2 | 0.000 | 0.000 |
| key_value_memory | 8 | stateful_v4_slots | 0.000 | 0.000 |
| key_value_memory | 8 | stateful_v5_addressed_slots | 0.000 | 0.000 |

## Slot Retrieval Diagnostics

| task | delay | variant | mean_slot_write_focus | mean_slot_query_focus | mean_slot_readout_selectivity | mean_slot_utilization | mean_query_slot_match_max | mean_slot_distractor_leak | mean_write_address_focus | mean_read_address_focus | mean_write_read_address_gap | mean_readout_address_concentration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 8 | stateful_v2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| key_value_memory | 8 | stateful_v4_slots | 0.624 | 0.131 | 0.369 | 1.000 | 0.565 | 0.254 | 0.000 | 0.000 | 0.000 | 0.000 |
| key_value_memory | 8 | stateful_v5_addressed_slots | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
