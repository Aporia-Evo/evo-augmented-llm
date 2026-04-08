# Generation Benchmark Suite: v13a-perf-kv-trivial-s1-5-g12-p40

- tasks: key_value_memory
- seeds: 1, 2, 3, 4, 5

| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful_v2 | 5 | 1.000 | 2.000000 | 2.40 | 3.00 | 9.20 | 6.80 |
| key_value_memory | 3 | stateful_v4_slots | 5 | 1.000 | 2.000000 | 1.80 | 1.00 | 9.00 | 6.20 |
| key_value_memory | 3 | stateful_v5_addressed_slots | 5 | 0.400 | 1.400000 | 3.50 | 3.50 | 9.20 | 7.40 |

## Retrieval Metrics

| task | delay | variant | mean_query_accuracy | mean_retrieval_score | mean_query_distance | mean_distractor_load |
| --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful_v2 | 1.000 | 0.665 | 1.000 | 0.000 |
| key_value_memory | 3 | stateful_v4_slots | 1.000 | 0.595 | 1.000 | 0.000 |
| key_value_memory | 3 | stateful_v5_addressed_slots | 0.700 | 0.564 | 1.000 | 0.000 |

## Retrieval Diagnostics

| task | delay | variant | mean_correct_key_selected | mean_correct_value_selected | mean_query_key_match_score | mean_value_margin | mean_distractor_competition_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful_v2 | 1.000 | 1.000 | 0.665 | 0.330 | 0.000 |
| key_value_memory | 3 | stateful_v4_slots | 1.000 | 1.000 | 0.595 | 0.190 | 0.000 |
| key_value_memory | 3 | stateful_v5_addressed_slots | 1.000 | 0.700 | 0.564 | 0.128 | 0.000 |

## KV Selectivity Diagnostics

| task | delay | variant | mean_store_vs_distractor_write_gap | mean_query_value_read_strength |
| --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful_v2 | 0.000 | 0.000 |
| key_value_memory | 3 | stateful_v4_slots | 0.000 | 0.000 |
| key_value_memory | 3 | stateful_v5_addressed_slots | 0.000 | 0.000 |

## Slot Retrieval Diagnostics

| task | delay | variant | mean_slot_write_focus | mean_slot_query_focus | mean_slot_readout_selectivity | mean_slot_utilization | mean_query_slot_match_max | mean_slot_distractor_leak | mean_write_address_focus | mean_read_address_focus | mean_write_read_address_gap | mean_readout_address_concentration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful_v2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| key_value_memory | 3 | stateful_v4_slots | 0.377 | 0.118 | 0.248 | 0.500 | 0.559 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| key_value_memory | 3 | stateful_v5_addressed_slots | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
