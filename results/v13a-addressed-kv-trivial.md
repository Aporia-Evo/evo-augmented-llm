# Generation Benchmark Suite: v13a-addressed-kv-trivial

- tasks: key_value_memory
- seeds: 1, 2

| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful_v4_slots | 2 | 1.000 | 2.000000 | 1.00 | 1.00 | 9.00 | 8.00 |
| key_value_memory | 3 | stateful_v5_addressed_slots | 2 | 0.000 | 1.000000 | n/a | n/a | 9.00 | 8.00 |

## Retrieval Metrics

| task | delay | variant | mean_query_accuracy | mean_retrieval_score | mean_query_distance | mean_distractor_load |
| --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful_v4_slots | 1.000 | 0.715 | 1.000 | 0.000 |
| key_value_memory | 3 | stateful_v5_addressed_slots | 0.500 | 0.500 | 1.000 | 0.000 |

## Retrieval Diagnostics

| task | delay | variant | mean_correct_key_selected | mean_correct_value_selected | mean_query_key_match_score | mean_value_margin | mean_distractor_competition_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful_v4_slots | 1.000 | 1.000 | 0.715 | 0.430 | 0.000 |
| key_value_memory | 3 | stateful_v5_addressed_slots | 1.000 | 0.500 | 0.500 | 0.000 | 0.000 |

## KV Selectivity Diagnostics

| task | delay | variant | mean_store_vs_distractor_write_gap | mean_query_value_read_strength |
| --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful_v4_slots | 0.000 | 0.000 |
| key_value_memory | 3 | stateful_v5_addressed_slots | 0.000 | 0.000 |

## Slot Retrieval Diagnostics

| task | delay | variant | mean_slot_write_focus | mean_slot_query_focus | mean_slot_readout_selectivity | mean_slot_utilization | mean_query_slot_match_max | mean_slot_distractor_leak | mean_write_address_focus | mean_read_address_focus | mean_write_read_address_gap | mean_readout_address_concentration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 3 | stateful_v4_slots | 0.426 | 0.135 | 0.506 | 0.500 | 0.567 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| key_value_memory | 3 | stateful_v5_addressed_slots | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
