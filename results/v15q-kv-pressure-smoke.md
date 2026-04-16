# Generation Benchmark Suite: v15q-kv-pressure-smoke

- tasks: key_value_memory
- seeds: 7
- curriculum: 8->8@g6

## Run Results

| seed | run_id | status | success | final_max_score | first_success_generation |
| --- | --- | --- | --- | --- | --- |
| 7 | v15q-delta-7-20260416T173332.965089+0000 | finished | False | 5.251373 | n/a |

| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 8->8@g6 | stateful_v6_delta_memory | 1 | 0.000 | 5.251373 | n/a | n/a | 9.00 | 5.00 |

## Retrieval Metrics

| task | delay | variant | mean_query_accuracy | mean_retrieval_score | mean_query_distance | mean_distractor_load |
| --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 8->8@g6 | stateful_v6_delta_memory | 0.417 | 0.667 | 7.333 | 5.333 |

## Retrieval Diagnostics

| task | delay | variant | mean_correct_key_selected | mean_correct_value_selected | mean_query_key_match_score | mean_value_margin | mean_distractor_competition_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 8->8@g6 | stateful_v6_delta_memory | 0.417 | 0.417 | -0.110 | -0.185 | 0.777 |

## KV Selectivity Diagnostics

| task | delay | variant | mean_store_vs_distractor_write_gap | mean_query_value_read_strength |
| --- | --- | --- | --- | --- |
| key_value_memory | 8->8@g6 | stateful_v6_delta_memory | 0.395 | 0.080 |

## Slot Retrieval Diagnostics

| task | delay | variant | mean_slot_write_focus | mean_slot_query_focus | mean_slot_readout_selectivity | mean_slot_utilization | mean_query_slot_match_max | mean_slot_distractor_leak | mean_write_address_focus | mean_read_address_focus | mean_write_read_address_gap | mean_readout_address_concentration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 8->8@g6 | stateful_v6_delta_memory | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Delta Memory Diagnostics

| task | delay | variant | mean_store_vs_distractor_beta_gap | mean_query_memory_alignment | mean_delta_correction_magnitude | mean_memory_frobenius_norm |
| --- | --- | --- | --- | --- | --- | --- |
| key_value_memory | 8->8@g6 | stateful_v6_delta_memory | 0.509 | 0.837 | 1.008 | 2.490 |
