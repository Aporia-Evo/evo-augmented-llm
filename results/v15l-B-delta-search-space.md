## Search Space Summary

| candidates | hof_candidates | success_rate | mean_final_max_score |
| --- | --- | --- | --- |
| 2400 | 180 | 0.000 | 4.281882 |

## Feature Stats

| feature | mean | std |
| --- | --- | --- |
| curriculum_switch_generation | 0.000000 | 0.000000 |
| score_current_phase | 4.281882 | 0.991194 |
| score_delay_3 | 0.000000 | 0.000000 |
| score_delay_5 | 0.000000 | 0.000000 |
| score_delay_8 | 4.281882 | 0.991194 |
| mean_score_over_delays | 4.281882 | 0.991194 |
| delay_score_std | 0.000000 | 0.000000 |
| delay_score_range | 0.000000 | 0.000000 |
| query_accuracy | 0.344028 | 0.079066 |
| retrieval_score | 0.607618 | 0.053514 |
| mean_query_distance | 7.333333 | 0.000000 |
| distractor_load | 5.333333 | 0.000000 |
| retrieval_margin | -0.253458 | 0.065968 |
| retrieval_confusion_rate | 0.655972 | 0.079066 |
| relevant_token_retention | 0.607618 | 0.053514 |
| query_response_margin | 0.193244 | 0.062197 |
| distractor_suppression_ratio | 3.044522 | 3.192141 |
| correct_key_selected | 0.393368 | 0.077344 |
| correct_value_selected | 0.344028 | 0.079066 |
| query_key_match_score | -0.159848 | 0.050026 |
| value_margin | -0.253458 | 0.065968 |
| distractor_competition_score | 0.767466 | 0.025738 |
| slot_key_separation | 0.000000 | 0.000000 |
| slot_value_separation | 0.000000 | 0.000000 |
| slot_write_focus | 0.000000 | 0.000000 |
| slot_query_focus | 0.000000 | 0.000000 |
| slot_readout_selectivity | 0.000000 | 0.000000 |
| slot_utilization | 0.000000 | 0.000000 |
| query_slot_match_max | 0.000000 | 0.000000 |
| slot_distractor_leak | 0.000000 | 0.000000 |
| mean_write_address_focus | 0.000000 | 0.000000 |
| mean_read_address_focus | 0.000000 | 0.000000 |
| write_read_address_gap | 0.000000 | 0.000000 |
| slot_write_specialization | 0.000000 | 0.000000 |
| slot_read_specialization | 0.000000 | 0.000000 |
| address_consistency | 0.000000 | 0.000000 |
| query_read_alignment | 0.000000 | 0.000000 |
| store_write_alignment | 0.000000 | 0.000000 |
| readout_address_concentration | 0.000000 | 0.000000 |
| mean_beta_write | 0.415628 | 0.080814 |
| beta_at_store | 0.472720 | 0.101167 |
| beta_at_distractor | 0.392231 | 0.089653 |
| beta_at_query | 0.426500 | 0.094196 |
| store_vs_distractor_beta_gap | 0.080489 | 0.099747 |
| mean_key_norm | 0.395711 | 0.019864 |
| mean_query_norm | 0.370438 | 0.012626 |
| mean_value_norm | 1.558674 | 0.259208 |
| mean_memory_frobenius_norm | 1.783737 | 0.555053 |
| query_memory_alignment | 0.752645 | 0.153259 |
| store_memory_update_strength | 0.281373 | 0.121070 |
| delta_correction_magnitude | 1.064333 | 0.255064 |
| memory_read_strength | 0.494753 | 0.174869 |
| mean_eta | 0.000000 | 0.000000 |
| mean_plastic_d | 0.000000 | 0.000000 |
| plastic_d_at_lower_bound_fraction | 0.000000 | 0.000000 |
| plastic_d_at_zero_fraction | 1.000000 | 0.000000 |
| mean_abs_delta_w | 0.000000 | 0.000000 |
| max_abs_delta_w | 0.000000 | 0.000000 |
| mean_abs_decay_term | 0.000000 | 0.000000 |
| max_abs_decay_term | 0.000000 | 0.000000 |
| decay_effect_ratio | 0.000000 | 0.000000 |
| decay_near_zero_fraction | 0.000000 | 0.000000 |
| clamp_hit_rate | 0.000000 | 0.000000 |
| plasticity_active_fraction | 0.000000 | 0.000000 |
| mean_abs_fast_state | 0.000000 | 0.000000 |
| mean_abs_slow_state | 0.000000 | 0.000000 |
| slow_fast_contribution_ratio | 0.000000 | 0.000000 |
| mean_abs_fast_state_during_store | 0.000000 | 0.000000 |
| mean_abs_slow_state_during_store | 0.000000 | 0.000000 |
| mean_abs_fast_state_during_query | 0.000000 | 0.000000 |
| mean_abs_slow_state_during_query | 0.000000 | 0.000000 |
| mean_abs_fast_state_during_distractor | 0.000000 | 0.000000 |
| mean_abs_slow_state_during_distractor | 0.000000 | 0.000000 |
| slow_query_coupling | 0.000000 | 0.000000 |
| store_query_state_gap | 0.000000 | 0.000000 |
| slow_fast_retrieval_ratio | 0.000000 | 0.000000 |
| retrieval_state_alignment | 0.000000 | 0.000000 |
| node_count | 9.218750 | 0.474410 |
| enabled_conn_count | 7.068333 | 1.242778 |

## HOF vs Non-HOF

| feature | hof_mean | hof_std | non_hof_mean | non_hof_std |
| --- | --- | --- | --- | --- |
| count | 180 | 0.000 | 2220 | 0.000 |
| final_max_score | 5.542874 | 0.585158 | 4.179639 | 0.946035 |
| curriculum_switch_generation | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| score_current_phase | 5.542874 | 0.585158 | 4.179639 | 0.946035 |
| score_delay_3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| score_delay_5 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| score_delay_8 | 5.542874 | 0.585158 | 4.179639 | 0.946035 |
| mean_score_over_delays | 5.542874 | 0.585158 | 4.179639 | 0.946035 |
| delay_score_std | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| delay_score_range | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| query_accuracy | 0.444444 | 0.047304 | 0.335886 | 0.075452 |
| retrieval_score | 0.639374 | 0.027068 | 0.605043 | 0.054297 |
| mean_query_distance | 7.333333 | 0.000000 | 7.333333 | 0.000000 |
| distractor_load | 5.333333 | 0.000000 | 5.333333 | 0.000000 |
| retrieval_margin | -0.205023 | 0.031294 | -0.257385 | 0.066480 |
| retrieval_confusion_rate | 0.555556 | 0.047304 | 0.664114 | 0.075452 |
| relevant_token_retention | 0.639374 | 0.027068 | 0.605043 | 0.054297 |
| query_response_margin | 0.166475 | 0.048871 | 0.195414 | 0.062655 |
| distractor_suppression_ratio | 3.265940 | 3.191966 | 3.026569 | 3.191482 |
| correct_key_selected | 0.460648 | 0.046837 | 0.387913 | 0.076762 |
| correct_value_selected | 0.444444 | 0.047304 | 0.335886 | 0.075452 |
| query_key_match_score | -0.118857 | 0.018991 | -0.163172 | 0.050289 |
| value_margin | -0.205023 | 0.031294 | -0.257385 | 0.066480 |
| distractor_competition_score | 0.758232 | 0.012890 | 0.768215 | 0.026367 |
| slot_key_separation | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| slot_value_separation | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| slot_write_focus | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| slot_query_focus | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| slot_readout_selectivity | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| slot_utilization | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| query_slot_match_max | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| slot_distractor_leak | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_write_address_focus | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_read_address_focus | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| write_read_address_gap | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| slot_write_specialization | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| slot_read_specialization | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| address_consistency | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| query_read_alignment | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| store_write_alignment | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| readout_address_concentration | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_beta_write | 0.445834 | 0.063730 | 0.413179 | 0.081554 |
| beta_at_store | 0.530327 | 0.097786 | 0.468049 | 0.099992 |
| beta_at_distractor | 0.410171 | 0.072940 | 0.390776 | 0.090719 |
| beta_at_query | 0.456993 | 0.070531 | 0.424027 | 0.095433 |
| store_vs_distractor_beta_gap | 0.120156 | 0.107799 | 0.077273 | 0.098367 |
| mean_key_norm | 0.402031 | 0.021577 | 0.395199 | 0.019630 |
| mean_query_norm | 0.369123 | 0.010348 | 0.370545 | 0.012787 |
| mean_value_norm | 1.591271 | 0.255736 | 1.556031 | 0.259308 |
| mean_memory_frobenius_norm | 1.951712 | 0.525190 | 1.770117 | 0.555181 |
| query_memory_alignment | 0.716613 | 0.167719 | 0.755567 | 0.151651 |
| store_memory_update_strength | 0.347675 | 0.142673 | 0.275997 | 0.117519 |
| delta_correction_magnitude | 1.083945 | 0.271403 | 1.062743 | 0.253626 |
| memory_read_strength | 0.520702 | 0.182143 | 0.492649 | 0.174097 |
| mean_eta | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_plastic_d | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| plastic_d_at_lower_bound_fraction | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| plastic_d_at_zero_fraction | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| mean_abs_delta_w | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| max_abs_delta_w | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_abs_decay_term | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| max_abs_decay_term | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| decay_effect_ratio | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| decay_near_zero_fraction | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| clamp_hit_rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| plasticity_active_fraction | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_abs_fast_state | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_abs_slow_state | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| slow_fast_contribution_ratio | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_abs_fast_state_during_store | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_abs_slow_state_during_store | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_abs_fast_state_during_query | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_abs_slow_state_during_query | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_abs_fast_state_during_distractor | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mean_abs_slow_state_during_distractor | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| slow_query_coupling | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| store_query_state_gap | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| slow_fast_retrieval_ratio | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| retrieval_state_alignment | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| node_count | 9.233333 | 0.528099 | 9.217568 | 0.469768 |
| enabled_conn_count | 7.005556 | 1.318658 | 7.073423 | 1.236282 |

## Success vs Failure

| feature | success_mean | success_std | failure_mean | failure_std |
| --- | --- | --- | --- | --- |
| count | 0 | 0.000 | 2400 | 0.000 |
| final_max_score | n/a | n/a | 4.281882 | 0.991194 |
| curriculum_switch_generation | n/a | n/a | 0.000000 | 0.000000 |
| score_current_phase | n/a | n/a | 4.281882 | 0.991194 |
| score_delay_3 | n/a | n/a | 0.000000 | 0.000000 |
| score_delay_5 | n/a | n/a | 0.000000 | 0.000000 |
| score_delay_8 | n/a | n/a | 4.281882 | 0.991194 |
| mean_score_over_delays | n/a | n/a | 4.281882 | 0.991194 |
| delay_score_std | n/a | n/a | 0.000000 | 0.000000 |
| delay_score_range | n/a | n/a | 0.000000 | 0.000000 |
| query_accuracy | n/a | n/a | 0.344028 | 0.079066 |
| retrieval_score | n/a | n/a | 0.607618 | 0.053514 |
| mean_query_distance | n/a | n/a | 7.333333 | 0.000000 |
| distractor_load | n/a | n/a | 5.333333 | 0.000000 |
| retrieval_margin | n/a | n/a | -0.253458 | 0.065968 |
| retrieval_confusion_rate | n/a | n/a | 0.655972 | 0.079066 |
| relevant_token_retention | n/a | n/a | 0.607618 | 0.053514 |
| query_response_margin | n/a | n/a | 0.193244 | 0.062197 |
| distractor_suppression_ratio | n/a | n/a | 3.044522 | 3.192141 |
| correct_key_selected | n/a | n/a | 0.393368 | 0.077344 |
| correct_value_selected | n/a | n/a | 0.344028 | 0.079066 |
| query_key_match_score | n/a | n/a | -0.159848 | 0.050026 |
| value_margin | n/a | n/a | -0.253458 | 0.065968 |
| distractor_competition_score | n/a | n/a | 0.767466 | 0.025738 |
| slot_key_separation | n/a | n/a | 0.000000 | 0.000000 |
| slot_value_separation | n/a | n/a | 0.000000 | 0.000000 |
| slot_write_focus | n/a | n/a | 0.000000 | 0.000000 |
| slot_query_focus | n/a | n/a | 0.000000 | 0.000000 |
| slot_readout_selectivity | n/a | n/a | 0.000000 | 0.000000 |
| slot_utilization | n/a | n/a | 0.000000 | 0.000000 |
| query_slot_match_max | n/a | n/a | 0.000000 | 0.000000 |
| slot_distractor_leak | n/a | n/a | 0.000000 | 0.000000 |
| mean_write_address_focus | n/a | n/a | 0.000000 | 0.000000 |
| mean_read_address_focus | n/a | n/a | 0.000000 | 0.000000 |
| write_read_address_gap | n/a | n/a | 0.000000 | 0.000000 |
| slot_write_specialization | n/a | n/a | 0.000000 | 0.000000 |
| slot_read_specialization | n/a | n/a | 0.000000 | 0.000000 |
| address_consistency | n/a | n/a | 0.000000 | 0.000000 |
| query_read_alignment | n/a | n/a | 0.000000 | 0.000000 |
| store_write_alignment | n/a | n/a | 0.000000 | 0.000000 |
| readout_address_concentration | n/a | n/a | 0.000000 | 0.000000 |
| mean_beta_write | n/a | n/a | 0.415628 | 0.080814 |
| beta_at_store | n/a | n/a | 0.472720 | 0.101167 |
| beta_at_distractor | n/a | n/a | 0.392231 | 0.089653 |
| beta_at_query | n/a | n/a | 0.426500 | 0.094196 |
| store_vs_distractor_beta_gap | n/a | n/a | 0.080489 | 0.099747 |
| mean_key_norm | n/a | n/a | 0.395711 | 0.019864 |
| mean_query_norm | n/a | n/a | 0.370438 | 0.012626 |
| mean_value_norm | n/a | n/a | 1.558674 | 0.259208 |
| mean_memory_frobenius_norm | n/a | n/a | 1.783737 | 0.555053 |
| query_memory_alignment | n/a | n/a | 0.752645 | 0.153259 |
| store_memory_update_strength | n/a | n/a | 0.281373 | 0.121070 |
| delta_correction_magnitude | n/a | n/a | 1.064333 | 0.255064 |
| memory_read_strength | n/a | n/a | 0.494753 | 0.174869 |
| mean_eta | n/a | n/a | 0.000000 | 0.000000 |
| mean_plastic_d | n/a | n/a | 0.000000 | 0.000000 |
| plastic_d_at_lower_bound_fraction | n/a | n/a | 0.000000 | 0.000000 |
| plastic_d_at_zero_fraction | n/a | n/a | 1.000000 | 0.000000 |
| mean_abs_delta_w | n/a | n/a | 0.000000 | 0.000000 |
| max_abs_delta_w | n/a | n/a | 0.000000 | 0.000000 |
| mean_abs_decay_term | n/a | n/a | 0.000000 | 0.000000 |
| max_abs_decay_term | n/a | n/a | 0.000000 | 0.000000 |
| decay_effect_ratio | n/a | n/a | 0.000000 | 0.000000 |
| decay_near_zero_fraction | n/a | n/a | 0.000000 | 0.000000 |
| clamp_hit_rate | n/a | n/a | 0.000000 | 0.000000 |
| plasticity_active_fraction | n/a | n/a | 0.000000 | 0.000000 |
| mean_abs_fast_state | n/a | n/a | 0.000000 | 0.000000 |
| mean_abs_slow_state | n/a | n/a | 0.000000 | 0.000000 |
| slow_fast_contribution_ratio | n/a | n/a | 0.000000 | 0.000000 |
| mean_abs_fast_state_during_store | n/a | n/a | 0.000000 | 0.000000 |
| mean_abs_slow_state_during_store | n/a | n/a | 0.000000 | 0.000000 |
| mean_abs_fast_state_during_query | n/a | n/a | 0.000000 | 0.000000 |
| mean_abs_slow_state_during_query | n/a | n/a | 0.000000 | 0.000000 |
| mean_abs_fast_state_during_distractor | n/a | n/a | 0.000000 | 0.000000 |
| mean_abs_slow_state_during_distractor | n/a | n/a | 0.000000 | 0.000000 |
| slow_query_coupling | n/a | n/a | 0.000000 | 0.000000 |
| store_query_state_gap | n/a | n/a | 0.000000 | 0.000000 |
| slow_fast_retrieval_ratio | n/a | n/a | 0.000000 | 0.000000 |
| retrieval_state_alignment | n/a | n/a | 0.000000 | 0.000000 |
| node_count | n/a | n/a | 9.218750 | 0.474410 |
| enabled_conn_count | n/a | n/a | 7.068333 | 1.242778 |

## Hints

- plastic_d haeuft sich stark bei 0.0; der AD-Suchraum scheint oft in wenig aktiven Decay-Regeln zu landen.
- clamp_hit_rate ist sehr niedrig; die Plastizitaet kollidiert selten mit der Clamp und wirkt eher untersteuert.
- delta_w bleibt meist klein; die plastische Laufzeitspur wirkt eher schwach oder vorsichtig.
- Der Decay-Beitrag wirkt meist schwach; D greift oft nur gering in die Laufzeitdynamik ein.
- Die Retrieval-Task ist noch fordernd; viele Kandidaten verlieren relevante Information vor der Query.
- Distraktoren werden im Mittel gut unterdrueckt; relevante Information setzt sich klarer gegen Rauschen durch.
- Schon die Key-Selektion bleibt oft unsicher; der Query-Pfad findet relevante Stores noch nicht verlaesslich.
- Die Ziel-Store-Spur gewinnt gegen konkurrierende Stores oft nicht klar; falsche Kontexte bleiben bei der Query konkurrenzfaehig.
- Distraktoren konkurrieren sichtbar mit dem Zielsignal; der Retrieval-Pfad ist noch anfaellig fuer falsche Attraktoren.
- Query-Match bleibt schwach; die Query koppelt nur locker an den relevanten Key-Zustand.
- Zwischen Store- und Query-Zustand geht viel interne Ausrichtung verloren; Retention und Abruf sind noch locker gekoppelt.
- Der Query-Pfad bleibt eher fast-state-nah; slow-state-Retrieval ist noch nicht dominant ausgepraegt.
- Slot-Nutzung ist unausgewogen; mindestens ein Slot bleibt oft ungenutzt und begrenzt die Retrieval-Kapazitaet.
- Query fokussiert Slots noch zu schwach; kein klar dominanter Match-Slot waehrend der Query.
- Slot-Adressierung bleibt weitgehend symmetrisch und diffus; Write/Read-Fokus kollabiert nahe Null.
- Query-Memory-Alignment ist klar messbar; der Delta-Pfad zeigt funktionale inhaltsbasierte Auslese.
- Memory-Norm bleibt im mittleren Bereich; kein offensichtlicher Kollaps oder Divergenz im Delta-Speicher.
- Die langsame Spur bleibt meist klein; der slow-state-Zweig wirkt oft kaum genutzt.
- HOF- und Nicht-HOF-Kandidaten liegen bei mean_eta nah beieinander.
