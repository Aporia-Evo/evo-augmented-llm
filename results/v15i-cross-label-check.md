# Search-Space Cross-Label Comparison

- labels: pr1-delta, pr2-delta, pr3-delta, baseline-delta-multiseed, v15i-delta

## Compact Metrics

| label | rows | plastic_d_at_zero_fraction | plastic_d_at_lower_bound_fraction | clamp_hit_rate | mean_abs_delta_w | max_abs_delta_w | slot_utilization | mean_abs_fast_state | mean_abs_slow_state | slow_fast_contribution_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pr1-delta | 480 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pr2-delta | 480 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pr3-delta | 480 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| baseline-delta-multiseed | 1440 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| v15i-delta | 2400 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Top Hints by Label

### pr1-delta
- plastic_d haeuft sich stark bei 0.0; der AD-Suchraum scheint oft in wenig aktiven Decay-Regeln zu landen.
- clamp_hit_rate ist sehr niedrig; die Plastizitaet kollidiert selten mit der Clamp und wirkt eher untersteuert.
- delta_w bleibt meist klein; die plastische Laufzeitspur wirkt eher schwach oder vorsichtig.

### pr2-delta
- plastic_d haeuft sich stark bei 0.0; der AD-Suchraum scheint oft in wenig aktiven Decay-Regeln zu landen.
- clamp_hit_rate ist sehr niedrig; die Plastizitaet kollidiert selten mit der Clamp und wirkt eher untersteuert.
- delta_w bleibt meist klein; die plastische Laufzeitspur wirkt eher schwach oder vorsichtig.

### pr3-delta
- plastic_d haeuft sich stark bei 0.0; der AD-Suchraum scheint oft in wenig aktiven Decay-Regeln zu landen.
- clamp_hit_rate ist sehr niedrig; die Plastizitaet kollidiert selten mit der Clamp und wirkt eher untersteuert.
- delta_w bleibt meist klein; die plastische Laufzeitspur wirkt eher schwach oder vorsichtig.

### baseline-delta-multiseed
- plastic_d haeuft sich stark bei 0.0; der AD-Suchraum scheint oft in wenig aktiven Decay-Regeln zu landen.
- clamp_hit_rate ist sehr niedrig; die Plastizitaet kollidiert selten mit der Clamp und wirkt eher untersteuert.
- delta_w bleibt meist klein; die plastische Laufzeitspur wirkt eher schwach oder vorsichtig.

### v15i-delta
- plastic_d haeuft sich stark bei 0.0; der AD-Suchraum scheint oft in wenig aktiven Decay-Regeln zu landen.
- clamp_hit_rate ist sehr niedrig; die Plastizitaet kollidiert selten mit der Clamp und wirkt eher untersteuert.
- delta_w bleibt meist klein; die plastische Laufzeitspur wirkt eher schwach oder vorsichtig.

## Missing Labels

- v15f-delta: no candidate-feature export or search-space markdown found
- v15h-delta: no candidate-feature export or search-space markdown found
