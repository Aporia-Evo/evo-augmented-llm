# Search-Space Cross-Label Comparison

- labels: v15q-delta, v14z-delta

## Compact Metrics

| label | rows | plastic_d_at_zero_fraction | plastic_d_at_lower_bound_fraction | clamp_hit_rate | mean_abs_delta_w | max_abs_delta_w | slot_utilization | mean_abs_fast_state | mean_abs_slow_state | slow_fast_contribution_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v15q-delta | 4320 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| v14z-delta | 480 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Top Hints by Label

### v15q-delta
- plastic_d haeuft sich stark bei 0.0; der AD-Suchraum scheint oft in wenig aktiven Decay-Regeln zu landen.
- clamp_hit_rate ist sehr niedrig; die Plastizitaet kollidiert selten mit der Clamp und wirkt eher untersteuert.
- delta_w bleibt meist klein; die plastische Laufzeitspur wirkt eher schwach oder vorsichtig.

### v14z-delta
- plastic_d haeuft sich stark bei 0.0; der AD-Suchraum scheint oft in wenig aktiven Decay-Regeln zu landen.
- clamp_hit_rate ist sehr niedrig; die Plastizitaet kollidiert selten mit der Clamp und wirkt eher untersteuert.
- delta_w bleibt meist klein; die plastische Laufzeitspur wirkt eher schwach oder vorsichtig.
