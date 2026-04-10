# V14l vs V14k/V14i/V14h/V14g/V14ff/V14e/V14d Comparison

Quelle: verfügbare Benchmark-Artefakte unter `results/` (primär `*-delta.md`, ergänzend `*-delta-search-space.md`).

| variant | query_key_match_score | correct_key_selected | correct_value_selected | store_vs_distractor_beta_gap | key_query_cosine_mean | key_query_cosine_at_query | key_variance_mean | query_variance_mean | mean_memory_frobenius_norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v14l | -0.109 | 0.500 | 0.500 | 0.013 | n/a | n/a | n/a | n/a | 0.594 |
| v14k | -0.110 | 0.500 | 0.500 | 0.013 | n/a | n/a | n/a | n/a | 2.749 |
| v14i | -0.110 | 0.500 | 0.500 | 0.017 | n/a | n/a | n/a | n/a | 1.405 |
| v14h | -0.114 | 0.417 | 0.417 | 0.015 | n/a | n/a | n/a | n/a | 2.678 |
| v14g | -0.114 | 0.417 | 0.417 | 0.015 | n/a | n/a | n/a | n/a | 2.678 |
| v14ff | -0.114 | 0.417 | 0.417 | 0.014 | n/a | n/a | n/a | n/a | 2.665 |
| v14e | -0.112 | 0.417 | 0.417 | -0.001 | n/a | n/a | n/a | n/a | 1.134 |
| v14d | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Notes

- Für `key_query_cosine_mean`, `key_query_cosine_at_query`, `key_variance_mean`, `query_variance_mean` wurden in den verfügbaren `v14*`-Artefakten keine direkt auslesbaren Werte gefunden; daher durchgehend `n/a`.
- Für `v14d` war kein `results/v14d-delta.md`-Artefakt verfügbar; daher `n/a`.
