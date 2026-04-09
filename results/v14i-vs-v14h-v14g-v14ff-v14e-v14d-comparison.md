# V14i vs V14h/V14g/V14ff/V14e/V14d Comparison

## Scope
- Task: `key_value_memory`
- Profile: `kv_easy`
- Variant: `stateful_v6_delta_memory`
- Delay: `8`
- Seed(s): `7`

## Retrieval + Core Delta-Memory Metrics (benchmark-suite run summary)

| metric | v14i-delta | v14h-delta | v14g-delta | v14ff-delta | v14e-delta | v14d-delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| query_key_match_score | -0.110 | -0.114 | -0.114 | -0.114 | -0.112 | n/a |
| correct_key_selected | 0.500 | 0.417 | 0.417 | 0.417 | 0.417 | n/a |
| correct_value_selected | 0.500 | 0.417 | 0.417 | 0.417 | 0.417 | n/a |
| store_vs_distractor_beta_gap | 0.017 | 0.015 | 0.015 | 0.014 | -0.001 | n/a |
| mean_memory_frobenius_norm | 1.405 | 2.678 | 2.678 | 2.665 | 1.134 | n/a |

## Geometry / Decoupling Diagnostics (search-space means where available)

| metric | v14i-delta | v14h-delta | v14g-delta | v14ff-delta | v14e-delta | v14d-delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| key_query_cosine_mean | n/a | 0.963290 | n/a | n/a | n/a | n/a |
| key_query_cosine_at_query | n/a | 0.955175 | n/a | n/a | n/a | n/a |
| key_variance_mean | n/a | 0.000671 | n/a | n/a | n/a | n/a |
| query_variance_mean | n/a | 0.000430 | n/a | n/a | n/a | n/a |

## V14i-only Search-Space Means (same metrics over all candidates)

| metric | v14i-delta mean |
| --- | ---: |
| query_key_match_score | -0.162348 |
| correct_key_selected | 0.393750 |
| correct_value_selected | 0.340972 |
| store_vs_distractor_beta_gap | 0.038645 |
| mean_memory_frobenius_norm | 2.088596 |

## Notes
- `v14d-delta` artifacts are not present under this label in the repository outputs; entries are marked `n/a`.
- Historical labels (`v14g`, `v14ff`, `v14e`) have benchmark markdown outputs for retrieval/core metrics, but no locally available candidate-features JSONL with geometry metrics, so geometry is `n/a` for those labels.
- The V14i benchmark run improved retrieval-key selection metrics versus V14h/V14g/V14ff/V14e on the run summary row, while maintaining a non-collapsed memory norm.
