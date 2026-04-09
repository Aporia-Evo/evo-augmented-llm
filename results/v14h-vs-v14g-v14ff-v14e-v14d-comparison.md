# V14h vs V14g/V14ff/V14e/V14d Comparison

## Scope
- Task: `key_value_memory`
- Profile: `kv_easy`
- Variant: `stateful_v6_delta_memory`
- Delay: `8`
- Seed(s): `7`

## Retrieval + Core Delta-Memory Metrics

| metric | v14h-delta | v14g-delta | v14ff-delta | v14e-delta | v14d-delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| query_key_match_score | -0.114 | -0.114 | -0.114 | -0.112 | n/a |
| correct_key_selected | 0.417 | 0.417 | 0.417 | 0.417 | n/a |
| correct_value_selected | 0.417 | 0.417 | 0.417 | 0.417 | n/a |
| store_vs_distractor_beta_gap | 0.015 | 0.015 | 0.014 | -0.001 | n/a |
| query_memory_alignment | 0.935 | 0.935 | 0.934 | 0.934 | n/a |
| mean_memory_frobenius_norm | 2.678 | 2.678 | 2.665 | 1.134 | n/a |

## Geometry / Decoupling Diagnostics

| metric | v14h-delta | v14g-delta | v14ff-delta | v14e-delta | v14d-delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| key_query_cosine_mean | n/a | n/a | n/a | n/a | n/a |
| key_query_cosine_at_query | n/a | n/a | n/a | n/a | n/a |
| key_variance_mean | n/a | n/a | n/a | n/a | n/a |
| query_variance_mean | n/a | n/a | n/a | n/a | n/a |
| key_query_projection_strength | n/a | n/a | n/a | n/a | n/a |
| query_decoupling_magnitude | n/a | n/a | n/a | n/a | n/a |

## Notes
- `v14d-delta` artifacts were not available in `results/` under this exact label; values are marked as `n/a`.
- The existing generation markdown reports (`v14g/v14ff/v14e`) do not include the new geometry diagnostics introduced for V14h (`key_variance_mean`, `query_variance_mean`, `key_query_projection_strength`, `query_decoupling_magnitude`), so historical entries remain `n/a`.
- No historical backfilling was performed.
