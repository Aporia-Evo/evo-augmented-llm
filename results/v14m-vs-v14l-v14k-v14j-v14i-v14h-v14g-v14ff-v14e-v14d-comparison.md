# V14m vs V14l/V14k/V14j/V14i/V14h/V14g/V14ff/V14e/V14d Comparison

This report separates metric provenance:
- **Benchmark summary row**: `*-delta.md` (single run aggregate for task `key_value_memory`, delay 8).
- **Candidate-features best candidate**: `*-delta.candidate-features.jsonl` (max `final_max_score`) for geometry/coupling metrics.
- **Historical carry-over**: older rows copied from the previous comparison report when no new run artifact exists.
- Missing historical values remain `n/a`.

| version | query_key_match_score | correct_key_selected | correct_value_selected | store_vs_distractor_beta_gap | store_vs_distractor_write_gap | query_value_read_strength | key_query_cosine_mean | key_query_cosine_at_query | key_variance_mean | query_variance_mean | key_query_projection_strength | query_decoupling_magnitude | mean_memory_frobenius_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v14m | -0.111 | 0.500 | 0.500 | 0.043 | 0.000 | 0.115 | 0.919* | 0.903* | 0.001* | 0.002* | 0.014* | 0.008* | 2.830 |
| v14l | -0.110 | 0.500 | 0.417 | 0.029 | 0.000 | 0.379 | 0.829† | 0.846† | 0.004† | 0.002† | 0.016† | 0.010† | 1.881 |
| v14k | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| v14j | -0.110 | 0.500 | 0.500 | 0.013 | 0.000 | 0.145 | 0.894‡ | 0.922‡ | 0.003‡ | 0.000‡ | n/a | n/a | 2.749 |
| v14i | -0.110 | 0.500 | 0.500 | 0.017 | 0.000 | 0.363 | n/a | n/a | n/a | n/a | n/a | n/a | 1.405 |
| v14h | -0.114 | 0.417 | 0.417 | 0.015 | 0.000 | 0.149 | n/a | n/a | n/a | n/a | n/a | n/a | 2.678 |
| v14g | -0.114 | 0.417 | 0.417 | 0.015 | 0.000 | 0.149 | n/a | n/a | n/a | n/a | n/a | n/a | 2.678 |
| v14ff | -0.114 | 0.417 | 0.417 | 0.014 | 0.000 | 0.149 | n/a | n/a | n/a | n/a | n/a | n/a | 2.665 |
| v14e | -0.112 | 0.417 | 0.417 | -0.001 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | 1.134 |
| v14d | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

\* from V14m best candidate in `v14m-delta.candidate-features.jsonl`.

† from V14l best candidate in `v14l-delta.candidate-features.jsonl`.

‡ historical values from `v14l-vs-v14k-v14j-v14i-v14h-v14g-v14ff-v14e-v14d-comparison.md`.
