# V14p vs V14o/V14n/V14m/V14l/V14k/V14j/V14i/V14h/V14g/V14ff/V14e/V14d Comparison

Provenance markers:
- **benchmark summary row**: `*-delta.md` (single run aggregate for task `key_value_memory`, delay 8).
- **candidate-features**: `*-delta.candidate-features.jsonl` best `final_max_score` candidate for geometry/coupling metrics.
- Historical labels without fresh artifacts remain **historisch `n/a`**.

| version | query_key_match_score | correct_key_selected | correct_value_selected | store_vs_distractor_beta_gap | store_vs_distractor_write_gap | query_value_read_strength | key_query_cosine_mean | key_query_cosine_at_query | key_variance_mean | query_variance_mean | key_query_projection_strength | query_decoupling_magnitude | mean_memory_frobenius_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v14p | -0.109 | 0.500 | 0.500 | 0.017 | 0.000 | 0.143 | 0.897* | 0.912* | 0.002* | 0.002* | 0.010* | 0.006* | 1.320 |
| v14o | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| v14n | -0.113 | 0.500 | 0.417 | 0.283 | 0.000 | 0.103 | 0.879† | 0.893† | 0.002† | 0.002† | 0.023† | 0.013† | 2.414 |
| v14m | -0.111 | 0.500 | 0.500 | 0.043 | 0.000 | 0.115 | 0.919‡ | 0.903‡ | 0.001‡ | 0.002‡ | 0.014‡ | 0.008‡ | 2.830 |
| v14l | -0.110 | 0.500 | 0.417 | 0.029 | 0.000 | 0.379 | 0.829§ | 0.846§ | 0.004§ | 0.002§ | 0.016§ | 0.010§ | 1.881 |
| v14k | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| v14j | -0.110 | 0.500 | 0.500 | 0.013 | 0.000 | 0.145 | 0.894¶ | 0.922¶ | 0.003¶ | 0.000¶ | n/a | n/a | 2.749 |
| v14i | -0.110 | 0.500 | 0.500 | 0.017 | 0.000 | 0.363 | n/a | n/a | n/a | n/a | n/a | n/a | 1.405 |
| v14h | -0.114 | 0.417 | 0.417 | 0.015 | 0.000 | 0.149 | n/a | n/a | n/a | n/a | n/a | n/a | 2.678 |
| v14g | -0.114 | 0.417 | 0.417 | 0.015 | 0.000 | 0.149 | n/a | n/a | n/a | n/a | n/a | n/a | 2.678 |
| v14ff | -0.114 | 0.417 | 0.417 | 0.014 | 0.000 | 0.149 | n/a | n/a | n/a | n/a | n/a | n/a | 2.665 |
| v14e | -0.112 | 0.417 | 0.417 | -0.001 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | 1.134 |
| v14d | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

\* from `v14p-delta.candidate-features.jsonl` best candidate (`key-value-memory-v1-7-20260410T120232.309672+0000-g0007-c0007`).

† from `v14n-delta.candidate-features.jsonl` best candidate (`key-value-memory-v1-7-20260410T102654.881397+0000-g0010-c0008`).

‡ from V14m row in `v14m-vs-v14l-v14k-v14j-v14i-v14h-v14g-v14ff-v14e-v14d-comparison.md`.

§ from V14l row in `v14m-vs-v14l-v14k-v14j-v14i-v14h-v14g-v14ff-v14e-v14d-comparison.md`.

¶ historical carry-over from `v14m-vs-v14l-v14k-v14j-v14i-v14h-v14g-v14ff-v14e-v14d-comparison.md`.
