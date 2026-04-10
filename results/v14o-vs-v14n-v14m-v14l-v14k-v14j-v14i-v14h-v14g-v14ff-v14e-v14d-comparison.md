# V14o vs V14n/V14m/V14l/V14k/V14j/V14i/V14h/V14g/V14ff/V14e/V14d Comparison

Provenance markers in `source_provenance`:
- `benchmark summary row` = values from `*-delta.md` summary row.
- `candidate-features` = values from `*-delta.candidate-features.jsonl` best candidate.
- `historical n/a` = no local artifact for this metric/version in this run context.

| version | query_key_match_score | correct_key_selected | correct_value_selected | store_vs_distractor_beta_gap | query_value_read_strength | mean_memory_frobenius_norm | key_query_cosine_mean | key_query_cosine_at_query | key_variance_mean | query_variance_mean | key_query_projection_strength | query_decoupling_magnitude | source_provenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v14o | -0.111 | 0.500 | 0.417 | 0.107 | 0.311 | 1.862 | 0.867 | 0.877 | 0.006 | 0.001 | 0.044 | 0.026 | benchmark summary row + candidate-features |
| v14n | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | historical n/a |
| v14m | -0.111 | 0.500 | 0.500 | 0.043 | 0.115 | 2.830 | 0.919 | 0.903 | 0.001 | 0.002 | 0.014 | 0.008 | benchmark summary row + candidate-features (historical carry-over) |
| v14l | -0.110 | 0.500 | 0.417 | 0.029 | 0.379 | 1.881 | 0.829 | 0.846 | 0.004 | 0.002 | 0.016 | 0.010 | benchmark summary row + candidate-features (historical carry-over) |
| v14k | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | historical n/a |
| v14j | -0.110 | 0.500 | 0.500 | 0.013 | 0.145 | 2.749 | 0.894 | 0.922 | 0.003 | 0.000 | n/a | n/a | benchmark summary row + candidate-features (historical carry-over) |
| v14i | -0.110 | 0.500 | 0.500 | 0.017 | 0.363 | 1.405 | n/a | n/a | n/a | n/a | n/a | n/a | benchmark summary row (historical carry-over) |
| v14h | -0.114 | 0.417 | 0.417 | 0.015 | 0.149 | 2.678 | n/a | n/a | n/a | n/a | n/a | n/a | benchmark summary row (historical carry-over) |
| v14g | -0.114 | 0.417 | 0.417 | 0.015 | 0.149 | 2.678 | n/a | n/a | n/a | n/a | n/a | n/a | benchmark summary row (historical carry-over) |
| v14ff | -0.114 | 0.417 | 0.417 | 0.014 | 0.149 | 2.665 | n/a | n/a | n/a | n/a | n/a | n/a | benchmark summary row (historical carry-over) |
| v14e | -0.112 | 0.417 | 0.417 | -0.001 | 0.000 | 1.134 | n/a | n/a | n/a | n/a | n/a | n/a | benchmark summary row (historical carry-over) |
| v14d | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | historical n/a |

Notes:
- V14o benchmark metrics come from `results/v14o-delta.md`.
- V14o geometry/coupling metrics come from the best candidate (`max final_max_score`) in `results/v14o-delta.candidate-features.jsonl`.
- V14m..V14d values are carried over from `results/v14m-vs-v14l-v14k-v14j-v14i-v14h-v14g-v14ff-v14e-v14d-comparison.md` where available.
