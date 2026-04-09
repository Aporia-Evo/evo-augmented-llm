# V14j vs V14i/V14h/V14g/V14ff/V14e/V14d Comparison

Baseline metrics are taken from each `*-delta.md` run summary when available. Missing historical metrics are reported as `n/a`. V14j-only cosine/variance metrics are sourced from `v14j-delta.candidate-features.jsonl` (best final-score candidate).

| version | query_key_match_score | correct_key_selected | correct_value_selected | store_vs_distractor_beta_gap | key_query_cosine_mean | key_query_cosine_at_query | key_variance_mean | query_variance_mean | mean_memory_frobenius_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v14j | -0.110 | 0.500 | 0.500 | 0.013 | 0.894 | 0.922 | 0.003 | 0.000 | 2.749 |
| v14i | -0.110 | 0.500 | 0.500 | 0.017 | n/a | n/a | n/a | n/a | 1.405 |
| v14h | -0.114 | 0.417 | 0.417 | 0.015 | n/a | n/a | n/a | n/a | 2.678 |
| v14g | -0.114 | 0.417 | 0.417 | 0.015 | n/a | n/a | n/a | n/a | 2.678 |
| v14ff | -0.114 | 0.417 | 0.417 | 0.014 | n/a | n/a | n/a | n/a | 2.665 |
| v14e | -0.112 | 0.417 | 0.417 | -0.001 | n/a | n/a | n/a | n/a | 1.134 |
| v14d | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
