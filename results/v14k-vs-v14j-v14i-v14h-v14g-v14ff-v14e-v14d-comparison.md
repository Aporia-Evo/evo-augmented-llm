# V14k vs V14j/V14i/V14h/V14g/V14ff/V14e/V14d Comparison

Core retrieval metrics are sourced from each `*-delta.md` summary when available. `readout_selectivity` and V14k cosine/variance metrics are taken from `v14k-delta.candidate-features.jsonl` for the run `best_candidate_id`; historical values not present in artifacts are marked `n/a`.

| version | query_key_match_score | correct_key_selected | correct_value_selected | store_vs_distractor_beta_gap | query_value_read_strength | readout_selectivity | key_query_cosine_mean | key_query_cosine_at_query | key_variance_mean | query_variance_mean | mean_memory_frobenius_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v14k | -0.110 | 0.500 | 0.500 | 0.013 | 0.145 | 0.001 | 0.894 | 0.922 | 0.003 | 0.000 | 2.749 |
| v14j | -0.110 | 0.500 | 0.500 | 0.013 | 0.145 | n/a | 0.894 | 0.922 | 0.003 | 0.000 | 2.749 |
| v14i | -0.110 | 0.500 | 0.500 | 0.017 | 0.363 | n/a | n/a | n/a | n/a | n/a | 1.405 |
| v14h | -0.114 | 0.417 | 0.417 | 0.015 | 0.149 | n/a | n/a | n/a | n/a | n/a | 2.678 |
| v14g | -0.114 | 0.417 | 0.417 | 0.015 | 0.149 | n/a | n/a | n/a | n/a | n/a | 2.678 |
| v14ff | -0.114 | 0.417 | 0.417 | 0.014 | 0.149 | n/a | n/a | n/a | n/a | n/a | 2.665 |
| v14e | -0.112 | 0.417 | 0.417 | -0.001 | 0.000 | n/a | n/a | n/a | n/a | n/a | 1.134 |
| v14d | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
