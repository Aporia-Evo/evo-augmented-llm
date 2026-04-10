# V14t vs V14s-B / V14r / V14i / V14h / V14g / V14ff / V14e / V14d Comparison

| version | query_key_match_score | correct_key_selected | correct_value_selected | store_vs_distractor_beta_gap | store_vs_distractor_write_gap | query_value_read_strength | key_query_cosine_mean | key_query_cosine_at_query | key_variance_mean | query_variance_mean | mean_memory_frobenius_norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v14t | -0.109 | 0.500 | 0.500 | 0.013 | 0.069 | 0.083 | 0.887 | 0.901 | 0.004 | 0.002 | 0.594 |
| v14s-b | -0.109 | 0.583 | 0.500 | 0.191 | 0.163 | 0.211 | n/a | n/a | n/a | n/a | n/a |
| v14r | -0.108 | 0.583 | 0.500 | 0.191 | 0.163 | 0.211 | 0.931 | 0.938 | 0.002 | 0.001 | 1.641 |
| v14i | -0.110 | 0.500 | 0.500 | 0.017 | 0.000 | 0.363 | n/a | n/a | n/a | n/a | 1.405 |
| v14h | -0.114 | 0.417 | 0.417 | 0.015 | 0.000 | 0.149 | n/a | n/a | n/a | n/a | 2.678 |
| v14g | -0.114 | 0.417 | 0.417 | 0.015 | 0.000 | 0.149 | n/a | n/a | n/a | n/a | 2.678 |
| v14ff | -0.114 | 0.417 | 0.417 | 0.014 | 0.000 | 0.149 | n/a | n/a | n/a | n/a | 2.665 |
| v14e | -0.112 | 0.417 | 0.417 | -0.001 | 0.000 | 0.000 | n/a | n/a | n/a | n/a | 1.134 |
| v14d | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Notes
- `v14t` row: retrieval/selectivity from `results/v14t-delta.md`; geometry + memory norm from best candidate in `results/v14t-delta.candidate-features.jsonl` (`key-value-memory-v1-7-20260410T161029.813691+0000-g0006-c0013`).
- `v14s-b` row: from `results/v14s-ablation-v14r-v14s-comparison.md`; geometry/memory metrics were not reported there, so `n/a`.
- `v14r` and older rows: historical carry-over from `results/v14s-vs-v14r-v14q-v14p-v14o-v14n-v14m-v14l-v14k-v14j-v14i-v14h-v14g-v14ff-v14e-v14d-comparison.md`.
- Regression callout vs V14s-B/V14r: `store_vs_distractor_beta_gap` and `store_vs_distractor_write_gap` are both substantially lower in V14t.
