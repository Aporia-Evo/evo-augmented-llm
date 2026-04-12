# Fitness Landscape Report — baseline-delta-multiseed

## Dataset Summary

- benchmark_label: `baseline-delta-multiseed`
- total candidate records: 1440
- generations: 12
- population per generation (mean): 120

## Fitness vs Retrieval Correlations (mean Pearson across generations)

| feature | mean_pearson | mean_spearman | min_pearson | max_pearson |
| --- | --- | --- | --- | --- |
| `correct_value_selected` | 1.000 | 0.842 | 1.000 | 1.000 |
| `query_key_match_score` | 0.754 | 0.722 | 0.611 | 0.814 |
| `store_vs_distractor_beta_gap` | 0.017 | 0.346 | -0.126 | 0.179 |
| `query_memory_alignment` | -0.080 | -0.166 | -0.181 | 0.012 |
| `retrieval_margin` | 0.543 | 0.558 | 0.343 | 0.625 |
| `readout_selectivity` | -0.040 | -0.056 | -0.315 | 0.156 |
| `key_query_cosine_mean` | -0.025 | -0.023 | -0.168 | 0.098 |
| `key_query_cosine_at_query` | -0.047 | -0.099 | -0.194 | 0.095 |

## Selection-Pressure Bonus Variance Decomposition

| term | variance_fraction |
| --- | --- |
| `correct_value_selected` | 64.8% |
| `positive_beta_gap` | 10.8% |
| `negative_match_penalty` | 0.9% |
| `correct_key_selected` | 0.1% |
| `kq_cos_query_penalty` | 0.0% |
| `kq_cos_mean_penalty` | 0.0% |
| `key_var_penalty` | 0.0% |
| `query_var_penalty` | 0.0% |
| `positive_query_match` | 0.0% |

## Top-k Phenotype Diversity Trend

| generation | mean_pairwise_L2 |
| --- | --- |
| 0 | 0.2894 |
| 1 | 0.2744 |
| 2 | 0.2133 |
| 3 | 0.2733 |
| 4 | 0.3236 |
| 5 | 0.2983 |
| 6 | 0.2850 |
| 7 | 0.2560 |
| 8 | 0.2421 |
| 9 | 0.2592 |
| 10 | 0.2639 |
| 11 | 0.2958 |

## Retrieval-Axis Fitness Curve

Bins of `correct_value_selected` with mean fitness per bin.

| bin_center | mean_fitness |
| --- | --- |
| 0.10 | 1.9727 |
| 0.30 | 3.8901 |
| 0.50 | 5.2385 |

## Verdict

**weak-but-monotone-gradient**

Weak but monotone gradient detected (best: correct_value_selected r=1.000). Evolution sees a signal but may be too slow. Consider increasing selection pressure or population size.
