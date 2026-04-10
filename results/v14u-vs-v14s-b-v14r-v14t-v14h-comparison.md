# V14u vs V14s-B / V14r / V14t / V14h Comparison

## Focus Metrics (key_value_memory, delay=8, seed=7)

| version | correct_key_selected | query_key_match_score | correct_value_selected | store_vs_distractor_beta_gap | store_vs_distractor_write_gap | query_value_read_strength | mean_memory_frobenius_norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v14u | 0.500 | -0.114 | 0.500 | 0.010 | 0.023 | 0.153 | 1.392 |
| v14s-b | 0.583 | -0.109 | 0.500 | 0.191 | 0.163 | 0.211 | n/a |
| v14r | 0.583 | -0.108 | 0.500 | 0.191 | 0.163 | 0.211 | 1.641 |
| v14t | 0.500 | -0.109 | 0.500 | 0.013 | 0.069 | 0.083 | 0.594 |
| v14h | 0.417 | -0.114 | 0.417 | 0.015 | 0.000 | 0.149 | 2.678 |

## Interpretation

- **Main retrieval-selection targets vs v14s-b/v14r are not improved.**
  - `correct_key_selected` is lower in v14u (0.500 vs 0.583).
  - `query_key_match_score` is slightly worse/more negative in v14u (-0.114 vs -0.109/-0.108).
  - `store_vs_distractor_beta_gap` is much lower in v14u (0.010 vs 0.191).
- **Value-side behavior is flat** on `correct_value_selected` (0.500 across v14u, v14s-b, v14r, v14t).
- Relative to v14t, v14u trades lower write/read selectivity (`store_vs_distractor_write_gap`) for slightly higher `query_value_read_strength`, but this does not translate into better key-selection metrics.

## Verdict

Under the stated success criteria, **v14u is not promising yet** and should be treated as **no meaningful progress** relative to v14s-b/v14r.

## Data sources

- v14u metrics: `results/v14u-delta.md`.
- v14t / v14s-b / v14r / v14h carry-over metrics: `results/v14t-vs-v14s-b-v14r-v14i-v14h-v14g-v14ff-v14e-v14d-comparison.md`.
