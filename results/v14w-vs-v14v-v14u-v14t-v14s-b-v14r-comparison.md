# v14w vs v14v/v14u/v14t/v14s-b/v14r (key_value_memory, delay=8, variant=stateful_v6_delta_memory)

## Retrieval-selection metrics

| label | query_key_match_score | correct_key_selected | correct_value_selected | store_vs_distractor_beta_gap | mean_memory_frobenius_norm |
| --- | ---:| ---:| ---:| ---:| ---:|
| v14w | -0.109 | 0.500 | 0.500 | 0.013 | 1.028 |
| v14v | -0.109 | 0.500 | 0.500 | 0.012 | 1.027 |
| v14u | -0.114 | 0.500 | 0.500 | 0.010 | 1.392 |
| v14t | -0.109 | 0.500 | 0.500 | 0.013 | 0.594 |
| v14s-b | -0.109 | 0.583 | 0.500 | 0.191 | 1.641 |
| v14r | -0.108 | 0.583 | 0.500 | 0.191 | 1.641 |

## Outcome assessment

- **Primary objective (correct_key_selected vs v14v/v14u):**
  - vs **v14v**: unchanged (0.500 -> 0.500)
  - vs **v14u**: unchanged (0.500 -> 0.500)
- **Primary objective (query_key_match_score vs v14v/v14u):**
  - vs **v14v**: unchanged (-0.109 -> -0.109)
  - vs **v14u**: improved (-0.114 -> -0.109)
- **Secondary checks:**
  - correct_value_selected: unchanged at 0.500 relative to all compared labels
  - store_vs_distractor_beta_gap: slightly above v14v/v14u/v14t, but far below v14s-b/v14r
  - mean_memory_frobenius_norm: stable vs v14v, lower than v14u/v14s-b/v14r, higher than v14t

## Conclusion

This iteration shows a **partial mechanical write-localization improvement signal** (query-key match improvement vs v14u and slight beta-gap lift vs v14v/v14u), but **does not produce a meaningful retrieval-selection gain** because `correct_key_selected` remains flat at 0.500 and below v14s-b/v14r.

## Guardrail smoke artifact

- Bit-memory smoke run exported at `results/v14w-bit-memory-smoke.md`.
