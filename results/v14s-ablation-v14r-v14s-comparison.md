# V14s Readout Ablations on top of V14r

## Setup
- Base implementation for all ablations: **V14r** delta-memory path.
- Task/run setting (all rows): `key_value_memory`, `kv_easy`, delay `8`, seed `7`, generations `12`, population `40`.
- Ablation labels:
  - `v14s-a-value-focus`
  - `v14s-b-contrast-branch`
  - `v14s-c-mild-readout`

## Key metrics (V14r/V14s + three ablations)

| label | query_key_match_score | correct_key_selected | correct_value_selected | store_vs_distractor_beta_gap | store_vs_distractor_write_gap | query_value_read_strength | mean_final_max_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v14r (reference) | -0.108 | 0.583 | 0.500 | 0.191 | 0.163 | 0.211 | 6.233692 |
| v14s (reference) | -0.111 | 0.500 | 0.417 | 0.042 | 0.163 | 0.146 | 5.219415 |
| v14s-a-value-focus | -0.139 | 0.500 | 0.500 | 0.005 | 0.031 | 0.154 | 6.206239 |
| v14s-b-contrast-branch | -0.109 | 0.583 | 0.500 | 0.191 | 0.163 | 0.211 | 6.233687 |
| v14s-c-mild-readout | -0.114 | 0.500 | 0.500 | 0.010 | 0.023 | 0.154 | 6.206924 |

## Readout interpretation
- **A (value_focus only)** preserves value hit-rate, but collapses the store-vs-distractor separation (`beta_gap`, `write_gap`) and weakens read strength.
- **B (contrast branch only)** tracks V14r almost exactly across all target metrics; no obvious regression signal.
- **C (mild reweight only)** again degrades store-vs-distractor separation and read strength while not improving value selection.

## Conclusion
The regression pattern from V14s is most consistent with the **value_focus reweighting branch** (especially its interaction with read-centered positivity), not with adding the bounded contrast branch itself.
