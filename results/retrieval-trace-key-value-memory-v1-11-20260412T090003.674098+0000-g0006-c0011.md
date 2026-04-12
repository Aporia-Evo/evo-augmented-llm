# Retrieval Trace Report — key-value-memory-v1-11-20260412T090003.674098+0000-g0006-c0011

## Episode Meta

- value_levels: `[0.0, 0.5, 1.0]`
- step_roles: `['store', 'distractor', 'store', 'distractor', 'separator', 'query']`
- store_value_ids: `[0, 1]`
- target_value_ids: `[0]`
- predicted_value_ids: `[0]`

## Per-Step Diagnostics

| step | role | node | beta_t | store_sig | query_sig | kq_cos | mem_frob | qm_align | readout_sel | update_frob | node_out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | store | 8 | 0.414 | 0.503 | 0.458 | 0.952 | 0.948 | 0.938 | 0.005 | 0.227 | -0.7860 |
| 1 | distractor | 8 | 0.351 | 0.473 | 0.488 | 0.896 | 1.615 | 0.925 | 0.029 | 0.218 | -0.9853 |
| 2 | store | 8 | 0.593 | 0.636 | 0.328 | 0.819 | 1.650 | 0.155 | 0.173 | 0.526 | 0.9514 |
| 3 | distractor | 8 | 0.346 | 0.467 | 0.494 | 0.892 | 1.938 | 0.301 | 0.132 | 0.235 | -0.9966 |
| 4 | separator | 8 | 0.346 | 0.472 | 0.488 | 0.896 | 2.319 | 0.501 | 0.061 | 0.191 | -0.9856 |
| 5 | query | 8 | 0.354 | 0.481 | 0.479 | 0.885 | 2.617 | 0.560 | 0.050 | 0.162 | -0.9486 |

## Query-Step Readout Detail

**Node 8 (step 5)**

- read_t: `[-0.2071, -0.2005, -0.1946, -0.1845, -0.1778, -0.1659, -0.1594, -0.1519]`
- q_focus: `[0.2242, 0.2116, 0.1757, 0.1314, 0.0578, 0.0584, 0.0652, 0.0755]`
- selective_readout: `-0.1891`
- read_contrast: `0.0552`
- readout_scalar: `-0.1581`
- node_output: `-0.9486`

### Value-Level Projections

| value_id | value | distance | winner | target |
| --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.0257 | **>>** | TARGET |
| 1 | 0.500 | 0.4743 |  |  |
| 2 | 1.000 | 0.9743 |  |  |

## Verdict

**Retrieval correct**

Triggered by `correct_value_selected` = 1.0000 (threshold: 1.0000)
