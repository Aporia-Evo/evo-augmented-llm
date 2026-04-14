# Retrieval Trace Sweep — v15l-B-delta

## 1. Dataset summary

- benchmark label: `v15l-B-delta`
- task: `key_value_memory`
- variant: `stateful_v6_delta_memory`
- number of candidates traced: `2`
- episodes per candidate: `6`
- candidate discovery source: `candidate-features`

## 2. Candidate table

| candidate_id | run_id | final score | dominant failure class | consistency fraction |
| --- | --- | ---: | --- | ---: |
| key-value-memory-v1-23-20260414T113705.544148+0000-g0009-c0016 | key-value-memory-v1-23-20260414T113705.544148+0000 | 6.2709 | none | 0.500 |
| key-value-memory-v1-11-20260414T112944.658248+0000-g0003-c0006 | key-value-memory-v1-11-20260414T112944.658248+0000 | 6.2541 | none | 0.500 |

## 3. Failure frequency table

| failure class | count |
| --- | ---: |
| none | 6 |
| output_decoding | 3 |
| readout_collapse | 3 |

## 4. Aggregate metric summary

- overall correct_value_selected fraction: `0.500`
- cross-candidate consistency: `1.000`
- mean query_key_match_score: `0.898`
- mean value_margin: `0.347`
- mean query_memory_alignment: `0.510`
- mean readout_selectivity: `0.162`

## 5. Final verdict

`mixed-failure-regime`

## 6. Next-step hint

Prioritize a broad diagnostic intervention to separate mixed failure regimes.
