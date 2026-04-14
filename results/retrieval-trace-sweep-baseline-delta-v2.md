# Retrieval Trace Sweep — baseline-delta-v2

## 1. Dataset summary

- benchmark label: `baseline-delta-v2`
- task: `key_value_memory`
- variant: `stateful_v6_delta_memory`
- number of candidates traced: `2`
- episodes per candidate: `6`
- candidate discovery source: `candidate-features`

## 2. Candidate table

| candidate_id | run_id | final score | dominant failure class | consistency fraction |
| --- | --- | ---: | --- | ---: |
| key-value-memory-v1-11-20260414T064924.584606+0000-g0006-c0011 | key-value-memory-v1-11-20260414T064924.584606+0000 | 6.2434 | none | 0.500 |
| key-value-memory-v1-23-20260414T065609.815856+0000-g0011-c0007 | key-value-memory-v1-23-20260414T065609.815856+0000 | 6.2367 | none | 0.500 |

## 3. Failure frequency table

| failure class | count |
| --- | ---: |
| none | 6 |
| readout_collapse | 6 |

## 4. Aggregate metric summary

- overall correct_value_selected fraction: `0.500`
- cross-candidate consistency: `1.000`
- mean query_key_match_score: `0.877`
- mean value_margin: `0.394`
- mean query_memory_alignment: `0.569`
- mean readout_selectivity: `0.088`

## 5. Final verdict

`mixed-failure-regime`

## 6. Next-step hint

Prioritize a broad diagnostic intervention to separate mixed failure regimes.
