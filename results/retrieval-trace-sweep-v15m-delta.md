# Retrieval Trace Sweep — v15m-delta

## 1. Dataset summary

- benchmark label: `v15m-delta`
- task: `key_value_memory`
- variant: `stateful_v6_delta_memory`
- number of candidates traced: `2`
- episodes per candidate: `3`
- candidate discovery source: `candidate-features`

## 2. Candidate table

| candidate_id | run_id | final score | dominant failure class | consistency fraction |
| --- | --- | ---: | --- | ---: |
| key-value-memory-v1-23-20260414T195701.818120+0000-g0009-c0007 | key-value-memory-v1-23-20260414T195701.818120+0000 | 6.2712 | none | 0.667 |
| key-value-memory-v1-7-20260414T194539.816337+0000-g0006-c0010 | key-value-memory-v1-7-20260414T194539.816337+0000 | 6.2640 | none | 0.667 |

## 3. Failure frequency table

| failure class | count |
| --- | ---: |
| none | 4 |
| output_decoding | 1 |
| readout_collapse | 1 |

## 4. Aggregate metric summary

- overall correct_value_selected fraction: `0.667`
- cross-candidate consistency: `1.000`
- mean query_key_match_score: `0.916`
- mean value_margin: `0.439`
- mean query_memory_alignment: `0.582`
- mean readout_selectivity: `0.078`

## 5. Final verdict

`mostly-correct-but-inconsistent`

## 6. Next-step hint

Prioritize interventions that stabilize retrieval policy consistency across episodes.
