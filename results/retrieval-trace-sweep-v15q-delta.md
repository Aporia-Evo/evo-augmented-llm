# Retrieval Trace Sweep — v15q-delta

## 1. Dataset summary

- benchmark label: `v15q-delta`
- task: `key_value_memory`
- variant: `stateful_v6_delta_memory`
- number of candidates traced: `1`
- episodes per candidate: `3`
- candidate discovery source: `candidate-features`

## 2. Candidate table

| candidate_id | run_id | final score | dominant failure class | consistency fraction |
| --- | --- | ---: | --- | ---: |
| v15q-delta-23-20260416T181120.134513+0000-g0013-c0035 | v15q-delta-23-20260416T181120.134513+0000 | 6.2361 | none | 0.667 |

## 3. Failure frequency table

| failure class | count |
| --- | ---: |
| none | 2 |
| output_decoding | 1 |

## 4. Aggregate metric summary

- overall correct_value_selected fraction: `0.667`
- cross-candidate consistency: `1.000`
- mean query_key_match_score: `0.936`
- mean value_margin: `0.249`
- mean query_memory_alignment: `0.759`
- mean readout_selectivity: `0.197`

## 5. Final verdict

`mostly-correct-but-inconsistent`

## 6. Next-step hint

Prioritize interventions that stabilize retrieval policy consistency across episodes.
