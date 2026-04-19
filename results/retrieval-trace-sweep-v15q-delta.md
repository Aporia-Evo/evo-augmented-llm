# Retrieval Trace Sweep — v15q-delta

## 1. Dataset summary

- benchmark label: `v15q-delta`
- task: `key_value_memory`
- variant: `stateful_v6_delta_memory`
- number of candidates traced: `2`
- episodes per candidate: `3`
- candidate discovery source: `candidate-features`

## 2. Candidate table

| candidate_id | run_id | final score | dominant failure class | consistency fraction |
| --- | --- | ---: | --- | ---: |
| key-value-memory-v1-17-20260416T170357.528517+0000-g0010-c0010 | key-value-memory-v1-17-20260416T170357.528517+0000 | 5.2297 | none | 0.667 |
| key-value-memory-v1-23-20260416T170651.155956+0000-g0011-c0011 | key-value-memory-v1-23-20260416T170651.155956+0000 | 5.2074 | none | 0.667 |

## 3. Failure frequency table

| failure class | count |
| --- | ---: |
| none | 4 |
| output_decoding | 1 |
| readout_collapse | 1 |

## 4. Aggregate metric summary

- overall correct_value_selected fraction: `0.667`
- cross-candidate consistency: `1.000`
- mean query_key_match_score: `0.860`
- mean value_margin: `0.433`
- mean query_memory_alignment: `0.604`
- mean readout_selectivity: `0.260`

## 5. Final verdict

`mostly-correct-but-inconsistent`

## 6. Next-step hint

Prioritize interventions that stabilize retrieval policy consistency across episodes.
