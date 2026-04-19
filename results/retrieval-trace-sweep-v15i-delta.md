# Retrieval Trace Sweep — v15i-delta

## 1. Dataset summary

- benchmark label: `v15i-delta`
- task: `key_value_memory`
- variant: `stateful_v6_delta_memory`
- number of candidates traced: `1`
- episodes per candidate: `3`
- candidate discovery source: `candidate-features`

## 2. Candidate table

| candidate_id | run_id | final score | dominant failure class | consistency fraction |
| --- | --- | ---: | --- | ---: |
| key-value-memory-v1-23-20260413T160226.231073+0000-g0003-c0010 | key-value-memory-v1-23-20260413T160226.231073+0000 | 6.2642 | none | 0.667 |

## 3. Failure frequency table

| failure class | count |
| --- | ---: |
| none | 2 |
| output_decoding | 1 |

## 4. Aggregate metric summary

- overall correct_value_selected fraction: `0.667`
- cross-candidate consistency: `1.000`
- mean query_key_match_score: `0.845`
- mean value_margin: `0.198`
- mean query_memory_alignment: `0.462`
- mean readout_selectivity: `0.292`

## 5. Final verdict

`mostly-correct-but-inconsistent`

## 6. Next-step hint

Prioritize interventions that stabilize retrieval policy consistency across episodes.
