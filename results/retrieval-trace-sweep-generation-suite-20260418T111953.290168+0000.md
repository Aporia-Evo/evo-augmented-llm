# Retrieval Trace Sweep — generation-suite-20260418T111953.290168+0000

## 1. Dataset summary

- benchmark label: `generation-suite-20260418T111953.290168+0000`
- task: `key_value_memory`
- variant: `stateful_v6_delta_memory_v16c`
- number of candidates traced: `1`
- episodes per candidate: `6`
- candidate discovery source: `candidate-features`

## 2. Candidate table

| candidate_id | run_id | final score | dominant failure class | consistency fraction |
| --- | --- | ---: | --- | ---: |
| v16c-amp-magnitude-11-20260418T112848.145933+0000-g0011-c0006 | v16c-amp-magnitude-11-20260418T112848.145933+0000 | 6.2143 | none | 0.500 |

## 3. Failure frequency table

| failure class | count |
| --- | ---: |
| none | 3 |
| output_decoding | 3 |

## 4. Aggregate metric summary

- overall correct_value_selected fraction: `0.500`
- cross-candidate consistency: `1.000`
- mean query_key_match_score: `0.886`
- mean value_margin: `0.409`
- mean query_memory_alignment: `0.489`
- mean readout_selectivity: `0.387`

## 5. Final verdict

`mixed-failure-regime`

## 6. Next-step hint

Prioritize a broad diagnostic intervention to separate mixed failure regimes.
