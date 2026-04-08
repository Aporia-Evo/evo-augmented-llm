# V13a Performance Test Report (2026-04-08)

## Scope

Compared three variants with a larger budget than prior smoke runs:

- `stateful_v2` (baseline)
- `stateful_v4_slots`
- `stateful_v5_addressed_slots`

Budget:

- seeds: `1,2,3,4,5`
- generations: `12`
- population size: `40`
- store: `memory`

## Commands

### bit_memory guardrail

```bash
PYTHONPATH=src python - <<'PY'
from ui.cli import main
raise SystemExit(main([
    'benchmark-suite','--store','memory','--tasks','bit_memory','--seeds','1,2,3,4,5',
    '--variants','stateful_v2,stateful_v4_slots,stateful_v5_addressed_slots',
    '--generations','12','--population-size','40',
    '--output-dir','results','--label','v13a-perf-bit-memory-s1-5-g12-p40'
]))
PY
```

### key_value_memory (kv_trivial via config)

```bash
PYTHONPATH=src python - <<'PY'
from ui.cli import main
raise SystemExit(main([
    'benchmark-suite','--config','configs/key_value_memory_trivial.yaml','--store','memory',
    '--tasks','key_value_memory','--seeds','1,2,3,4,5',
    '--variants','stateful_v2,stateful_v4_slots,stateful_v5_addressed_slots',
    '--generations','12','--population-size','40',
    '--output-dir','results','--label','v13a-perf-kv-trivial-s1-5-g12-p40'
]))
PY
```

### key_value_memory (kv_easy)

```bash
PYTHONPATH=src python - <<'PY'
from ui.cli import main
raise SystemExit(main([
    'benchmark-suite','--store','memory','--tasks','key_value_memory','--seeds','1,2,3,4,5',
    '--variants','stateful_v2,stateful_v4_slots,stateful_v5_addressed_slots',
    '--generations','12','--population-size','40',
    '--key-value-profile','kv_easy',
    '--output-dir','results','--label','v13a-perf-kv-easy-s1-5-g12-p40'
]))
PY
```

## Aggregate Results

### bit_memory (`v13a-perf-bit-memory-s1-5-g12-p40`)

| variant | success_rate | mean_final_max_score | mean_first_success_generation |
|---|---:|---:|---:|
| stateful_v2 | 1.000 | 3.999985 / 4.0 | 5.80 |
| stateful_v4_slots | 1.000 | 3.685430 / 4.0 | 3.00 |
| stateful_v5_addressed_slots | 1.000 | 3.837343 / 4.0 | 5.00 |

### key_value_memory, kv_trivial (`v13a-perf-kv-trivial-s1-5-g12-p40`)

| variant | success_rate | mean_final_max_score | mean_query_accuracy | mean_retrieval_score |
|---|---:|---:|---:|---:|
| stateful_v2 | 1.000 | 2.000000 / 2.0 | 1.000 | 0.665 |
| stateful_v4_slots | 1.000 | 2.000000 / 2.0 | 1.000 | 0.595 |
| stateful_v5_addressed_slots | 0.400 | 1.400000 / 2.0 | 0.700 | 0.564 |

Addressed diagnostics (`stateful_v5_addressed_slots`):

- mean_write_address_focus: 0.000
- mean_read_address_focus: 0.000
- mean_write_read_address_gap: 0.000
- mean_readout_address_concentration: 0.500

### key_value_memory, kv_easy (`v13a-perf-kv-easy-s1-5-g12-p40`)

| variant | success_rate | mean_final_max_score | mean_query_accuracy | mean_retrieval_score |
|---|---:|---:|---:|---:|
| stateful_v2 | 0.000 | 6.600000 / 12.0 | 0.550 | 0.656 |
| stateful_v4_slots | 0.000 | 6.200000 / 12.0 | 0.517 | 0.655 |
| stateful_v5_addressed_slots | 0.000 | 5.000000 / 12.0 | 0.417 | 0.643 |

Addressed diagnostics (`stateful_v5_addressed_slots`):

- mean_write_address_focus: 0.000
- mean_read_address_focus: 0.000
- mean_write_read_address_gap: 0.000
- mean_readout_address_concentration: 0.500

## Verdict

- **Engineering:** ✅ Green (variant runs and diagnostics export consistently across benchmark-suite outputs).
- **Research:** ⚠️ Still open (on this budget, v5 addressed-slots does not outperform v2/v4 on KV, and addressing-focus metrics remain collapsed at ~0 while concentration stays ~0.5).

## Artifacts

- `results/v13a/v13a-perf-bit-memory-s1-5-g12-p40.md`
- `results/v13a/v13a-perf-kv-trivial-s1-5-g12-p40.md`
- `results/v13a/v13a-perf-kv-easy-s1-5-g12-p40.md`
- `results/v13a/v13a-performance-test-2026-04-08.md`
