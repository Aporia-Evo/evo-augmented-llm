# V13a Addressed-Slots: Benchmark Evidence Snapshot (2026-04-08)

This note closes the previously missing benchmark evidence items:

- bit_memory guardrail comparison
- key_value_memory with trivial profile
- key_value_memory with kv_easy profile
- concrete numeric readout for `stateful_v5_addressed_slots`

## Commands run

```bash
PYTHONPATH=src python - <<'PY'
from ui.cli import main
raise SystemExit(main([
    'benchmark-suite','--store','memory','--tasks','bit_memory','--seeds','1,2,3',
    '--variants','stateful_v4_slots,stateful_v5_addressed_slots',
    '--generations','8','--population-size','30',
    '--output-dir','results','--label','v13a-addressed-bit-memory-guardrail'
]))
PY
```

```bash
PYTHONPATH=src python - <<'PY'
from ui.cli import main
raise SystemExit(main([
    'benchmark-suite','--config','configs/key_value_memory_trivial.yaml',
    '--store','memory','--tasks','key_value_memory','--seeds','1,2',
    '--variants','stateful_v4_slots,stateful_v5_addressed_slots',
    '--generations','8','--population-size','30',
    '--output-dir','results','--label','v13a-addressed-kv-trivial'
]))
PY
```

```bash
PYTHONPATH=src python - <<'PY'
from ui.cli import main
raise SystemExit(main([
    'benchmark-suite','--store','memory','--tasks','key_value_memory','--seeds','1,2',
    '--variants','stateful_v4_slots,stateful_v5_addressed_slots',
    '--generations','8','--population-size','30',
    '--key-value-profile','kv_easy',
    '--output-dir','results','--label','v13a-addressed-kv-easy'
]))
PY
```

## Results

### 1) bit_memory guardrail

Source: `results/v13a-addressed-bit-memory-guardrail.csv`

| variant | runs | success_rate | mean_final_max_score | mean_first_success_generation |
|---|---:|---:|---:|---:|
| stateful_v4_slots | 3 | 0.667 | 3.3649 / 4.0 | 1.5 |
| stateful_v5_addressed_slots | 3 | 0.667 | 3.1458 / 4.0 | 5.5 |

### 2) key_value_memory (trivial)

Source: `results/v13a-addressed-kv-trivial.csv`

| variant | runs | success_rate | mean_final_max_score | mean_query_accuracy | mean_retrieval_score |
|---|---:|---:|---:|---:|---:|
| stateful_v4_slots | 2 | 1.000 | 2.0 / 2.0 | 1.000 | 0.715 |
| stateful_v5_addressed_slots | 2 | 0.000 | 1.0 / 2.0 | 0.500 | 0.500 |

Addressed metrics (`stateful_v5_addressed_slots`):

- mean_write_address_focus: 0.000
- mean_read_address_focus: 0.000
- mean_write_read_address_gap: 0.000
- mean_readout_address_concentration: 0.500

### 3) key_value_memory (kv_easy)

Source: `results/v13a-addressed-kv-easy.csv`

| variant | runs | success_rate | mean_final_max_score | mean_query_accuracy | mean_retrieval_score |
|---|---:|---:|---:|---:|---:|
| stateful_v4_slots | 2 | 0.000 | 5.5 / 12.0 | 0.458 | 0.640 |
| stateful_v5_addressed_slots | 2 | 0.000 | 4.5 / 12.0 | 0.375 | 0.622 |

Addressed metrics (`stateful_v5_addressed_slots`):

- mean_write_address_focus: 0.000
- mean_read_address_focus: 0.000
- mean_write_read_address_gap: 0.000
- mean_readout_address_concentration: 0.500

## Interpretation

- **Engineering status:** feature wiring/export paths are working (new variant runs, archive cells/metrics exported, CLI tables include addressed columns).
- **Research status:** still open. With this sample budget, addressed-slots do not beat slot-retrieval baselines on KV tasks and show weak addressing focus signals (focus metrics ~0, concentration ~0.5).

## Artifacts

- `results/v13a-addressed-bit-memory-guardrail.md`
- `results/v13a-addressed-bit-memory-guardrail.csv`
- `results/v13a-addressed-kv-trivial.md`
- `results/v13a-addressed-kv-trivial.csv`
- `results/v13a-addressed-kv-easy.md`
- `results/v13a-addressed-kv-easy.csv`
