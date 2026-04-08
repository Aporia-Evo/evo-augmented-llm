# Results Directory Guide

This folder contains experiment notes and benchmark exports.

## Structure

- `v13a/` — all artifacts related to addressed-slot workstream (`stateful_v5_addressed_slots`), including benchmark-suite exports and consolidated interpretations.
- `v*` files in root — historical snapshots from earlier phases.
- `generation-suite-*.md` — timestamped suite exports.

## Current V13a Canonical Files

- `results/v13a/v13a-addressed-evidence.md`
- `results/v13a/v13a-performance-test-2026-04-08.md`
- `results/v13a/v13a-addressed-bit-memory-guardrail.md`
- `results/v13a/v13a-addressed-kv-trivial.md`
- `results/v13a/v13a-addressed-kv-easy.md`
- `results/v13a/v13a-perf-bit-memory-s1-5-g12-p40.md`
- `results/v13a/v13a-perf-kv-trivial-s1-5-g12-p40.md`
- `results/v13a/v13a-perf-kv-easy-s1-5-g12-p40.md`

## Naming Convention (going forward)

- Keep one workstream per subdirectory (`results/<track>/...`).
- Keep one short evidence summary file per track (`<track>-evidence.md`).
- Keep one larger performance summary file per track/date (`<track>-performance-test-YYYY-MM-DD.md`).
