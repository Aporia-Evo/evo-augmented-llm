from __future__ import annotations

from argparse import Namespace

from db.models import CandidateRecord, FitnessRecord, GenerationRecord
from db.reducers import InMemoryRepository
from ui.cli import _build_compare_summary, _resolve_compare_runs


def test_compare_summary_reports_resume_hint_and_first_max_generation() -> None:
    repository = InMemoryRepository(run_id_prefix="compare")
    run = repository.create_run(
        task_name="delayed_xor",
        seed=7,
        config_json='{"run":{"variant":"stateful"}}',
    )
    repository.generations[(run.run_id, 0)] = GenerationRecord(
        run_id=run.run_id,
        generation_id=0,
        state="committed",
        best_candidate_id="c0",
        best_score=2.0,
        avg_score=5.0,
        created_at="2026-03-29T00:00:00+00:00",
        committed_at="2026-03-29T00:00:01+00:00",
        eval_duration_ms=11,
        commit_duration_ms=3,
    )
    repository.generations[(run.run_id, 1)] = GenerationRecord(
        run_id=run.run_id,
        generation_id=1,
        state="committed",
        best_candidate_id="c1",
        best_score=3.0,
        avg_score=8.0,
        created_at="2026-03-29T00:00:02+00:00",
        committed_at="2026-03-29T00:00:03+00:00",
        eval_duration_ms=9,
        commit_duration_ms=4,
    )
    repository.generations[(run.run_id, 2)] = GenerationRecord(
        run_id=run.run_id,
        generation_id=2,
        state="committed",
        best_candidate_id="c2",
        best_score=3.0,
        avg_score=9.0,
        created_at="2026-03-29T00:00:04+00:00",
        committed_at="2026-03-29T00:00:05+00:00",
        eval_duration_ms=8,
        commit_duration_ms=5,
    )
    repository.append_event(run.run_id, "run_resumed", '{"next_generation_id":2}')

    summary = _build_compare_summary(repository, run)  # type: ignore[arg-type]

    assert summary.first_max_generation == 1
    assert summary.first_success_generation is None
    assert summary.final_max_score == 3.0
    assert summary.score_ceiling == 4.0
    assert summary.delay_steps == 1
    assert summary.success is False
    assert summary.avg_score_path == "5.000->8.000->9.000"
    assert summary.resume_hint == "resumed x1 next_generation=2"


def test_compare_summary_prefers_recorded_success_over_score_proximity() -> None:
    repository = InMemoryRepository(run_id_prefix="compare")
    run = repository.create_run(
        task_name="bit_memory",
        seed=11,
        config_json='{"task":{"name":"bit_memory","temporal_delay_steps":3},"run":{"variant":"stateful"}}',
    )
    repository.generations[(run.run_id, 0)] = GenerationRecord(
        run_id=run.run_id,
        generation_id=0,
        state="committed",
        best_candidate_id="c0",
        best_score=3.97,
        avg_score=2.5,
        created_at="2026-03-29T00:00:00+00:00",
        committed_at="2026-03-29T00:00:01+00:00",
        eval_duration_ms=10,
        commit_duration_ms=2,
    )
    repository.candidates["c0"] = CandidateRecord(
        candidate_id="c0",
        run_id=run.run_id,
        generation_id=0,
        genome_blob="{}",
        status="evaluated",
        parent_ids=[],
        created_at="2026-03-29T00:00:00+00:00",
    )
    repository.fitness["c0"] = FitnessRecord(
        candidate_id="c0",
        run_id=run.run_id,
        generation_id=0,
        score=3.97,
        raw_metrics={"success": True},
        evaluated_at="2026-03-29T00:00:00+00:00",
    )

    summary = _build_compare_summary(repository, run)  # type: ignore[arg-type]

    assert summary.success is True
    assert summary.first_success_generation == 0
    assert summary.score_ceiling == 4.0


def test_compare_summary_treats_near_ceiling_score_as_success_even_if_metric_flag_is_false() -> None:
    repository = InMemoryRepository(run_id_prefix="compare")
    run = repository.create_run(
        task_name="bit_memory",
        seed=13,
        config_json='{"task":{"name":"bit_memory","temporal_delay_steps":5},"run":{"variant":"stateful"}}',
    )
    repository.generations[(run.run_id, 0)] = GenerationRecord(
        run_id=run.run_id,
        generation_id=0,
        state="committed",
        best_candidate_id="c0",
        best_score=3.993196,
        avg_score=2.9,
        created_at="2026-03-30T00:00:00+00:00",
        committed_at="2026-03-30T00:00:01+00:00",
        eval_duration_ms=10,
        commit_duration_ms=2,
    )
    repository.candidates["c0"] = CandidateRecord(
        candidate_id="c0",
        run_id=run.run_id,
        generation_id=0,
        genome_blob="{}",
        status="evaluated",
        parent_ids=[],
        created_at="2026-03-30T00:00:00+00:00",
    )
    repository.fitness["c0"] = FitnessRecord(
        candidate_id="c0",
        run_id=run.run_id,
        generation_id=0,
        score=3.993196,
        raw_metrics={"success": False},
        evaluated_at="2026-03-30T00:00:00+00:00",
    )

    summary = _build_compare_summary(repository, run)  # type: ignore[arg-type]

    assert summary.success is True
    assert summary.first_success_generation == 0


def test_compare_resolution_finds_stateful_and_stateless_by_seed_and_task() -> None:
    repository = InMemoryRepository(run_id_prefix="compare")
    stateful = repository.create_run(
        task_name="delayed_xor",
        seed=5,
        config_json='{"run":{"variant":"stateful"}}',
    )
    stateless = repository.create_run(
        task_name="delayed_xor",
        seed=5,
        config_json='{"run":{"variant":"stateless"}}',
    )

    args = Namespace(
        stateful_run_id=None,
        stateless_run_id=None,
        seed=5,
        task_name="delayed_xor",
        search_limit=20,
    )
    resolved_stateful, resolved_stateless = _resolve_compare_runs(repository, args)  # type: ignore[arg-type]

    assert resolved_stateful is not None
    assert resolved_stateless is not None
    assert resolved_stateful.run_id == stateful.run_id
    assert resolved_stateless.run_id == stateless.run_id
