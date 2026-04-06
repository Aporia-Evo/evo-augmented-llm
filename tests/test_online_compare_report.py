from __future__ import annotations

from db.models import (
    CandidateLifecycleEventRecord,
    EvaluationResultRecord,
    HallOfFameEntryRecord,
    OnlineMetricRecord,
    OnlineStateRecord,
)
from db.reducers import InMemoryRepository
from ui.cli import _pair_online_seed_summaries
from ui.compare_report import OnlineCompareSummary, build_online_benchmark_aggregates, build_online_compare_summary


def test_build_online_compare_summary_tracks_first_success_and_replacements() -> None:
    repository = InMemoryRepository(run_id_prefix="online-compare")
    run = repository.create_online_run(
        "event_memory",
        7,
        '{"task":{"name":"event_memory","temporal_delay_steps":2},"run":{"variant":"stateful"}}',
    )
    repository.online_states[run.run_id] = OnlineStateRecord(
        run_id=run.run_id,
        step=8,
        replacement_count=3,
        success_window_json="[true, true, false]",
        adapter_state_blob="{}",
        created_at="2026-03-30T00:00:00+00:00",
        updated_at="2026-03-30T00:00:08+00:00",
    )
    repository.online_metrics[f"{run.run_id}-metric"] = OnlineMetricRecord(
        metric_id=f"{run.run_id}-metric",
        run_id=run.run_id,
        timestamp="2026-03-30T00:00:08+00:00",
        active_population_size=6,
        rolling_best_score=3.8,
        rolling_avg_score=2.7,
        replacement_count=3,
        success_rate_window=0.5,
    )
    repository.evaluation_results[f"{run.run_id}-result-1"] = EvaluationResultRecord(
        result_id=f"{run.run_id}-result-1",
        run_id=run.run_id,
        candidate_id="c0",
        score=2.0,
        raw_metrics={"success": False},
        created_at="2026-03-30T00:00:02+00:00",
    )
    repository.evaluation_results[f"{run.run_id}-result-2"] = EvaluationResultRecord(
        result_id=f"{run.run_id}-result-2",
        run_id=run.run_id,
        candidate_id="c1",
        score=3.9999,
        raw_metrics={"success": False},
        created_at="2026-03-30T00:00:06+00:00",
    )
    repository.candidate_lifecycle_events[f"{run.run_id}-ev-1"] = CandidateLifecycleEventRecord(
        event_id=f"{run.run_id}-ev-1",
        run_id=run.run_id,
        candidate_id="c0",
        event_type="candidate_retired",
        payload_json="{}",
        created_at="2026-03-30T00:00:03+00:00",
    )
    repository.candidate_lifecycle_events[f"{run.run_id}-ev-2"] = CandidateLifecycleEventRecord(
        event_id=f"{run.run_id}-ev-2",
        run_id=run.run_id,
        candidate_id="c1",
        event_type="candidate_retired",
        payload_json="{}",
        created_at="2026-03-30T00:00:05+00:00",
    )
    repository.hall_of_fame[f"{run.run_id}-hof-1"] = HallOfFameEntryRecord(
        entry_id=f"{run.run_id}-hof-1",
        run_id=run.run_id,
        candidate_id="c1",
        score=3.9999,
        frozen_genome_blob="{}",
        inserted_at="2026-03-30T00:00:06+00:00",
    )

    summary = build_online_compare_summary(repository, run, score_ceiling=4.0)

    assert summary.success_observed is True
    assert summary.time_to_first_success == 2
    assert summary.replacements_until_first_success == 2
    assert summary.rolling_best_score == 3.8
    assert summary.rolling_avg_score == 2.7
    assert summary.hall_of_fame_size == 1
    assert summary.hall_of_fame_growth == 1


def test_online_benchmark_aggregates_and_seed_pairs_include_new_metrics() -> None:
    summaries = [
        OnlineCompareSummary(
            run_id="r1",
            variant="stateful",
            seed=7,
            task_name="event_memory",
            status="finished",
            steps=12,
            success_observed=True,
            rolling_best_score=3.5,
            rolling_avg_score=2.4,
            replacement_count=3,
            success_rate_window=0.5,
            time_to_first_success=4,
            replacements_until_first_success=1,
            hall_of_fame_size=2,
            hall_of_fame_growth=2,
            resumed_jobs=0,
        ),
        OnlineCompareSummary(
            run_id="r2",
            variant="stateless",
            seed=7,
            task_name="event_memory",
            status="finished",
            steps=12,
            success_observed=False,
            rolling_best_score=2.2,
            rolling_avg_score=2.0,
            replacement_count=3,
            success_rate_window=0.0,
            time_to_first_success=None,
            replacements_until_first_success=None,
            hall_of_fame_size=1,
            hall_of_fame_growth=1,
            resumed_jobs=0,
        ),
    ]

    aggregates = build_online_benchmark_aggregates(summaries)
    stateful = next(aggregate for aggregate in aggregates if aggregate.variant == "stateful")
    stateless = next(aggregate for aggregate in aggregates if aggregate.variant == "stateless")
    pairs = _pair_online_seed_summaries(summaries)

    assert stateful.run_success_rate == 1.0
    assert stateful.mean_final_best_score == 3.5
    assert stateful.mean_final_rolling_avg_score == 2.4
    assert stateful.mean_replacements_until_first_success == 1.0
    assert stateful.mean_hall_of_fame_growth == 2.0
    assert stateless.run_success_rate == 0.0
    assert stateless.mean_replacements_until_first_success is None
    assert pairs == [
        {
            "seed": 7,
            "stateful_success_observed": True,
            "stateless_success_observed": False,
            "stateful_final_best_score": 3.5,
            "stateless_final_best_score": 2.2,
            "stateful_time_to_first_success": 4,
            "stateless_time_to_first_success": None,
            "stateful_replacements_until_first_success": 1,
            "stateless_replacements_until_first_success": None,
        }
    ]
