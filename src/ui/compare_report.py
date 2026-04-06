from __future__ import annotations

import json
from dataclasses import dataclass

from db.models import RunRecord
from db.online_repository import OnlineRunRepository
from utils.scoring import resolve_success


@dataclass(frozen=True)
class OnlineCompareSummary:
    run_id: str
    variant: str
    seed: int
    task_name: str
    status: str
    steps: int
    success_observed: bool
    rolling_best_score: float
    rolling_avg_score: float
    replacement_count: int
    success_rate_window: float
    time_to_first_success: int | None
    replacements_until_first_success: int | None
    hall_of_fame_size: int
    hall_of_fame_growth: int
    resumed_jobs: int


@dataclass(frozen=True)
class OnlineBenchmarkAggregate:
    variant: str
    run_count: int
    run_success_rate: float
    mean_final_best_score: float
    mean_final_rolling_avg_score: float
    mean_success_rate_window: float
    mean_replacement_count: float
    mean_time_to_first_success: float | None
    mean_replacements_until_first_success: float | None
    mean_hall_of_fame_size: float
    mean_hall_of_fame_growth: float


def build_online_compare_summary(
    repository: OnlineRunRepository,
    run: RunRecord,
    score_ceiling: float,
) -> OnlineCompareSummary:
    metrics = repository.list_online_metrics(run.run_id, limit=1)
    latest_metric = metrics[0] if metrics else None
    results = list(reversed(repository.list_evaluation_results(run.run_id, limit=5000)))
    time_to_first_success = None
    first_success_timestamp = None
    for index, result in enumerate(results, start=1):
        if resolve_success(result.raw_metrics, result.score, score_ceiling):
            time_to_first_success = index
            first_success_timestamp = result.created_at
            break
    resume_events = repository.list_events(run.run_id, limit=500)
    resumed_jobs = sum(1 for event in resume_events if event.type == "run_resumed")
    config_payload = json.loads(run.config_json)
    variant = config_payload.get("run", {}).get("variant", "stateful")
    state = repository.get_online_state(run.run_id)
    hall_of_fame_size = len(repository.list_hall_of_fame(run.run_id, limit=5000))
    lifecycle_events = repository.list_candidate_lifecycle_events(run.run_id, limit=5000)
    replacements_until_first_success = _replacements_until_first_success(
        lifecycle_events=lifecycle_events,
        first_success_timestamp=first_success_timestamp,
    )
    return OnlineCompareSummary(
        run_id=run.run_id,
        variant=variant,
        seed=run.seed,
        task_name=run.task_name,
        status=run.status,
        steps=state.step if state is not None else 0,
        success_observed=time_to_first_success is not None,
        rolling_best_score=latest_metric.rolling_best_score if latest_metric is not None else 0.0,
        rolling_avg_score=latest_metric.rolling_avg_score if latest_metric is not None else 0.0,
        replacement_count=latest_metric.replacement_count if latest_metric is not None else 0,
        success_rate_window=latest_metric.success_rate_window if latest_metric is not None else 0.0,
        time_to_first_success=time_to_first_success,
        replacements_until_first_success=replacements_until_first_success,
        hall_of_fame_size=hall_of_fame_size,
        hall_of_fame_growth=hall_of_fame_size,
        resumed_jobs=resumed_jobs,
    )


def build_online_benchmark_aggregates(summaries: list[OnlineCompareSummary]) -> list[OnlineBenchmarkAggregate]:
    rows: list[OnlineBenchmarkAggregate] = []
    for variant in sorted({summary.variant for summary in summaries}):
        subset = [summary for summary in summaries if summary.variant == variant]
        success_steps = [float(summary.time_to_first_success) for summary in subset if summary.time_to_first_success is not None]
        success_replacements = [
            float(summary.replacements_until_first_success)
            for summary in subset
            if summary.replacements_until_first_success is not None
        ]
        rows.append(
            OnlineBenchmarkAggregate(
                variant=variant,
                run_count=len(subset),
                run_success_rate=(sum(1 for summary in subset if summary.success_observed) / len(subset)),
                mean_final_best_score=sum(summary.rolling_best_score for summary in subset) / len(subset),
                mean_final_rolling_avg_score=sum(summary.rolling_avg_score for summary in subset) / len(subset),
                mean_success_rate_window=sum(summary.success_rate_window for summary in subset) / len(subset),
                mean_replacement_count=sum(summary.replacement_count for summary in subset) / len(subset),
                mean_time_to_first_success=(sum(success_steps) / len(success_steps)) if success_steps else None,
                mean_replacements_until_first_success=(
                    (sum(success_replacements) / len(success_replacements))
                    if success_replacements
                    else None
                ),
                mean_hall_of_fame_size=sum(summary.hall_of_fame_size for summary in subset) / len(subset),
                mean_hall_of_fame_growth=sum(summary.hall_of_fame_growth for summary in subset) / len(subset),
            )
        )
    return rows


def _replacements_until_first_success(
    *,
    lifecycle_events: list,
    first_success_timestamp: str | None,
) -> int | None:
    if first_success_timestamp is None:
        return None
    return sum(
        1
        for event in lifecycle_events
        if event.event_type == "candidate_retired" and event.created_at <= first_success_timestamp
    )
