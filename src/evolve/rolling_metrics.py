from __future__ import annotations

from dataclasses import dataclass

from db.models import ActiveCandidateRecord


@dataclass(frozen=True)
class RollingMetricSnapshot:
    active_population_size: int
    rolling_best_score: float
    rolling_avg_score: float
    success_rate_window: float


def success_window_append(window: list[bool], success: bool, window_size: int) -> list[bool]:
    updated = [*window, bool(success)]
    if window_size <= 0:
        return updated
    return updated[-window_size:]


def success_rate_window(window: list[bool]) -> float:
    if not window:
        return 0.0
    return sum(1.0 for value in window if value) / float(len(window))


def build_metric_snapshot(
    active_candidates: list[ActiveCandidateRecord],
    success_window_values: list[bool],
) -> RollingMetricSnapshot:
    if not active_candidates:
        return RollingMetricSnapshot(
            active_population_size=0,
            rolling_best_score=0.0,
            rolling_avg_score=0.0,
            success_rate_window=success_rate_window(success_window_values),
        )
    scores = [candidate.rolling_score for candidate in active_candidates]
    return RollingMetricSnapshot(
        active_population_size=len(active_candidates),
        rolling_best_score=max(scores),
        rolling_avg_score=sum(scores) / len(scores),
        success_rate_window=success_rate_window(success_window_values),
    )
