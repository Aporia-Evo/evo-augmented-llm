from __future__ import annotations

from typing import Mapping


def score_success_tolerance(score_ceiling: float) -> float:
    ceiling = float(score_ceiling)
    return max(1e-6, min(0.01, ceiling * 0.0025))


def is_success_score(score: float, score_ceiling: float) -> bool:
    return float(score) >= float(score_ceiling) - score_success_tolerance(score_ceiling)


def resolve_success(
    raw_metrics: Mapping[str, object] | None,
    score: float,
    score_ceiling: float,
) -> bool:
    score_success = is_success_score(score, score_ceiling)
    if raw_metrics is None:
        return score_success
    if "success" not in raw_metrics:
        return score_success
    return bool(raw_metrics["success"]) or score_success


def update_exponential_rolling_score(
    previous_score: float,
    new_score: float,
    alpha: float,
    eval_count: int,
) -> float:
    if eval_count <= 0:
        return float(new_score)
    clamped_alpha = min(1.0, max(0.0, float(alpha)))
    return (clamped_alpha * float(new_score)) + ((1.0 - clamped_alpha) * float(previous_score))
