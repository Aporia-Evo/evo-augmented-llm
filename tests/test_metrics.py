from __future__ import annotations

from evolve.rolling_metrics import build_metric_snapshot, success_window_append
from utils.scoring import update_exponential_rolling_score
from db.models import ActiveCandidateRecord


def test_rolling_score_update_uses_exponential_average() -> None:
    first = update_exponential_rolling_score(previous_score=0.0, new_score=3.0, alpha=0.3, eval_count=0)
    second = update_exponential_rolling_score(previous_score=first, new_score=1.0, alpha=0.3, eval_count=1)

    assert first == 3.0
    assert round(second, 6) == 2.4


def test_success_window_and_metric_snapshot_are_consistent() -> None:
    window = []
    for value in (True, False, True, True):
        window = success_window_append(window, value, window_size=3)

    candidates = [
        ActiveCandidateRecord("c1", "r1", 0, "stateful", "{}", "active", 3.0, 2, 0, None, [], "2026-01-01"),
        ActiveCandidateRecord("c2", "r1", 1, "stateful", "{}", "active", 1.0, 2, 0, None, [], "2026-01-01"),
    ]
    snapshot = build_metric_snapshot(candidates, window)

    assert window == [False, True, True]
    assert snapshot.active_population_size == 2
    assert snapshot.rolling_best_score == 3.0
    assert snapshot.rolling_avg_score == 2.0
    assert round(snapshot.success_rate_window, 6) == round(2 / 3, 6)
