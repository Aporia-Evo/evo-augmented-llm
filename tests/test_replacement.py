from __future__ import annotations

from db.models import ActiveCandidateRecord
from evolve.replacement import choose_parent_ids, choose_replacement_target


def test_replacement_selects_weakest_evaluated_candidate() -> None:
    candidates = [
        ActiveCandidateRecord("c1", "r1", 0, "stateful", "{}", "active", 3.0, 3, 0, None, [], "2026-01-01"),
        ActiveCandidateRecord("c2", "r1", 1, "stateful", "{}", "active", 1.5, 5, 0, None, [], "2026-01-01"),
        ActiveCandidateRecord("c3", "r1", 2, "stateful", "{}", "active", 2.0, 4, 0, None, [], "2026-01-01"),
    ]

    target = choose_replacement_target(candidates)

    assert target is not None
    assert target.candidate_id == "c2"


def test_parent_selection_prefers_highest_scoring_candidates() -> None:
    candidates = [
        ActiveCandidateRecord("c1", "r1", 0, "stateful", "{}", "active", 5.0, 3, 0, None, [], "2026-01-01"),
        ActiveCandidateRecord("c2", "r1", 1, "stateful", "{}", "active", 4.5, 3, 0, None, [], "2026-01-01"),
        ActiveCandidateRecord("c3", "r1", 2, "stateful", "{}", "active", 1.0, 3, 0, None, [], "2026-01-01"),
    ]

    parent_ids = choose_parent_ids(candidates, exclude_candidate_id="c3")

    assert parent_ids == ["c1", "c2"]
