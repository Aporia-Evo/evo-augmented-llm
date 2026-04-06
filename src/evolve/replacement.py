from __future__ import annotations

from db.models import ActiveCandidateRecord


def choose_replacement_target(candidates: list[ActiveCandidateRecord]) -> ActiveCandidateRecord | None:
    eligible = [candidate for candidate in candidates if candidate.status != "retired" and candidate.eval_count > 0]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda candidate: (
            candidate.rolling_score,
            -candidate.eval_count,
            candidate.birth_step,
            candidate.candidate_id,
        ),
    )[0]


def choose_parent_ids(
    candidates: list[ActiveCandidateRecord],
    exclude_candidate_id: str | None = None,
) -> list[str]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.status != "retired" and candidate.candidate_id != exclude_candidate_id
    ]
    ranked = sorted(
        eligible,
        key=lambda candidate: (
            -candidate.rolling_score,
            -candidate.eval_count,
            candidate.birth_step,
            candidate.candidate_id,
        ),
    )
    if not ranked:
        return []
    if len(ranked) == 1:
        return [ranked[0].candidate_id]
    return [ranked[0].candidate_id, ranked[1].candidate_id]
