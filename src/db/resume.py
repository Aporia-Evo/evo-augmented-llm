from __future__ import annotations

from db.models import ActiveCandidateRecord, OnlineStateRecord
from db.online_repository import OnlineRunRepository


def load_online_resume_snapshot(
    repository: OnlineRunRepository,
    run_id: str,
) -> tuple[OnlineStateRecord, list[ActiveCandidateRecord]]:
    state = repository.get_online_state(run_id)
    if state is None:
        raise RuntimeError(f"Missing online state for run {run_id}")
    active_candidates = repository.list_active_candidates(
        run_id,
        statuses=["created", "queued", "evaluating", "active"],
    )
    if not active_candidates:
        raise RuntimeError(f"Missing active candidates for run {run_id}")
    return state, active_candidates
