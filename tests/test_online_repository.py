from __future__ import annotations

from db.models import ActiveCandidateRecord
from db.reducers import InMemoryRepository


def test_claim_job_prevents_duplicate_claim() -> None:
    repository = InMemoryRepository(run_id_prefix="jobs")
    run = repository.create_online_run("event_memory", 3, '{"run":{"variant":"stateful"}}')
    candidate = ActiveCandidateRecord(
        candidate_id="c1",
        run_id=run.run_id,
        slot_index=0,
        variant="stateful",
        genome_blob="{}",
        status="created",
        rolling_score=0.0,
        eval_count=0,
        birth_step=0,
        last_eval_at=None,
        parent_ids=[],
        created_at="2026-01-01T00:00:00+00:00",
    )
    repository.seed_active_population(run.run_id, [candidate])
    repository.enqueue_evaluation(run.run_id, candidate.candidate_id, '{"task_name":"event_memory"}')

    first = repository.claim_job(run.run_id, "worker-a")
    second = repository.claim_job(run.run_id, "worker-b")

    assert first is not None
    assert second is None


def test_submit_result_and_rolling_score_update_transition_candidate_back_to_active() -> None:
    repository = InMemoryRepository(run_id_prefix="jobs")
    run = repository.create_online_run("event_memory", 3, '{"run":{"variant":"stateful"}}')
    candidate = ActiveCandidateRecord(
        candidate_id="c1",
        run_id=run.run_id,
        slot_index=0,
        variant="stateful",
        genome_blob="{}",
        status="created",
        rolling_score=0.0,
        eval_count=0,
        birth_step=0,
        last_eval_at=None,
        parent_ids=[],
        created_at="2026-01-01T00:00:00+00:00",
    )
    repository.seed_active_population(run.run_id, [candidate])
    job = repository.enqueue_evaluation(run.run_id, candidate.candidate_id, '{"task_name":"event_memory"}')
    claimed = repository.claim_job(run.run_id, "worker-a")
    assert claimed is not None

    repository.submit_result(job.job_id, candidate.candidate_id, 3.5, '{"success": true}')
    updated = repository.update_candidate_rolling_score(candidate.candidate_id, 3.5, 1, "2026-01-01T00:00:01+00:00")

    assert updated.status == "active"
    assert updated.rolling_score == 3.5
    assert updated.eval_count == 1
