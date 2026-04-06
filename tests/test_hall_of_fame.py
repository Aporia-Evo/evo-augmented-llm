from __future__ import annotations

from db.models import ActiveCandidateRecord
from db.reducers import InMemoryRepository


def test_hall_of_fame_stores_snapshot_not_reference() -> None:
    repository = InMemoryRepository(run_id_prefix="hof")
    run = repository.create_online_run("event_memory", 7, '{"run":{"variant":"stateful"}}')
    candidate = ActiveCandidateRecord(
        candidate_id="c1",
        run_id=run.run_id,
        slot_index=0,
        variant="stateful",
        genome_blob='{"version":1}',
        status="active",
        rolling_score=4.0,
        eval_count=3,
        birth_step=0,
        last_eval_at=None,
        parent_ids=[],
        created_at="2026-01-01T00:00:00+00:00",
    )
    repository.seed_active_population(run.run_id, [candidate])

    entry = repository.promote_to_hall_of_fame(candidate.candidate_id)
    assert entry is not None

    repository.active_candidates[candidate.candidate_id] = ActiveCandidateRecord(
        **{**candidate.__dict__, "genome_blob": '{"version":2}'}
    )

    hall_of_fame_rows = repository.list_hall_of_fame(run.run_id)
    assert hall_of_fame_rows[0].frozen_genome_blob == '{"version":1}'
