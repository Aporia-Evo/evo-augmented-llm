from __future__ import annotations

from db.reducers import InMemoryRepository


def test_commit_generation_ranks_top_k_and_aggregates_scores() -> None:
    repository = InMemoryRepository(run_id_prefix="test")
    run = repository.create_run(
        task_name="xor",
        seed=1,
        config_json='{"run":{"elite_top_k":2}}',
    )
    repository.insert_population(
        run.run_id,
        0,
        [
            _candidate(run.run_id, 0, "c0", '{"id":0}'),
            _candidate(run.run_id, 0, "c1", '{"id":1}'),
            _candidate(run.run_id, 0, "c2", '{"id":2}'),
        ],
    )
    repository.record_fitness("c0", run.run_id, 0, 1.0, '{"mse":1.0}')
    repository.record_fitness("c1", run.run_id, 0, 3.0, '{"mse":0.2}')
    repository.record_fitness("c2", run.run_id, 0, 2.0, '{"mse":0.4}')
    repository.mark_generation_ready(run.run_id, 0)

    committed = repository.commit_generation(run.run_id, 0)

    assert committed.generation.best_candidate_id == "c1"
    assert committed.generation.best_score == 3.0
    assert committed.generation.avg_score == 2.0
    assert [elite.candidate_id for elite in committed.elites] == ["c1", "c2"]


def _candidate(run_id: str, generation_id: int, candidate_id: str, genome_blob: str):
    from db.models import CandidateRecord

    return CandidateRecord(
        candidate_id=candidate_id,
        run_id=run_id,
        generation_id=generation_id,
        genome_blob=genome_blob,
        status="created",
        parent_ids=[],
        created_at="2026-03-29T00:00:00+00:00",
    )

