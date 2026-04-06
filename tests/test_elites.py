from __future__ import annotations

from dataclasses import asdict

from db.models import CandidateRecord
from db.reducers import InMemoryRepository


def test_elite_archive_stores_snapshot_not_reference() -> None:
    repository = InMemoryRepository(run_id_prefix="test")
    run = repository.create_run(
        task_name="xor",
        seed=1,
        config_json='{"run":{"elite_top_k":1}}',
    )
    candidate = CandidateRecord(
        candidate_id="c0",
        run_id=run.run_id,
        generation_id=0,
        genome_blob='{"genome":"original"}',
        status="created",
        parent_ids=[],
        created_at="2026-03-29T00:00:00+00:00",
    )
    repository.insert_population(run.run_id, 0, [candidate])
    repository.record_fitness("c0", run.run_id, 0, 2.5, '{"mse":0.1}')
    repository.mark_generation_ready(run.run_id, 0)
    committed = repository.commit_generation(run.run_id, 0)

    repository.candidates["c0"] = CandidateRecord(
        **{**asdict(repository.candidates["c0"]), "genome_blob": '{"genome":"mutated"}'}
    )

    assert committed.elites[0].frozen_genome_blob == '{"genome":"original"}'

