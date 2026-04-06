from __future__ import annotations

from dataclasses import replace

import pytest

from config import AppConfig, EvolutionConfig, RunConfig
from db.models import CandidateRecord
from db.reducers import InMemoryRepository
from evolve.run_loop import _next_first_success_generation, _scores_for_source_candidates, execute_run


def test_run_loop_produces_multiple_committed_generations() -> None:
    config = AppConfig(
        run=replace(RunConfig(), generations=10, seed=5, elite_top_k=2, run_id_prefix="testrun"),
        evolution=replace(EvolutionConfig(), population_size=16, max_nodes=14, max_conns=24),
    )
    repository = InMemoryRepository(run_id_prefix=config.run.run_id_prefix)

    result = execute_run(config=config, repository=repository)

    generations = repository.list_generations(result.run.run_id)
    assert len(generations) == 10
    assert all(generation.state == "committed" for generation in generations)
    assert all(generation.best_score is not None for generation in generations)
    assert repository.list_elites(result.run.run_id, limit=50)


def test_first_success_generation_keeps_earliest_success_generation() -> None:
    first = _next_first_success_generation(current=None, generation_id=3, success=True)
    later_success = _next_first_success_generation(current=first, generation_id=7, success=True)
    failure = _next_first_success_generation(current=later_success, generation_id=9, success=False)

    assert first == 3
    assert later_success == 3
    assert failure == 3


def test_run_loop_raises_clear_error_when_fitness_record_is_missing() -> None:
    candidates = [
        CandidateRecord(
            candidate_id="c-0",
            run_id="run-1",
            generation_id=0,
            genome_blob='{"id":0}',
            status="created",
            parent_ids=[],
            created_at="2026-04-04T00:00:00+00:00",
        ),
        CandidateRecord(
            candidate_id="c-1",
            run_id="run-1",
            generation_id=0,
            genome_blob='{"id":1}',
            status="created",
            parent_ids=[],
            created_at="2026-04-04T00:00:00+00:00",
        ),
    ]

    with pytest.raises(RuntimeError, match="Missing fitness records for generation 0: c-1"):
        _scores_for_source_candidates(
            source_candidates=candidates,
            fitness_by_candidate={"c-0": 1.0},
            generation_id=0,
        )
