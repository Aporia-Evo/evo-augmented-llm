from __future__ import annotations

from dataclasses import replace

import numpy as np

from db.models import GenerationRecord
from config import AppConfig, EvolutionConfig, RunConfig, TaskConfig
from db.reducers import InMemoryRepository
from evolve.evaluator import build_evaluator
from evolve.run_loop import _latest_generation_record, execute_run
from evolve.tensorneat_adapter import TensorNEATAdapter


class CrashAfterCommits:
    def __init__(self, stop_after: int) -> None:
        self.stop_after = stop_after
        self.run_id: str | None = None
        self.commit_count = 0

    def on_run_started(self, run) -> None:
        self.run_id = run.run_id

    def on_generation_committed(self, result) -> None:
        self.commit_count += 1
        if self.commit_count >= self.stop_after:
            raise RuntimeError("simulated crash")

    def on_run_finished(self, run) -> None:
        return None


def test_adapter_state_round_trip_preserves_population() -> None:
    config = AppConfig(run=replace(RunConfig(), seed=13))
    evaluator = build_evaluator(config.task)
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    restored = adapter.deserialize_state(adapter.serialize_state(state))

    original_nodes, original_conns = adapter.ask(state)
    restored_nodes, restored_conns = adapter.ask(restored)

    assert np.allclose(original_nodes, restored_nodes, equal_nan=True)
    assert np.allclose(original_conns, restored_conns, equal_nan=True)


def test_same_seed_reproduces_initial_population() -> None:
    config = AppConfig(run=replace(RunConfig(), seed=23))
    evaluator = build_evaluator(config.task)
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)
    first_state = adapter.initialize(config.run.seed)
    second_state = adapter.initialize(config.run.seed)

    first_nodes, first_conns = adapter.ask(first_state)
    second_nodes, second_conns = adapter.ask(second_state)

    assert np.allclose(first_nodes, second_nodes, equal_nan=True)
    assert np.allclose(first_conns, second_conns, equal_nan=True)


def test_resume_continues_from_last_committed_generation() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="delayed_xor", activation_steps=4),
        run=replace(RunConfig(), generations=6, seed=17, elite_top_k=2, run_id_prefix="resume-test"),
        evolution=replace(EvolutionConfig(), population_size=18, max_nodes=16, max_conns=28),
    )
    repository = InMemoryRepository(run_id_prefix=config.run.run_id_prefix)
    observer = CrashAfterCommits(stop_after=3)

    try:
        execute_run(config=config, repository=repository, observer=observer)
    except RuntimeError as exc:
        assert str(exc) == "simulated crash"
    else:
        raise AssertionError("Expected simulated crash during run.")

    assert observer.run_id is not None
    committed_before_resume = repository.list_generations(observer.run_id)
    assert len(committed_before_resume) == 3
    assert all(generation.state == "committed" for generation in committed_before_resume)

    resumed = execute_run(
        config=config,
        repository=repository,
        resume_run_id=observer.run_id,
    )

    generations = repository.list_generations(resumed.run.run_id)
    assert len(generations) == config.run.generations
    assert all(generation.state == "committed" for generation in generations)
    assert repository.get_checkpoint(resumed.run.run_id, 0) is not None
    assert repository.list_elites(resumed.run.run_id, limit=100)


def test_resume_uses_highest_generation_even_if_repository_order_is_unsorted() -> None:
    generations = [
        GenerationRecord(
            run_id="run-1",
            generation_id=1,
            state="committed",
            best_candidate_id="c-1",
            best_score=1.0,
            avg_score=1.0,
            created_at="2026-04-04T00:00:01+00:00",
            committed_at="2026-04-04T00:00:02+00:00",
        ),
        GenerationRecord(
            run_id="run-1",
            generation_id=4,
            state="committed",
            best_candidate_id="c-4",
            best_score=4.0,
            avg_score=4.0,
            created_at="2026-04-04T00:00:04+00:00",
            committed_at="2026-04-04T00:00:05+00:00",
        ),
        GenerationRecord(
            run_id="run-1",
            generation_id=2,
            state="committed",
            best_candidate_id="c-2",
            best_score=2.0,
            avg_score=2.0,
            created_at="2026-04-04T00:00:02+00:00",
            committed_at="2026-04-04T00:00:03+00:00",
        ),
    ]

    selected = _latest_generation_record(generations)

    assert selected.generation_id == 4
