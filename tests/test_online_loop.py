from __future__ import annotations

from dataclasses import replace

from db.models import ActiveCandidateRecord
from config import AppConfig, EvolutionConfig, OnlineConfig, RunConfig, TaskConfig
from db.reducers import InMemoryRepository
from evolve.online_loop import _maybe_promote_candidate, _notify_replacement_observer, execute_online_run


def test_online_loop_runs_multiple_steps_and_keeps_population_stable() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="event_memory", activation_steps=4, temporal_delay_steps=2),
        run=replace(RunConfig(), seed=5, mode="online", run_id_prefix="online-loop"),
        online=replace(OnlineConfig(), active_population_size=8, max_steps=12, replacement_interval=4, metrics_interval=3),
        evolution=replace(EvolutionConfig(), population_size=8, max_nodes=16, max_conns=28),
    )
    repository = InMemoryRepository(run_id_prefix=config.run.run_id_prefix)

    result = execute_online_run(config=config, repository=repository)

    active_candidates = [candidate for candidate in repository.list_active_candidates(result.run.run_id) if candidate.status != "retired"]
    assert result.step == 12
    assert result.run.status == "finished"
    assert len(active_candidates) == config.online.active_population_size
    assert repository.list_online_metrics(result.run.run_id)


def test_hall_of_fame_bootstrap_requires_success_or_improvement() -> None:
    repository = InMemoryRepository(run_id_prefix="online-hof")
    run = repository.create_online_run("event_memory", 7, '{"run":{"variant":"stateful"}}')
    candidate = ActiveCandidateRecord(
        candidate_id="candidate-0",
        run_id=run.run_id,
        slot_index=0,
        variant="stateful",
        genome_blob='{"version":1}',
        status="active",
        rolling_score=0.0,
        eval_count=1,
        birth_step=0,
        last_eval_at=None,
        parent_ids=[],
        created_at="2026-04-04T00:00:00+00:00",
    )
    repository.seed_active_population(run.run_id, [candidate])

    promoted = _maybe_promote_candidate(repository, candidate, score_ceiling=4.0, success=False)

    assert promoted is None
    assert repository.list_hall_of_fame(run.run_id, limit=10) == []


def test_replacement_observer_does_not_receive_unrelated_promotion(monkeypatch) -> None:
    retired_candidate = ActiveCandidateRecord(
        candidate_id="retired-1",
        run_id="synthetic-run",
        slot_index=0,
        variant="stateful",
        genome_blob='{"version":1}',
        status="retired",
        rolling_score=1.0,
        eval_count=1,
        birth_step=0,
        last_eval_at=None,
        parent_ids=[],
        created_at="2026-04-04T00:00:00+00:00",
    )
    offspring_candidate = ActiveCandidateRecord(
        candidate_id="offspring-1",
        run_id="synthetic-run",
        slot_index=0,
        variant="stateful",
        genome_blob='{"version":2}',
        status="active",
        rolling_score=0.0,
        eval_count=0,
        birth_step=1,
        last_eval_at=None,
        parent_ids=[],
        created_at="2026-04-04T00:00:01+00:00",
    )

    class Observer:
        def __init__(self) -> None:
            self.replacement_entry = "unset"

        def on_replacement(self, retired, offspring, hall_of_fame_entry) -> None:
            del retired, offspring
            self.replacement_entry = hall_of_fame_entry

    observer = Observer()
    _notify_replacement_observer(observer, retired_candidate, offspring_candidate)

    assert observer.replacement_entry is None
