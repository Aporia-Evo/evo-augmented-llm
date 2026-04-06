from __future__ import annotations

from dataclasses import replace

import pytest

from config import AppConfig, EvolutionConfig, OnlineConfig, RunConfig, TaskConfig
from db.reducers import InMemoryRepository
from evolve.online_loop import execute_online_run


class CrashAtOnlineStage:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.run_id: str | None = None
        self.triggered = False

    def on_run_started(self, run) -> None:
        self.run_id = run.run_id

    def on_jobs_queued(self, run, step: int, queued_job_count: int) -> None:
        del run, step
        if self.stage == "queued" and queued_job_count > 0:
            self._crash("simulated online crash at queued")

    def on_job_claimed(self, job, candidate) -> None:
        del job, candidate
        if self.stage == "claimed":
            self._crash("simulated online crash at claimed")

    def on_result_submitted(self, job, result, candidate) -> None:
        del job, result, candidate
        if self.stage == "submitted":
            self._crash("simulated online crash at submitted")

    def on_job_finished(self, job, result, candidate) -> None:
        del job, result, candidate

    def on_candidate_promoted(self, candidate, hall_of_fame_entry) -> None:
        del candidate, hall_of_fame_entry
        if self.stage == "promoted":
            self._crash("simulated online crash at promoted")

    def on_candidate_retired(self, retired_candidate) -> None:
        del retired_candidate
        if self.stage == "retired":
            self._crash("simulated online crash at retired")

    def on_replacement(self, retired_candidate, offspring_candidate, hall_of_fame_entry) -> None:
        del retired_candidate, offspring_candidate, hall_of_fame_entry

    def on_metrics(self, metric) -> None:
        del metric

    def on_run_finished(self, run) -> None:
        del run

    def _crash(self, message: str) -> None:
        if self.triggered:
            return
        self.triggered = True
        raise RuntimeError(message)


def test_online_resume_requeues_claimed_jobs_and_finishes_cleanly() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="event_memory", activation_steps=4, temporal_delay_steps=2),
        run=replace(RunConfig(), seed=9, mode="online", run_id_prefix="online-resume"),
        online=replace(OnlineConfig(), active_population_size=6, max_steps=10, replacement_interval=3, metrics_interval=2),
        evolution=replace(EvolutionConfig(), population_size=6, max_nodes=14, max_conns=24),
    )
    repository = InMemoryRepository(run_id_prefix=config.run.run_id_prefix)
    observer = CrashAtOnlineStage(stage="claimed")

    with pytest.raises(RuntimeError, match="simulated online crash at claimed"):
        execute_online_run(config=config, repository=repository, observer=observer)

    assert observer.run_id is not None
    state_before = repository.get_online_state(observer.run_id)
    assert state_before is not None

    resumed = execute_online_run(
        config=config,
        repository=repository,
        resume_run_id=observer.run_id,
    )

    assert resumed.run.status == "finished"
    _assert_online_run_consistent(repository, observer.run_id, config.online.active_population_size)


@pytest.mark.parametrize(
    ("stage", "crash_message"),
    [
        ("submitted", "simulated online crash at submitted"),
        ("retired", "simulated online crash at retired"),
        ("promoted", "simulated online crash at promoted"),
    ],
)
def test_online_resume_recovers_from_partial_online_crash_points(stage: str, crash_message: str) -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="event_memory", activation_steps=4, temporal_delay_steps=2),
        run=replace(RunConfig(), seed=17, mode="online", run_id_prefix=f"online-resume-{stage}"),
        online=replace(OnlineConfig(), active_population_size=6, max_steps=10, replacement_interval=3, metrics_interval=2),
        evolution=replace(EvolutionConfig(), population_size=6, max_nodes=14, max_conns=24),
    )
    repository = InMemoryRepository(run_id_prefix=config.run.run_id_prefix)
    observer = CrashAtOnlineStage(stage=stage)

    with pytest.raises(RuntimeError, match=crash_message):
        execute_online_run(config=config, repository=repository, observer=observer)

    assert observer.run_id is not None
    state_before = repository.get_online_state(observer.run_id)
    assert state_before is not None
    hall_of_fame_before = _hall_of_fame_snapshot(repository, observer.run_id)
    results_before = repository.list_evaluation_results(observer.run_id, limit=1000)

    if stage == "submitted":
        assert len(results_before) > state_before.step
        assert sum(candidate.eval_count for candidate in repository.list_active_candidates(observer.run_id)) < len(results_before)
    if stage == "retired":
        active_before = [
            candidate
            for candidate in repository.list_active_candidates(observer.run_id)
            if candidate.status != "retired"
        ]
        assert len(active_before) == config.online.active_population_size - 1

    resumed = execute_online_run(
        config=config,
        repository=repository,
        resume_run_id=observer.run_id,
    )

    assert resumed.run.status == "finished"
    _assert_online_run_consistent(repository, observer.run_id, config.online.active_population_size)
    _assert_hall_of_fame_snapshot_preserved(repository, observer.run_id, hall_of_fame_before)


def _assert_online_run_consistent(repository: InMemoryRepository, run_id: str, active_population_size: int) -> None:
    all_candidates = repository.list_active_candidates(run_id)
    active_candidates = [candidate for candidate in all_candidates if candidate.status != "retired"]
    assert len(active_candidates) == active_population_size
    assert len({candidate.slot_index for candidate in active_candidates}) == active_population_size

    jobs = repository.list_evaluation_jobs(run_id, limit=1000)
    assert all(job.status != "claimed" for job in jobs)
    retired_ids = {candidate.candidate_id for candidate in all_candidates if candidate.status == "retired"}
    assert not any(job.status in {"queued", "claimed"} and job.candidate_id in retired_ids for job in jobs)

    results = repository.list_evaluation_results(run_id, limit=1000)
    assert sum(candidate.eval_count for candidate in all_candidates) == len(results)


def _hall_of_fame_snapshot(repository: InMemoryRepository, run_id: str) -> list[tuple[str, str, float, str]]:
    return [
        (entry.entry_id, entry.candidate_id, entry.score, entry.frozen_genome_blob)
        for entry in sorted(
            repository.list_hall_of_fame(run_id, limit=200),
            key=lambda entry: (entry.inserted_at, entry.entry_id),
        )
    ]


def _assert_hall_of_fame_snapshot_preserved(
    repository: InMemoryRepository,
    run_id: str,
    expected_entries: list[tuple[str, str, float, str]],
) -> None:
    after_entries = {
        entry.entry_id: (entry.entry_id, entry.candidate_id, entry.score, entry.frozen_genome_blob)
        for entry in repository.list_hall_of_fame(run_id, limit=200)
    }
    assert [after_entries[entry_id] for entry_id, *_ in expected_entries] == expected_entries
