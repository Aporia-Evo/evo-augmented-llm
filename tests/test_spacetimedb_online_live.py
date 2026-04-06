from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from config import AppConfig, EvolutionConfig, OnlineConfig, RunConfig, TaskConfig
from db.client import SpacetimeHttpClient
from db.reducers import SpacetimeRepository
from evolve.online_loop import execute_online_run


class CrashAtLiveOnlineStage:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.run_id: str | None = None
        self.triggered = False

    def on_run_started(self, run) -> None:
        self.run_id = run.run_id

    def on_job_claimed(self, job, candidate) -> None:
        del job, candidate
        if self.stage == "claimed":
            self._crash("simulated live online crash at claimed")

    def on_result_submitted(self, job, result, candidate) -> None:
        del job, result, candidate
        if self.stage == "submitted":
            self._crash("simulated live online crash at submitted")

    def on_candidate_retired(self, retired_candidate) -> None:
        del retired_candidate
        if self.stage == "retired":
            self._crash("simulated live online crash at retired")

    def on_job_finished(self, job, result, candidate) -> None:
        del job, result, candidate

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


@pytest.mark.parametrize(
    ("stage", "crash_message"),
    [
        ("claimed", "simulated live online crash at claimed"),
        ("submitted", "simulated live online crash at submitted"),
        ("retired", "simulated live online crash at retired"),
    ],
)
def test_live_spacetimedb_online_resume_recovers_critical_crash_points(stage: str, crash_message: str) -> None:
    if shutil.which("spacetime") is None:
        pytest.skip("spacetime CLI not available")
    if shutil.which("npm") is None:
        pytest.skip("npm not available")

    repo_root = Path(__file__).resolve().parents[1]
    module_dir = repo_root / "spacetimedb"
    if not (module_dir / "node_modules").exists():
        subprocess.run(["npm", "install", "--no-bin-links"], cwd=module_dir, check=True)

    process: subprocess.Popen[bytes] | None = None
    if not _port_open("127.0.0.1", 3000):
        process = subprocess.Popen(
            ["spacetime", "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_port("127.0.0.1", 3000, timeout_seconds=15.0)

    try:
        cli_config = Path.home() / ".config" / "spacetime" / "cli.toml"
        if not cli_config.exists():
            subprocess.run(
                ["spacetime", "login", "--server-issued-login", "http://127.0.0.1:3000", "--no-browser"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        database_name = f"neat-online-live-{uuid4().hex[:10]}"
        subprocess.run(
            ["spacetime", "publish", "--server", "local", database_name],
            cwd=module_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        config = AppConfig(
            task=replace(TaskConfig(), name="event_memory", activation_steps=4, temporal_delay_steps=2),
            run=replace(
                RunConfig(),
                seed=41,
                variant="stateful",
                mode="online",
                run_id_prefix=f"spacetime-online-live-{stage}",
            ),
            online=replace(
                OnlineConfig(),
                active_population_size=6,
                max_steps=10,
                replacement_interval=3,
                metrics_interval=2,
            ),
            evolution=replace(EvolutionConfig(), population_size=6, max_nodes=14, max_conns=24),
        )
        repository = SpacetimeRepository(
            client=SpacetimeHttpClient(server_url="http://127.0.0.1:3000", database_name=database_name),
            run_id_prefix=config.run.run_id_prefix,
        )

        observer = CrashAtLiveOnlineStage(stage=stage)
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
        _assert_live_online_run_consistent(repository, observer.run_id, config.online.active_population_size)
        _assert_hall_of_fame_snapshot_preserved(repository, observer.run_id, hall_of_fame_before)

        resume_events = [
            event
            for event in repository.list_events(observer.run_id, limit=500)
            if event.type == "run_resumed"
        ]
        assert len(resume_events) == 1
        payload = json.loads(resume_events[0].payload_json)
        assert payload["mode"] == "online"
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=10)


def _assert_live_online_run_consistent(repository: SpacetimeRepository, run_id: str, active_population_size: int) -> None:
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


def _hall_of_fame_snapshot(repository: SpacetimeRepository, run_id: str) -> list[tuple[str, str, float, str]]:
    return [
        (entry.entry_id, entry.candidate_id, entry.score, entry.frozen_genome_blob)
        for entry in sorted(
            repository.list_hall_of_fame(run_id, limit=200),
            key=lambda entry: (entry.inserted_at, entry.entry_id),
        )
    ]


def _assert_hall_of_fame_snapshot_preserved(
    repository: SpacetimeRepository,
    run_id: str,
    expected_entries: list[tuple[str, str, float, str]],
) -> None:
    after_entries = {
        entry.entry_id: (entry.entry_id, entry.candidate_id, entry.score, entry.frozen_genome_blob)
        for entry in repository.list_hall_of_fame(run_id, limit=200)
    }
    assert [after_entries[entry_id] for entry_id, *_ in expected_entries] == expected_entries


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _wait_for_port(host: str, port: int, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _port_open(host, port):
            return
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")
