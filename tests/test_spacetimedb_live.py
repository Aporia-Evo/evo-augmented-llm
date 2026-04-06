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

from config import AppConfig, EvolutionConfig, RunConfig, TaskConfig
from db.client import SpacetimeHttpClient
from db.reducers import SpacetimeRepository
from evolve.run_loop import execute_run


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
            raise RuntimeError("simulated live crash")

    def on_run_finished(self, run) -> None:
        return None


def test_live_spacetimedb_resume_preserves_commits_and_elites() -> None:
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

        database_name = f"neat-live-resume-{uuid4().hex[:10]}"
        subprocess.run(
            ["spacetime", "publish", "--server", "local", database_name],
            cwd=module_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        config = AppConfig(
            task=replace(TaskConfig(), name="delayed_xor", activation_steps=4, temporal_delay_steps=1),
            run=replace(
                RunConfig(),
                generations=5,
                seed=37,
                elite_top_k=2,
                run_id_prefix="spacetime-live-resume",
            ),
            evolution=replace(EvolutionConfig(), population_size=16, max_nodes=16, max_conns=28),
        )
        repository = SpacetimeRepository(
            client=SpacetimeHttpClient(server_url="http://127.0.0.1:3000", database_name=database_name),
            run_id_prefix=config.run.run_id_prefix,
        )

        observer = CrashAfterCommits(stop_after=2)
        with pytest.raises(RuntimeError, match="simulated live crash"):
            execute_run(config=config, repository=repository, observer=observer)

        assert observer.run_id is not None
        committed_before_resume = repository.list_generations(observer.run_id)
        assert [generation.generation_id for generation in committed_before_resume] == [0, 1]
        assert all(generation.state == "committed" for generation in committed_before_resume)

        elites_before_resume = sorted(
            repository.list_elites(observer.run_id, limit=100),
            key=lambda elite: (elite.source_generation, elite.rank),
        )
        assert len(elites_before_resume) == 4

        resumed = execute_run(
            config=config,
            repository=repository,
            resume_run_id=observer.run_id,
        )

        generations_after_resume = repository.list_generations(resumed.run.run_id)
        assert len(generations_after_resume) == config.run.generations
        assert all(generation.state == "committed" for generation in generations_after_resume)
        assert len({generation.generation_id for generation in generations_after_resume}) == config.run.generations

        elites_after_resume = sorted(
            [
                elite
                for elite in repository.list_elites(resumed.run.run_id, limit=200)
                if elite.source_generation <= 1
            ],
            key=lambda elite: (elite.source_generation, elite.rank),
        )
        assert [
            (elite.source_generation, elite.rank, elite.candidate_id, elite.score, elite.frozen_genome_blob)
            for elite in elites_after_resume
        ] == [
            (elite.source_generation, elite.rank, elite.candidate_id, elite.score, elite.frozen_genome_blob)
            for elite in elites_before_resume
        ]

        commit_events = [
            event
            for event in repository.list_events(resumed.run.run_id, limit=400)
            if event.type == "generation_committed"
        ]
        committed_generation_ids = sorted(
            json.loads(event.payload_json)["generation_id"]
            for event in commit_events
        )
        assert committed_generation_ids == list(range(config.run.generations))

        resume_events = [
            event
            for event in repository.list_events(resumed.run.run_id, limit=400)
            if event.type == "run_resumed"
        ]
        assert len(resume_events) == 1
        assert json.loads(resume_events[0].payload_json)["next_generation_id"] == 2
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=10)


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
