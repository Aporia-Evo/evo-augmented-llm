from __future__ import annotations

import json
from dataclasses import asdict
from typing import Protocol
from uuid import uuid4

from db import queries
from db.client import SpacetimeHttpClient
from db.models import (
    ArchiveCellRecord,
    ArchiveEventRecord,
    CandidateRecord,
    CandidateFeatureRecord,
    CandidateFeatureVectorRecord,
    CheckpointRecord,
    EliteRecord,
    EventRecord,
    FitnessRecord,
    GenerationCommitResult,
    GenerationRecord,
    RunRecord,
)
from utils.serialization import stable_json_dumps, utc_now_iso


class RunRepository(Protocol):
    def create_run(self, task_name: str, seed: int, config_json: str, mode: str = "generation") -> RunRecord:
        ...

    def insert_population(self, run_id: str, generation_id: int, candidates: list[CandidateRecord]) -> None:
        ...

    def record_fitness(
        self,
        candidate_id: str,
        run_id: str,
        generation_id: int,
        score: float,
        raw_metrics_json: str,
    ) -> FitnessRecord:
        ...

    def mark_generation_ready(
        self,
        run_id: str,
        generation_id: int,
        eval_duration_ms: int | None = None,
    ) -> GenerationRecord:
        ...

    def commit_generation(
        self,
        run_id: str,
        generation_id: int,
        commit_duration_ms: int | None = None,
    ) -> GenerationCommitResult:
        ...

    def create_next_generation(self, run_id: str, next_generation_id: int, offspring: list[CandidateRecord]) -> None:
        ...

    def save_checkpoint(
        self,
        run_id: str,
        generation_id: int,
        state_blob: str,
        parent_ids_json: str,
    ) -> CheckpointRecord:
        ...

    def append_event(self, run_id: str, event_type: str, payload_json: str) -> EventRecord:
        ...

    def get_checkpoint(self, run_id: str, generation_id: int) -> CheckpointRecord | None:
        ...

    def finish_run(self, run_id: str) -> RunRecord:
        ...

    def get_run(self, run_id: str) -> RunRecord | None:
        ...

    def list_runs(self, limit: int = 10) -> list[RunRecord]:
        ...

    def list_generations(self, run_id: str) -> list[GenerationRecord]:
        ...

    def get_generation(self, run_id: str, generation_id: int) -> GenerationRecord | None:
        ...

    def list_candidates(self, run_id: str, generation_id: int) -> list[CandidateRecord]:
        ...

    def list_fitness(self, run_id: str, generation_id: int) -> list[FitnessRecord]:
        ...

    def list_elites(self, run_id: str, generation_id: int | None = None, limit: int = 10) -> list[EliteRecord]:
        ...

    def list_events(self, run_id: str, limit: int = 20) -> list[EventRecord]:
        ...

    def upsert_candidate_features(self, record: CandidateFeatureRecord) -> CandidateFeatureRecord:
        ...

    def mark_hof_candidate(self, candidate_id: str, hof_flag: bool = True) -> CandidateFeatureRecord | None:
        ...

    def upsert_candidate_feature_vector(self, record: CandidateFeatureVectorRecord) -> CandidateFeatureVectorRecord:
        ...

    def list_candidate_features(
        self,
        *,
        benchmark_label: str | None = None,
        task_name: str | None = None,
        variant: str | None = None,
        delay_steps: int | None = None,
        run_id: str | None = None,
    ) -> list[CandidateFeatureRecord]:
        ...

    def list_candidate_feature_vectors(
        self,
        *,
        candidate_ids: list[str] | None = None,
        feature_version: str | None = None,
    ) -> list[CandidateFeatureVectorRecord]:
        ...

    def consider_archive_candidate(self, record: ArchiveCellRecord) -> ArchiveEventRecord:
        ...

    def list_archive_cells(
        self,
        *,
        benchmark_label: str | None = None,
        task_name: str | None = None,
        variant: str | None = None,
        delay_steps: int | None = None,
        qd_profile: str | None = None,
    ) -> list[ArchiveCellRecord]:
        ...

    def list_archive_events(
        self,
        *,
        benchmark_label: str | None = None,
        task_name: str | None = None,
        variant: str | None = None,
        delay_steps: int | None = None,
        qd_profile: str | None = None,
    ) -> list[ArchiveEventRecord]:
        ...


class GenerationInMemoryRepository:
    def __init__(self, run_id_prefix: str) -> None:
        self.run_id_prefix = run_id_prefix
        self.runs: dict[str, RunRecord] = {}
        self.generations: dict[tuple[str, int], GenerationRecord] = {}
        self.candidates: dict[str, CandidateRecord] = {}
        self.fitness: dict[str, FitnessRecord] = {}
        self.elites: dict[str, EliteRecord] = {}
        self.events: dict[str, EventRecord] = {}
        self.checkpoints: dict[tuple[str, int], CheckpointRecord] = {}
        self.candidate_features: dict[str, CandidateFeatureRecord] = {}
        self.candidate_feature_vectors: dict[tuple[str, str], CandidateFeatureVectorRecord] = {}
        self.archive_cells: dict[str, ArchiveCellRecord] = {}
        self.archive_events: dict[str, ArchiveEventRecord] = {}

    def create_run(self, task_name: str, seed: int, config_json: str, mode: str = "generation") -> RunRecord:
        created_at = utc_now_iso()
        run_id = f"{self.run_id_prefix}-{seed}-{created_at.replace(':', '').replace('-', '')}"
        record = RunRecord(
            run_id=run_id,
            task_name=task_name,
            seed=seed,
            status="running",
            config_json=config_json,
            created_at=created_at,
            finished_at=None,
            mode=mode,
        )
        self.runs[run_id] = record
        self._append_event(run_id, "run_created", {"seed": seed, "task_name": task_name, "mode": mode})
        return record

    def insert_population(self, run_id: str, generation_id: int, candidates: list[CandidateRecord]) -> None:
        self._ensure_generation(run_id, generation_id)
        for candidate in candidates:
            self.candidates[candidate.candidate_id] = candidate
        self._append_event(
            run_id,
            "population_inserted",
            {"generation_id": generation_id, "candidate_count": len(candidates)},
        )

    def record_fitness(
        self,
        candidate_id: str,
        run_id: str,
        generation_id: int,
        score: float,
        raw_metrics_json: str,
    ) -> FitnessRecord:
        record = FitnessRecord(
            candidate_id=candidate_id,
            run_id=run_id,
            generation_id=generation_id,
            score=score,
            raw_metrics=json.loads(raw_metrics_json),
            evaluated_at=utc_now_iso(),
        )
        self.fitness[candidate_id] = record
        candidate = self.candidates[candidate_id]
        self.candidates[candidate_id] = CandidateRecord(**{**asdict(candidate), "status": "evaluated"})
        return record

    def mark_generation_ready(
        self,
        run_id: str,
        generation_id: int,
        eval_duration_ms: int | None = None,
    ) -> GenerationRecord:
        generation = self._require_generation(run_id, generation_id)
        generation = GenerationRecord(
            **{
                **asdict(generation),
                "state": "ready",
                "eval_duration_ms": eval_duration_ms,
            }
        )
        self.generations[(run_id, generation_id)] = generation
        self._append_event(
            run_id,
            "generation_ready",
            {"generation_id": generation_id, "eval_duration_ms": eval_duration_ms},
        )
        return generation

    def commit_generation(
        self,
        run_id: str,
        generation_id: int,
        commit_duration_ms: int | None = None,
    ) -> GenerationCommitResult:
        generation = self._require_generation(run_id, generation_id)
        if generation.state == "committed":
            if commit_duration_ms is not None and generation.commit_duration_ms != commit_duration_ms:
                generation = GenerationRecord(**{**asdict(generation), "commit_duration_ms": commit_duration_ms})
                self.generations[(run_id, generation_id)] = generation
            return GenerationCommitResult(
                generation=generation,
                elites=tuple(self.list_elites(run_id, generation_id=generation_id, limit=100)),
            )
        if generation.state not in {"ready", "evaluating"}:
            raise ValueError(f"Generation {generation_id} is not ready to commit: {generation.state}")

        generation_candidates = self.list_candidates(run_id, generation_id)
        if not generation_candidates:
            raise ValueError(f"No candidates found for generation {generation_id}")

        scored: list[tuple[CandidateRecord, FitnessRecord]] = []
        for candidate in generation_candidates:
            fitness = self.fitness.get(candidate.candidate_id)
            if fitness is None:
                raise ValueError(f"Missing fitness for candidate {candidate.candidate_id}")
            scored.append((candidate, fitness))

        scored.sort(key=lambda pair: (-pair[1].score, pair[0].candidate_id))
        best_candidate, best_fitness = scored[0]
        avg_score = sum(item[1].score for item in scored) / len(scored)
        elite_top_k = _elite_top_k_from_config(self.runs[run_id].config_json)
        archived_at = utc_now_iso()
        elites: list[EliteRecord] = []
        for rank, (candidate, fitness) in enumerate(scored[:elite_top_k], start=1):
            elite = EliteRecord(
                elite_id=f"{run_id}-g{generation_id:04d}-elite-{rank:02d}",
                run_id=run_id,
                source_generation=generation_id,
                candidate_id=candidate.candidate_id,
                rank=rank,
                score=fitness.score,
                frozen_genome_blob=str(candidate.genome_blob),
                archived_at=archived_at,
            )
            self.elites[elite.elite_id] = elite
            elites.append(elite)

        committed = GenerationRecord(
            run_id=run_id,
            generation_id=generation_id,
            state="committed",
            best_candidate_id=best_candidate.candidate_id,
            best_score=best_fitness.score,
            avg_score=avg_score,
            created_at=generation.created_at,
            committed_at=archived_at,
            eval_duration_ms=generation.eval_duration_ms,
            commit_duration_ms=commit_duration_ms,
        )
        self.generations[(run_id, generation_id)] = committed
        self._append_event(
            run_id,
            "generation_committed",
            {
                "generation_id": generation_id,
                "best_candidate_id": best_candidate.candidate_id,
                "best_score": best_fitness.score,
                "avg_score": avg_score,
                "elite_candidate_ids": [elite.candidate_id for elite in elites],
                "eval_duration_ms": generation.eval_duration_ms,
                "commit_duration_ms": commit_duration_ms,
            },
        )
        return GenerationCommitResult(generation=committed, elites=tuple(elites))

    def create_next_generation(self, run_id: str, next_generation_id: int, offspring: list[CandidateRecord]) -> None:
        self.insert_population(run_id, next_generation_id, offspring)

    def save_checkpoint(
        self,
        run_id: str,
        generation_id: int,
        state_blob: str,
        parent_ids_json: str,
    ) -> CheckpointRecord:
        record = CheckpointRecord(
            run_id=run_id,
            generation_id=generation_id,
            state_blob=state_blob,
            parent_ids_json=parent_ids_json,
            created_at=utc_now_iso(),
        )
        self.checkpoints[(run_id, generation_id)] = record
        self._append_event(run_id, "checkpoint_saved", {"generation_id": generation_id})
        return record

    def get_checkpoint(self, run_id: str, generation_id: int) -> CheckpointRecord | None:
        return self.checkpoints.get((run_id, generation_id))

    def append_event(self, run_id: str, event_type: str, payload_json: str) -> EventRecord:
        payload = json.loads(payload_json)
        self._append_event(run_id, event_type, payload)
        latest = self.list_events(run_id, limit=1)
        if not latest:
            raise RuntimeError("Event append did not persist.")
        return latest[0]

    def finish_run(self, run_id: str) -> RunRecord:
        run = self.runs[run_id]
        if run.status == "finished":
            return run
        finished = RunRecord(**{**asdict(run), "status": "finished", "finished_at": utc_now_iso()})
        self.runs[run_id] = finished
        self._append_event(run_id, "run_finished", {})
        return finished

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.runs.get(run_id)

    def list_runs(self, limit: int = 10) -> list[RunRecord]:
        return sorted(self.runs.values(), key=lambda run: run.created_at, reverse=True)[:limit]

    def list_generations(self, run_id: str) -> list[GenerationRecord]:
        return sorted(
            (generation for generation in self.generations.values() if generation.run_id == run_id),
            key=lambda generation: generation.generation_id,
        )

    def get_generation(self, run_id: str, generation_id: int) -> GenerationRecord | None:
        return self.generations.get((run_id, generation_id))

    def list_candidates(self, run_id: str, generation_id: int) -> list[CandidateRecord]:
        rows = [
            candidate
            for candidate in self.candidates.values()
            if candidate.run_id == run_id and candidate.generation_id == generation_id
        ]
        rows.sort(key=lambda candidate: candidate.candidate_id)
        return rows

    def list_fitness(self, run_id: str, generation_id: int) -> list[FitnessRecord]:
        rows = [
            record
            for record in self.fitness.values()
            if record.run_id == run_id and record.generation_id == generation_id
        ]
        rows.sort(key=lambda record: record.candidate_id)
        return rows

    def list_elites(self, run_id: str, generation_id: int | None = None, limit: int = 10) -> list[EliteRecord]:
        rows = [
            elite
            for elite in self.elites.values()
            if elite.run_id == run_id and (generation_id is None or elite.source_generation == generation_id)
        ]
        rows.sort(key=lambda elite: (-elite.source_generation, elite.rank))
        return rows[:limit]

    def list_events(self, run_id: str, limit: int = 20) -> list[EventRecord]:
        rows = [event for event in self.events.values() if event.run_id == run_id]
        rows.sort(key=lambda event: event.created_at, reverse=True)
        return rows[:limit]

    def upsert_candidate_features(self, record: CandidateFeatureRecord) -> CandidateFeatureRecord:
        self.candidate_features[record.candidate_id] = record
        return record

    def mark_hof_candidate(self, candidate_id: str, hof_flag: bool = True) -> CandidateFeatureRecord | None:
        existing = self.candidate_features.get(candidate_id)
        if existing is None:
            return None
        updated = CandidateFeatureRecord(**{**asdict(existing), "hof_flag": bool(hof_flag)})
        self.candidate_features[candidate_id] = updated
        return updated

    def upsert_candidate_feature_vector(self, record: CandidateFeatureVectorRecord) -> CandidateFeatureVectorRecord:
        self.candidate_feature_vectors[(record.candidate_id, record.feature_version)] = record
        return record

    def list_candidate_features(
        self,
        *,
        benchmark_label: str | None = None,
        task_name: str | None = None,
        variant: str | None = None,
        delay_steps: int | None = None,
        run_id: str | None = None,
    ) -> list[CandidateFeatureRecord]:
        rows = list(self.candidate_features.values())
        if benchmark_label is not None:
            rows = [row for row in rows if row.benchmark_label == benchmark_label]
        if task_name is not None:
            rows = [row for row in rows if row.task_name == task_name]
        if variant is not None:
            rows = [row for row in rows if row.variant == variant]
        if delay_steps is not None:
            rows = [row for row in rows if row.delay_steps == delay_steps]
        if run_id is not None:
            rows = [row for row in rows if row.run_id == run_id]
        rows.sort(key=lambda row: (row.task_name, row.delay_steps, row.variant, row.seed, row.generation, row.candidate_id))
        return rows

    def list_candidate_feature_vectors(
        self,
        *,
        candidate_ids: list[str] | None = None,
        feature_version: str | None = None,
    ) -> list[CandidateFeatureVectorRecord]:
        rows = list(self.candidate_feature_vectors.values())
        if candidate_ids is not None:
            allowed = set(candidate_ids)
            rows = [row for row in rows if row.candidate_id in allowed]
        if feature_version is not None:
            rows = [row for row in rows if row.feature_version == feature_version]
        rows.sort(key=lambda row: (row.feature_version, row.candidate_id))
        return rows

    def consider_archive_candidate(self, record: ArchiveCellRecord) -> ArchiveEventRecord:
        existing = self.archive_cells.get(record.archive_id)
        event_type = "skip"
        if existing is None:
            self.archive_cells[record.archive_id] = record
            event_type = "insert"
        elif record.elite_score > existing.elite_score:
            self.archive_cells[record.archive_id] = record
            event_type = "replace"
        event = ArchiveEventRecord(
            event_id=f"{record.archive_id}:event:{uuid4().hex}",
            archive_id=record.archive_id,
            benchmark_label=record.benchmark_label,
            task_name=record.task_name,
            delay_steps=record.delay_steps,
            variant=record.variant,
            descriptor_key=record.descriptor_key,
            candidate_id=record.elite_candidate_id,
            event_type=event_type,
            score=record.elite_score,
            created_at=record.updated_at,
            qd_profile=record.qd_profile,
            descriptor_schema_version=record.descriptor_schema_version,
            curriculum_enabled=record.curriculum_enabled,
            curriculum_phase_1_delays=record.curriculum_phase_1_delays,
            curriculum_phase_2_delays=record.curriculum_phase_2_delays,
            curriculum_switch_generation=record.curriculum_switch_generation,
        )
        self.archive_events[event.event_id] = event
        return event

    def list_archive_cells(
        self,
        *,
        benchmark_label: str | None = None,
        task_name: str | None = None,
        variant: str | None = None,
        delay_steps: int | None = None,
        qd_profile: str | None = None,
    ) -> list[ArchiveCellRecord]:
        rows = list(self.archive_cells.values())
        if benchmark_label is not None:
            rows = [row for row in rows if row.benchmark_label == benchmark_label]
        if task_name is not None:
            rows = [row for row in rows if row.task_name == task_name]
        if variant is not None:
            rows = [row for row in rows if row.variant == variant]
        if delay_steps is not None:
            rows = [row for row in rows if row.delay_steps == delay_steps]
        if qd_profile is not None:
            rows = [row for row in rows if row.qd_profile == qd_profile]
        rows.sort(key=lambda row: (row.task_name, row.delay_steps, row.variant, row.qd_profile, -row.elite_score, row.archive_id))
        return rows

    def list_archive_events(
        self,
        *,
        benchmark_label: str | None = None,
        task_name: str | None = None,
        variant: str | None = None,
        delay_steps: int | None = None,
        qd_profile: str | None = None,
    ) -> list[ArchiveEventRecord]:
        rows = list(self.archive_events.values())
        if benchmark_label is not None:
            rows = [row for row in rows if row.benchmark_label == benchmark_label]
        if task_name is not None:
            rows = [row for row in rows if row.task_name == task_name]
        if variant is not None:
            rows = [row for row in rows if row.variant == variant]
        if delay_steps is not None:
            rows = [row for row in rows if row.delay_steps == delay_steps]
        if qd_profile is not None:
            rows = [row for row in rows if row.qd_profile == qd_profile]
        rows.sort(key=lambda row: (row.task_name, row.delay_steps, row.variant, row.qd_profile, row.created_at, row.event_id))
        return rows

    def _ensure_generation(self, run_id: str, generation_id: int) -> None:
        key = (run_id, generation_id)
        if key not in self.generations:
            self.generations[key] = GenerationRecord(
                run_id=run_id,
                generation_id=generation_id,
                state="evaluating",
                best_candidate_id=None,
                best_score=None,
                avg_score=None,
                created_at=utc_now_iso(),
                committed_at=None,
                eval_duration_ms=None,
                commit_duration_ms=None,
            )

    def _require_generation(self, run_id: str, generation_id: int) -> GenerationRecord:
        key = (run_id, generation_id)
        if key not in self.generations:
            raise KeyError(f"Generation not found: {run_id} / {generation_id}")
        return self.generations[key]

    def _append_event(self, run_id: str, event_type: str, payload: dict[str, object]) -> None:
        event = EventRecord(
            event_id=f"{run_id}-event-{uuid4().hex}",
            run_id=run_id,
            type=event_type,
            payload_json=stable_json_dumps(payload),
            created_at=utc_now_iso(),
        )
        self.events[event.event_id] = event


class GenerationSpacetimeRepository:
    def __init__(self, client: SpacetimeHttpClient, run_id_prefix: str) -> None:
        self.client = client
        self.run_id_prefix = run_id_prefix

    def create_run(self, task_name: str, seed: int, config_json: str, mode: str = "generation") -> RunRecord:
        created_at = utc_now_iso()
        run_id = f"{self.run_id_prefix}-{seed}-{created_at.replace(':', '').replace('-', '')}"
        self.client.call_reducer("create_run", run_id, task_name, seed, config_json, created_at, mode)
        run = self.get_run(run_id)
        if run is None:
            raise RuntimeError("Run creation succeeded but run is not queryable.")
        return run

    def insert_population(self, run_id: str, generation_id: int, candidates: list[CandidateRecord]) -> None:
        payload = [
            {
                "candidate_id": candidate.candidate_id,
                "genome_blob": candidate.genome_blob,
                "status": candidate.status,
                "parent_ids_json": stable_json_dumps(candidate.parent_ids),
                "created_at": candidate.created_at,
            }
            for candidate in candidates
        ]
        self.client.call_reducer("insert_population", run_id, generation_id, payload)

    def record_fitness(
        self,
        candidate_id: str,
        run_id: str,
        generation_id: int,
        score: float,
        raw_metrics_json: str,
    ) -> FitnessRecord:
        evaluated_at = utc_now_iso()
        self.client.call_reducer("record_fitness", candidate_id, run_id, generation_id, score, raw_metrics_json, evaluated_at)
        return FitnessRecord(
            candidate_id=candidate_id,
            run_id=run_id,
            generation_id=generation_id,
            score=score,
            raw_metrics=json.loads(raw_metrics_json),
            evaluated_at=evaluated_at,
        )

    def mark_generation_ready(
        self,
        run_id: str,
        generation_id: int,
        eval_duration_ms: int | None = None,
    ) -> GenerationRecord:
        self.client.call_reducer("mark_generation_ready", run_id, generation_id, _duration_arg(eval_duration_ms))
        generation = self.get_generation(run_id, generation_id)
        if generation is None:
            raise RuntimeError("Generation not found after mark_generation_ready.")
        return generation

    def commit_generation(
        self,
        run_id: str,
        generation_id: int,
        commit_duration_ms: int | None = None,
    ) -> GenerationCommitResult:
        self.client.call_reducer("commit_generation", run_id, generation_id, utc_now_iso(), _duration_arg(commit_duration_ms))
        generation = self.get_generation(run_id, generation_id)
        if generation is None:
            raise RuntimeError("Generation not found after commit_generation.")
        elites = self.list_elites(run_id, generation_id=generation_id, limit=100)
        return GenerationCommitResult(generation=generation, elites=tuple(elites))

    def create_next_generation(self, run_id: str, next_generation_id: int, offspring: list[CandidateRecord]) -> None:
        payload = [
            {
                "candidate_id": candidate.candidate_id,
                "genome_blob": candidate.genome_blob,
                "status": candidate.status,
                "parent_ids_json": stable_json_dumps(candidate.parent_ids),
                "created_at": candidate.created_at,
            }
            for candidate in offspring
        ]
        self.client.call_reducer("create_next_generation", run_id, next_generation_id, payload)

    def save_checkpoint(
        self,
        run_id: str,
        generation_id: int,
        state_blob: str,
        parent_ids_json: str,
    ) -> CheckpointRecord:
        created_at = utc_now_iso()
        self.client.call_reducer("upsert_checkpoint", run_id, generation_id, state_blob, parent_ids_json, created_at)
        checkpoint = self.get_checkpoint(run_id, generation_id)
        if checkpoint is None:
            raise RuntimeError("Checkpoint not found after upsert_checkpoint.")
        return checkpoint

    def get_checkpoint(self, run_id: str, generation_id: int) -> CheckpointRecord | None:
        return queries.get_checkpoint(self.client, run_id, generation_id)

    def append_event(self, run_id: str, event_type: str, payload_json: str) -> EventRecord:
        created_at = utc_now_iso()
        self.client.call_reducer("append_event", run_id, event_type, payload_json, created_at)
        latest = self.list_events(run_id, limit=1)
        if not latest:
            raise RuntimeError("Event append did not persist.")
        return latest[0]

    def finish_run(self, run_id: str) -> RunRecord:
        self.client.call_reducer("finish_run", run_id, utc_now_iso())
        run = self.get_run(run_id)
        if run is None:
            raise RuntimeError("Run not found after finish_run.")
        return run

    def get_run(self, run_id: str) -> RunRecord | None:
        return queries.get_run(self.client, run_id)

    def list_runs(self, limit: int = 10) -> list[RunRecord]:
        return queries.list_runs(self.client, limit=limit)

    def list_generations(self, run_id: str) -> list[GenerationRecord]:
        return queries.list_generations(self.client, run_id)

    def get_generation(self, run_id: str, generation_id: int) -> GenerationRecord | None:
        return queries.get_generation(self.client, run_id, generation_id)

    def list_candidates(self, run_id: str, generation_id: int) -> list[CandidateRecord]:
        return queries.list_candidates(self.client, run_id, generation_id)

    def list_fitness(self, run_id: str, generation_id: int) -> list[FitnessRecord]:
        return queries.list_fitness(self.client, run_id, generation_id)

    def list_elites(self, run_id: str, generation_id: int | None = None, limit: int = 10) -> list[EliteRecord]:
        return queries.list_elites(self.client, run_id, generation_id=generation_id, limit=limit)

    def list_events(self, run_id: str, limit: int = 20) -> list[EventRecord]:
        return queries.list_events(self.client, run_id, limit=limit)

    def upsert_candidate_features(self, record: CandidateFeatureRecord) -> CandidateFeatureRecord:
        payload = {
            "candidate_id": record.candidate_id,
            "run_id": record.run_id,
            "benchmark_label": record.benchmark_label,
            "task_name": record.task_name,
            "delay_steps": record.delay_steps,
            "variant": record.variant,
            "seed": record.seed,
            "generation": record.generation,
            "hof_flag": record.hof_flag,
            "success": record.success,
            "final_max_score": record.final_max_score,
            "first_success_generation": -1 if record.first_success_generation is None else int(record.first_success_generation),
            "mean_alpha": record.mean_alpha,
            "std_alpha": record.std_alpha,
            "mean_eta": record.mean_eta,
            "std_eta": record.std_eta,
            "mean_plastic_d": record.mean_plastic_d,
            "std_plastic_d": record.std_plastic_d,
            "plastic_d_at_lower_bound_fraction": record.plastic_d_at_lower_bound_fraction,
            "plastic_d_at_zero_fraction": record.plastic_d_at_zero_fraction,
            "node_count": record.node_count,
            "enabled_conn_count": record.enabled_conn_count,
            "mean_abs_delta_w": record.mean_abs_delta_w,
            "max_abs_delta_w": record.max_abs_delta_w,
            "clamp_hit_rate": record.clamp_hit_rate,
            "plasticity_active_fraction": record.plasticity_active_fraction,
            "mean_abs_fast_state": record.mean_abs_fast_state,
            "mean_abs_slow_state": record.mean_abs_slow_state,
            "slow_fast_contribution_ratio": record.slow_fast_contribution_ratio,
            "mean_abs_decay_term": record.mean_abs_decay_term,
            "max_abs_decay_term": record.max_abs_decay_term,
            "decay_effect_ratio": record.decay_effect_ratio,
            "decay_near_zero_fraction": record.decay_near_zero_fraction,
            "score_delay_3": record.score_delay_3,
            "score_delay_5": record.score_delay_5,
            "score_delay_8": record.score_delay_8,
            "success_delay_3": record.success_delay_3,
            "success_delay_5": record.success_delay_5,
            "success_delay_8": record.success_delay_8,
            "mean_score_over_delays": record.mean_score_over_delays,
            "delay_score_std": record.delay_score_std,
            "delay_score_range": record.delay_score_range,
            "curriculum_enabled": record.curriculum_enabled,
            "curriculum_phase_1_delays": record.curriculum_phase_1_delays,
            "curriculum_phase_2_delays": record.curriculum_phase_2_delays,
            "curriculum_switch_generation": record.curriculum_switch_generation,
            "curriculum_phase": record.curriculum_phase,
            "active_evaluation_delays": record.active_evaluation_delays,
            "score_current_phase": record.score_current_phase,
            "query_accuracy": record.query_accuracy,
            "retrieval_score": record.retrieval_score,
            "exact_match_success": record.exact_match_success,
            "mean_query_distance": record.mean_query_distance,
            "distractor_load": record.distractor_load,
            "num_stores": record.num_stores,
            "num_queries": record.num_queries,
            "num_distractors": record.num_distractors,
            "retrieval_margin": record.retrieval_margin,
            "retrieval_confusion_rate": record.retrieval_confusion_rate,
            "relevant_token_retention": record.relevant_token_retention,
            "query_response_margin": record.query_response_margin,
            "distractor_suppression_ratio": record.distractor_suppression_ratio,
            "correct_key_selected": record.correct_key_selected,
            "correct_value_selected": record.correct_value_selected,
            "query_key_match_score": record.query_key_match_score,
            "value_margin": record.value_margin,
            "distractor_competition_score": record.distractor_competition_score,
            "mean_abs_fast_state_during_store": record.mean_abs_fast_state_during_store,
            "mean_abs_slow_state_during_store": record.mean_abs_slow_state_during_store,
            "mean_abs_fast_state_during_query": record.mean_abs_fast_state_during_query,
            "mean_abs_slow_state_during_query": record.mean_abs_slow_state_during_query,
            "mean_abs_fast_state_during_distractor": record.mean_abs_fast_state_during_distractor,
            "mean_abs_slow_state_during_distractor": record.mean_abs_slow_state_during_distractor,
            "slow_query_coupling": record.slow_query_coupling,
            "store_query_state_gap": record.store_query_state_gap,
            "slow_fast_retrieval_ratio": record.slow_fast_retrieval_ratio,
            "retrieval_state_alignment": record.retrieval_state_alignment,
        }
        self.client.call_reducer("upsert_candidate_features", payload)
        rows = self.list_candidate_features(run_id=record.run_id)
        for row in rows:
            if row.candidate_id == record.candidate_id:
                return row
        return record

    def mark_hof_candidate(self, candidate_id: str, hof_flag: bool = True) -> CandidateFeatureRecord | None:
        self.client.call_reducer("mark_candidate_feature_hof", candidate_id, bool(hof_flag))
        rows = queries.list_candidate_features(self.client, candidate_id=candidate_id)
        return rows[0] if rows else None

    def upsert_candidate_feature_vector(self, record: CandidateFeatureVectorRecord) -> CandidateFeatureVectorRecord:
        payload = {
            "candidate_id": record.candidate_id,
            "feature_version": record.feature_version,
            "vector_json": record.vector_json,
            "norm_l2": record.norm_l2,
        }
        self.client.call_reducer("upsert_candidate_feature_vector", payload)
        rows = self.list_candidate_feature_vectors(
            candidate_ids=[record.candidate_id],
            feature_version=record.feature_version,
        )
        return rows[0] if rows else record

    def list_candidate_features(
        self,
        *,
        benchmark_label: str | None = None,
        task_name: str | None = None,
        variant: str | None = None,
        delay_steps: int | None = None,
        run_id: str | None = None,
    ) -> list[CandidateFeatureRecord]:
        return queries.list_candidate_features(
            self.client,
            benchmark_label=benchmark_label,
            task_name=task_name,
            variant=variant,
            delay_steps=delay_steps,
            run_id=run_id,
        )

    def list_candidate_feature_vectors(
        self,
        *,
        candidate_ids: list[str] | None = None,
        feature_version: str | None = None,
    ) -> list[CandidateFeatureVectorRecord]:
        return queries.list_candidate_feature_vectors(
            self.client,
            candidate_ids=candidate_ids,
            feature_version=feature_version,
        )

    def consider_archive_candidate(self, record: ArchiveCellRecord) -> ArchiveEventRecord:
        payload = {
            "archive_id": record.archive_id,
            "benchmark_label": record.benchmark_label,
            "task_name": record.task_name,
            "delay_steps": record.delay_steps,
            "variant": record.variant,
            "qd_profile": record.qd_profile,
            "descriptor_schema_version": record.descriptor_schema_version,
            "descriptor_key": record.descriptor_key,
            "descriptor_values_json": record.descriptor_values_json,
            "elite_candidate_id": record.elite_candidate_id,
            "elite_score": record.elite_score,
            "elite_run_id": record.elite_run_id,
            "updated_at": record.updated_at,
            "curriculum_enabled": record.curriculum_enabled,
            "curriculum_phase_1_delays": record.curriculum_phase_1_delays,
            "curriculum_phase_2_delays": record.curriculum_phase_2_delays,
            "curriculum_switch_generation": record.curriculum_switch_generation,
        }
        event_id = f"{record.archive_id}:event:{uuid4().hex}"
        self.client.call_reducer("consider_archive_candidate", payload, event_id)
        rows = self.list_archive_events(
            benchmark_label=record.benchmark_label,
            task_name=record.task_name,
            variant=record.variant,
            delay_steps=record.delay_steps,
            qd_profile=record.qd_profile,
        )
        for row in rows:
            if row.event_id == event_id:
                return row
        return ArchiveEventRecord(
            event_id=event_id,
            archive_id=record.archive_id,
            benchmark_label=record.benchmark_label,
            task_name=record.task_name,
            delay_steps=record.delay_steps,
            variant=record.variant,
            descriptor_key=record.descriptor_key,
            candidate_id=record.elite_candidate_id,
            event_type="skip",
            score=record.elite_score,
            created_at=record.updated_at,
            qd_profile=record.qd_profile,
            descriptor_schema_version=record.descriptor_schema_version,
            curriculum_enabled=record.curriculum_enabled,
            curriculum_phase_1_delays=record.curriculum_phase_1_delays,
            curriculum_phase_2_delays=record.curriculum_phase_2_delays,
            curriculum_switch_generation=record.curriculum_switch_generation,
        )

    def list_archive_cells(
        self,
        *,
        benchmark_label: str | None = None,
        task_name: str | None = None,
        variant: str | None = None,
        delay_steps: int | None = None,
        qd_profile: str | None = None,
    ) -> list[ArchiveCellRecord]:
        return queries.list_archive_cells(
            self.client,
            benchmark_label=benchmark_label,
            task_name=task_name,
            variant=variant,
            delay_steps=delay_steps,
            qd_profile=qd_profile,
        )

    def list_archive_events(
        self,
        *,
        benchmark_label: str | None = None,
        task_name: str | None = None,
        variant: str | None = None,
        delay_steps: int | None = None,
        qd_profile: str | None = None,
    ) -> list[ArchiveEventRecord]:
        return queries.list_archive_events(
            self.client,
            benchmark_label=benchmark_label,
            task_name=task_name,
            variant=variant,
            delay_steps=delay_steps,
            qd_profile=qd_profile,
        )


def _elite_top_k_from_config(config_json: str) -> int:
    payload = json.loads(config_json)
    return int(payload.get("run", {}).get("elite_top_k", 3))


def _duration_arg(duration_ms: int | None) -> int:
    return -1 if duration_ms is None else int(duration_ms)
