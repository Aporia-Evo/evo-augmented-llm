from __future__ import annotations

import json
from dataclasses import asdict
from typing import Protocol
from uuid import uuid4

from db import queries
from db.client import SpacetimeHttpClient
from db.generation_repository import GenerationInMemoryRepository, GenerationSpacetimeRepository
from db.models import (
    ActiveCandidateRecord,
    CandidateLifecycleEventRecord,
    EvaluationJobRecord,
    EvaluationResultRecord,
    HallOfFameEntryRecord,
    OnlineMetricRecord,
    OnlineStateRecord,
    RunRecord,
)
from evolve.replacement import choose_replacement_target
from utils.serialization import stable_json_dumps, utc_now_iso


class OnlineRunRepository(Protocol):
    def create_online_run(self, task_name: str, seed: int, config_json: str) -> RunRecord:
        ...

    def seed_active_population(self, run_id: str, candidates: list[ActiveCandidateRecord]) -> None:
        ...

    def list_active_candidates(self, run_id: str, statuses: list[str] | None = None) -> list[ActiveCandidateRecord]:
        ...

    def get_active_candidate(self, candidate_id: str) -> ActiveCandidateRecord | None:
        ...

    def enqueue_evaluation(self, run_id: str, candidate_id: str, task_payload_json: str) -> EvaluationJobRecord:
        ...

    def claim_job(self, run_id: str, worker_id: str) -> EvaluationJobRecord | None:
        ...

    def submit_result(
        self,
        job_id: str,
        candidate_id: str,
        score: float,
        raw_metrics_json: str,
    ) -> EvaluationResultRecord:
        ...

    def update_candidate_rolling_score(
        self,
        candidate_id: str,
        rolling_score: float,
        eval_count: int,
        last_eval_at: str,
        status: str = "active",
    ) -> ActiveCandidateRecord:
        ...

    def select_replacement_target(self, run_id: str) -> ActiveCandidateRecord | None:
        ...

    def spawn_offspring(self, run_id: str, candidate: ActiveCandidateRecord) -> ActiveCandidateRecord:
        ...

    def retire_candidate(self, candidate_id: str) -> ActiveCandidateRecord:
        ...

    def activate_candidate(self, candidate_id: str) -> ActiveCandidateRecord:
        ...

    def promote_to_hall_of_fame(self, candidate_id: str, score: float | None = None) -> HallOfFameEntryRecord | None:
        ...

    def list_hall_of_fame(self, run_id: str, limit: int = 50) -> list[HallOfFameEntryRecord]:
        ...

    def list_evaluation_jobs(self, run_id: str, status: str | None = None, limit: int = 200) -> list[EvaluationJobRecord]:
        ...

    def get_evaluation_job(self, job_id: str) -> EvaluationJobRecord | None:
        ...

    def list_evaluation_results(self, run_id: str, limit: int = 200) -> list[EvaluationResultRecord]:
        ...

    def list_candidate_lifecycle_events(
        self,
        run_id: str,
        candidate_id: str | None = None,
        limit: int = 100,
    ) -> list[CandidateLifecycleEventRecord]:
        ...

    def capture_online_metrics(
        self,
        run_id: str,
        replacement_count: int,
        success_rate_window: float,
    ) -> OnlineMetricRecord:
        ...

    def list_online_metrics(self, run_id: str, limit: int = 100) -> list[OnlineMetricRecord]:
        ...

    def save_online_state(
        self,
        run_id: str,
        step: int,
        replacement_count: int,
        success_window_json: str,
        adapter_state_blob: str,
    ) -> OnlineStateRecord:
        ...

    def get_online_state(self, run_id: str) -> OnlineStateRecord | None:
        ...

    def resume_online_run(self, run_id: str) -> RunRecord:
        ...


class OnlineCapableInMemoryRepository(GenerationInMemoryRepository):
    def __init__(self, run_id_prefix: str) -> None:
        super().__init__(run_id_prefix)
        self.active_candidates: dict[str, ActiveCandidateRecord] = {}
        self.evaluation_jobs: dict[str, EvaluationJobRecord] = {}
        self.evaluation_results: dict[str, EvaluationResultRecord] = {}
        self.hall_of_fame: dict[str, HallOfFameEntryRecord] = {}
        self.candidate_lifecycle_events: dict[str, CandidateLifecycleEventRecord] = {}
        self.online_metrics: dict[str, OnlineMetricRecord] = {}
        self.online_states: dict[str, OnlineStateRecord] = {}

    def create_online_run(self, task_name: str, seed: int, config_json: str) -> RunRecord:
        return self.create_run(task_name, seed, config_json, mode="online")

    def seed_active_population(self, run_id: str, candidates: list[ActiveCandidateRecord]) -> None:
        for candidate in candidates:
            self.active_candidates[candidate.candidate_id] = candidate
            self._append_candidate_lifecycle_event(
                run_id,
                candidate.candidate_id,
                "candidate_created",
                {"slot_index": candidate.slot_index},
                candidate.created_at,
            )

    def list_active_candidates(self, run_id: str, statuses: list[str] | None = None) -> list[ActiveCandidateRecord]:
        rows = [candidate for candidate in self.active_candidates.values() if candidate.run_id == run_id]
        if statuses is not None:
            allowed = set(statuses)
            rows = [candidate for candidate in rows if candidate.status in allowed]
        rows.sort(key=lambda candidate: candidate.slot_index)
        return rows

    def get_active_candidate(self, candidate_id: str) -> ActiveCandidateRecord | None:
        return self.active_candidates.get(candidate_id)

    def enqueue_evaluation(self, run_id: str, candidate_id: str, task_payload_json: str) -> EvaluationJobRecord:
        candidate = self.active_candidates[candidate_id]
        created_at = utc_now_iso()
        job = EvaluationJobRecord(
            job_id=f"{run_id}-job-{uuid4().hex}",
            run_id=run_id,
            candidate_id=candidate_id,
            task_payload_json=task_payload_json,
            status="queued",
            claimed_by=None,
            created_at=created_at,
            claimed_at=None,
            finished_at=None,
        )
        self.evaluation_jobs[job.job_id] = job
        self.active_candidates[candidate_id] = ActiveCandidateRecord(**{**asdict(candidate), "status": "queued"})
        self._append_candidate_lifecycle_event(run_id, candidate_id, "evaluation_enqueued", {"job_id": job.job_id}, created_at)
        return job

    def claim_job(self, run_id: str, worker_id: str) -> EvaluationJobRecord | None:
        queued_jobs = self.list_evaluation_jobs(run_id, status="queued", limit=10000)
        if not queued_jobs:
            return None
        job = queued_jobs[0]
        claimed_at = utc_now_iso()
        claimed = EvaluationJobRecord(
            **{**asdict(job), "status": "claimed", "claimed_by": worker_id, "claimed_at": claimed_at}
        )
        self.evaluation_jobs[job.job_id] = claimed
        candidate = self.active_candidates[job.candidate_id]
        self.active_candidates[job.candidate_id] = ActiveCandidateRecord(**{**asdict(candidate), "status": "evaluating"})
        self._append_candidate_lifecycle_event(run_id, job.candidate_id, "evaluation_claimed", {"job_id": job.job_id, "worker_id": worker_id}, claimed_at)
        return claimed

    def submit_result(
        self,
        job_id: str,
        candidate_id: str,
        score: float,
        raw_metrics_json: str,
    ) -> EvaluationResultRecord:
        job = self.evaluation_jobs[job_id]
        finished_at = utc_now_iso()
        self.evaluation_jobs[job_id] = EvaluationJobRecord(**{**asdict(job), "status": "finished", "finished_at": finished_at})
        result = EvaluationResultRecord(
            result_id=f"{job.run_id}-result-{uuid4().hex}",
            run_id=job.run_id,
            candidate_id=candidate_id,
            score=score,
            raw_metrics=json.loads(raw_metrics_json),
            created_at=finished_at,
        )
        self.evaluation_results[result.result_id] = result
        self._append_candidate_lifecycle_event(job.run_id, candidate_id, "evaluation_finished", {"job_id": job_id, "score": score}, finished_at)
        return result

    def update_candidate_rolling_score(
        self,
        candidate_id: str,
        rolling_score: float,
        eval_count: int,
        last_eval_at: str,
        status: str = "active",
    ) -> ActiveCandidateRecord:
        candidate = self.active_candidates[candidate_id]
        updated = ActiveCandidateRecord(
            **{
                **asdict(candidate),
                "rolling_score": rolling_score,
                "eval_count": eval_count,
                "last_eval_at": last_eval_at,
                "status": status,
            }
        )
        self.active_candidates[candidate_id] = updated
        self._append_candidate_lifecycle_event(updated.run_id, candidate_id, "rolling_score_updated", {"rolling_score": rolling_score, "eval_count": eval_count}, last_eval_at)
        return updated

    def select_replacement_target(self, run_id: str) -> ActiveCandidateRecord | None:
        return choose_replacement_target(self.list_active_candidates(run_id))

    def spawn_offspring(self, run_id: str, candidate: ActiveCandidateRecord) -> ActiveCandidateRecord:
        self.active_candidates[candidate.candidate_id] = candidate
        self._append_candidate_lifecycle_event(run_id, candidate.candidate_id, "candidate_spawned", {"slot_index": candidate.slot_index, "parent_ids": candidate.parent_ids}, candidate.created_at)
        return candidate

    def retire_candidate(self, candidate_id: str) -> ActiveCandidateRecord:
        candidate = self.active_candidates[candidate_id]
        retired_at = utc_now_iso()
        updated = ActiveCandidateRecord(**{**asdict(candidate), "status": "retired"})
        self.active_candidates[candidate_id] = updated
        for job in list(self.evaluation_jobs.values()):
            if job.candidate_id == candidate_id and job.status not in {"finished", "failed", "cancelled"}:
                self.evaluation_jobs[job.job_id] = EvaluationJobRecord(**{**asdict(job), "status": "cancelled", "finished_at": retired_at})
        self._append_candidate_lifecycle_event(candidate.run_id, candidate_id, "candidate_retired", {"slot_index": candidate.slot_index}, retired_at)
        return updated

    def activate_candidate(self, candidate_id: str) -> ActiveCandidateRecord:
        candidate = self.active_candidates[candidate_id]
        activated_at = utc_now_iso()
        updated = ActiveCandidateRecord(**{**asdict(candidate), "status": "active"})
        self.active_candidates[candidate_id] = updated
        self._append_candidate_lifecycle_event(candidate.run_id, candidate_id, "candidate_activated", {"slot_index": candidate.slot_index}, activated_at)
        return updated

    def promote_to_hall_of_fame(self, candidate_id: str, score: float | None = None) -> HallOfFameEntryRecord | None:
        candidate = self.active_candidates[candidate_id]
        if any(entry.candidate_id == candidate_id for entry in self.hall_of_fame.values() if entry.run_id == candidate.run_id):
            return None
        inserted_at = utc_now_iso()
        entry = HallOfFameEntryRecord(
            entry_id=f"{candidate.run_id}-hof-{uuid4().hex}",
            run_id=candidate.run_id,
            candidate_id=candidate_id,
            score=candidate.rolling_score if score is None else float(score),
            frozen_genome_blob=str(candidate.genome_blob),
            inserted_at=inserted_at,
        )
        self.hall_of_fame[entry.entry_id] = entry
        self._append_candidate_lifecycle_event(candidate.run_id, candidate_id, "candidate_promoted_to_hall_of_fame", {"entry_id": entry.entry_id, "score": entry.score}, inserted_at)
        return entry

    def list_hall_of_fame(self, run_id: str, limit: int = 50) -> list[HallOfFameEntryRecord]:
        rows = [entry for entry in self.hall_of_fame.values() if entry.run_id == run_id]
        rows.sort(key=lambda entry: (-entry.score, entry.inserted_at, entry.entry_id))
        return rows[:limit]

    def list_evaluation_jobs(self, run_id: str, status: str | None = None, limit: int = 200) -> list[EvaluationJobRecord]:
        rows = [job for job in self.evaluation_jobs.values() if job.run_id == run_id]
        if status is not None:
            rows = [job for job in rows if job.status == status]
        rows.sort(key=lambda job: (job.created_at, job.job_id))
        return rows[:limit]

    def get_evaluation_job(self, job_id: str) -> EvaluationJobRecord | None:
        return self.evaluation_jobs.get(job_id)

    def list_evaluation_results(self, run_id: str, limit: int = 200) -> list[EvaluationResultRecord]:
        rows = [result for result in self.evaluation_results.values() if result.run_id == run_id]
        rows.sort(key=lambda result: result.created_at, reverse=True)
        return rows[:limit]

    def list_candidate_lifecycle_events(self, run_id: str, candidate_id: str | None = None, limit: int = 100) -> list[CandidateLifecycleEventRecord]:
        rows = [event for event in self.candidate_lifecycle_events.values() if event.run_id == run_id]
        if candidate_id is not None:
            rows = [event for event in rows if event.candidate_id == candidate_id]
        rows.sort(key=lambda event: event.created_at, reverse=True)
        return rows[:limit]

    def capture_online_metrics(self, run_id: str, replacement_count: int, success_rate_window: float) -> OnlineMetricRecord:
        active_rows = [candidate for candidate in self.list_active_candidates(run_id) if candidate.status != "retired"]
        scores = [candidate.rolling_score for candidate in active_rows]
        metric = OnlineMetricRecord(
            metric_id=f"{run_id}-metric-{uuid4().hex}",
            run_id=run_id,
            timestamp=utc_now_iso(),
            active_population_size=len(active_rows),
            rolling_best_score=max(scores) if scores else 0.0,
            rolling_avg_score=(sum(scores) / len(scores)) if scores else 0.0,
            replacement_count=int(replacement_count),
            success_rate_window=float(success_rate_window),
        )
        self.online_metrics[metric.metric_id] = metric
        return metric

    def list_online_metrics(self, run_id: str, limit: int = 100) -> list[OnlineMetricRecord]:
        rows = [metric for metric in self.online_metrics.values() if metric.run_id == run_id]
        rows.sort(key=lambda metric: metric.timestamp, reverse=True)
        return rows[:limit]

    def save_online_state(self, run_id: str, step: int, replacement_count: int, success_window_json: str, adapter_state_blob: str) -> OnlineStateRecord:
        now = utc_now_iso()
        existing = self.online_states.get(run_id)
        record = OnlineStateRecord(
            run_id=run_id,
            step=step,
            replacement_count=replacement_count,
            success_window_json=success_window_json,
            adapter_state_blob=adapter_state_blob,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        self.online_states[run_id] = record
        return record

    def get_online_state(self, run_id: str) -> OnlineStateRecord | None:
        return self.online_states.get(run_id)

    def resume_online_run(self, run_id: str) -> RunRecord:
        run = self.runs[run_id]
        for job in list(self.evaluation_jobs.values()):
            if job.run_id == run_id and job.status == "claimed":
                self.evaluation_jobs[job.job_id] = EvaluationJobRecord(**{**asdict(job), "status": "queued", "claimed_by": None, "claimed_at": None})
        for candidate in list(self.active_candidates.values()):
            if candidate.run_id == run_id and candidate.status == "evaluating":
                self.active_candidates[candidate.candidate_id] = ActiveCandidateRecord(**{**asdict(candidate), "status": "active"})
        self._append_event(run_id, "run_resumed", {"mode": "online"})
        return run

    def _append_candidate_lifecycle_event(
        self,
        run_id: str,
        candidate_id: str,
        event_type: str,
        payload: dict[str, object],
        created_at: str,
    ) -> None:
        event = CandidateLifecycleEventRecord(
            event_id=f"{run_id}-candidate-event-{uuid4().hex}",
            run_id=run_id,
            candidate_id=candidate_id,
            event_type=event_type,
            payload_json=stable_json_dumps(payload),
            created_at=created_at,
        )
        self.candidate_lifecycle_events[event.event_id] = event


class OnlineCapableSpacetimeRepository(GenerationSpacetimeRepository):
    def create_online_run(self, task_name: str, seed: int, config_json: str) -> RunRecord:
        created_at = utc_now_iso()
        run_id = f"{self.run_id_prefix}-{seed}-{created_at.replace(':', '').replace('-', '')}"
        self.client.call_reducer("create_online_run", run_id, task_name, seed, config_json, created_at)
        run = self.get_run(run_id)
        if run is None:
            raise RuntimeError("Online run creation succeeded but run is not queryable.")
        return run

    def seed_active_population(self, run_id: str, candidates: list[ActiveCandidateRecord]) -> None:
        payload = [
            {
                "candidate_id": candidate.candidate_id,
                "slot_index": candidate.slot_index,
                "variant": candidate.variant,
                "genome_blob": candidate.genome_blob,
                "status": candidate.status,
                "rolling_score": candidate.rolling_score,
                "eval_count": candidate.eval_count,
                "birth_step": candidate.birth_step,
                "last_eval_at": candidate.last_eval_at or "",
                "parent_ids_json": stable_json_dumps(candidate.parent_ids),
                "created_at": candidate.created_at,
            }
            for candidate in candidates
        ]
        self.client.call_reducer("seed_active_population", run_id, payload)

    def list_active_candidates(self, run_id: str, statuses: list[str] | None = None) -> list[ActiveCandidateRecord]:
        return queries.list_active_candidates(self.client, run_id, statuses=statuses)

    def get_active_candidate(self, candidate_id: str) -> ActiveCandidateRecord | None:
        return queries.get_active_candidate(self.client, candidate_id)

    def enqueue_evaluation(self, run_id: str, candidate_id: str, task_payload_json: str) -> EvaluationJobRecord:
        job_id = f"{run_id}-job-{uuid4().hex}"
        created_at = utc_now_iso()
        self.client.call_reducer("enqueue_evaluation", job_id, run_id, candidate_id, task_payload_json, created_at)
        job = self.get_evaluation_job(job_id)
        if job is None:
            raise RuntimeError("Evaluation job not found after enqueue.")
        return job

    def claim_job(self, run_id: str, worker_id: str) -> EvaluationJobRecord | None:
        queued = self.list_evaluation_jobs(run_id, status="queued", limit=1)
        if not queued:
            return None
        job = queued[0]
        self.client.call_reducer("claim_job", job.job_id, worker_id, utc_now_iso())
        claimed = self.get_evaluation_job(job.job_id)
        if claimed is None or claimed.status != "claimed" or claimed.claimed_by != worker_id:
            return None
        return claimed

    def submit_result(self, job_id: str, candidate_id: str, score: float, raw_metrics_json: str) -> EvaluationResultRecord:
        result_id = f"{candidate_id}-result-{uuid4().hex}"
        created_at = utc_now_iso()
        self.client.call_reducer("submit_result", result_id, job_id, candidate_id, score, raw_metrics_json, created_at)
        candidate = self.get_active_candidate(candidate_id)
        run_id = candidate.run_id if candidate is not None else ""
        rows = self.list_evaluation_results(run_id, limit=500)
        return next(
            (row for row in rows if row.result_id == result_id),
            EvaluationResultRecord(
                result_id=result_id,
                run_id=run_id,
                candidate_id=candidate_id,
                score=score,
                raw_metrics=json.loads(raw_metrics_json),
                created_at=created_at,
            ),
        )

    def update_candidate_rolling_score(self, candidate_id: str, rolling_score: float, eval_count: int, last_eval_at: str, status: str = "active") -> ActiveCandidateRecord:
        self.client.call_reducer("update_candidate_rolling_score", candidate_id, rolling_score, eval_count, last_eval_at, status)
        candidate = self.get_active_candidate(candidate_id)
        if candidate is None:
            raise RuntimeError("Candidate not found after rolling score update.")
        return candidate

    def select_replacement_target(self, run_id: str) -> ActiveCandidateRecord | None:
        return choose_replacement_target(self.list_active_candidates(run_id))

    def spawn_offspring(self, run_id: str, candidate: ActiveCandidateRecord) -> ActiveCandidateRecord:
        self.client.call_reducer(
            "spawn_offspring",
            run_id,
            candidate.candidate_id,
            candidate.slot_index,
            candidate.variant,
            candidate.genome_blob,
            candidate.status,
            candidate.rolling_score,
            candidate.eval_count,
            candidate.birth_step,
            candidate.last_eval_at or "",
            stable_json_dumps(candidate.parent_ids),
            candidate.created_at,
        )
        created = self.get_active_candidate(candidate.candidate_id)
        if created is None:
            raise RuntimeError("Offspring not found after spawn.")
        return created

    def retire_candidate(self, candidate_id: str) -> ActiveCandidateRecord:
        self.client.call_reducer("retire_candidate", candidate_id, utc_now_iso())
        candidate = self.get_active_candidate(candidate_id)
        if candidate is None:
            raise RuntimeError("Candidate not found after retirement.")
        return candidate

    def activate_candidate(self, candidate_id: str) -> ActiveCandidateRecord:
        self.client.call_reducer("activate_candidate", candidate_id, utc_now_iso())
        candidate = self.get_active_candidate(candidate_id)
        if candidate is None:
            raise RuntimeError("Candidate not found after activation.")
        return candidate

    def promote_to_hall_of_fame(self, candidate_id: str, score: float | None = None) -> HallOfFameEntryRecord | None:
        candidate = self.get_active_candidate(candidate_id)
        if candidate is None:
            return None
        if any(entry.candidate_id == candidate_id for entry in self.list_hall_of_fame(candidate.run_id, limit=1000)):
            return None
        entry_id = f"{candidate.run_id}-hof-{uuid4().hex}"
        self.client.call_reducer("promote_to_hall_of_fame", entry_id, candidate.run_id, candidate_id, candidate.rolling_score if score is None else float(score), candidate.genome_blob, utc_now_iso())
        return next((entry for entry in self.list_hall_of_fame(candidate.run_id, limit=1000) if entry.entry_id == entry_id), None)

    def list_hall_of_fame(self, run_id: str, limit: int = 50) -> list[HallOfFameEntryRecord]:
        return queries.list_hall_of_fame(self.client, run_id, limit=limit)

    def list_evaluation_jobs(self, run_id: str, status: str | None = None, limit: int = 200) -> list[EvaluationJobRecord]:
        return queries.list_evaluation_jobs(self.client, run_id, status=status, limit=limit)

    def get_evaluation_job(self, job_id: str) -> EvaluationJobRecord | None:
        return queries.get_evaluation_job(self.client, job_id)

    def list_evaluation_results(self, run_id: str, limit: int = 200) -> list[EvaluationResultRecord]:
        return queries.list_evaluation_results(self.client, run_id, limit=limit)

    def list_candidate_lifecycle_events(self, run_id: str, candidate_id: str | None = None, limit: int = 100) -> list[CandidateLifecycleEventRecord]:
        return queries.list_candidate_lifecycle_events(self.client, run_id, candidate_id=candidate_id, limit=limit)

    def capture_online_metrics(self, run_id: str, replacement_count: int, success_rate_window: float) -> OnlineMetricRecord:
        active_rows = [candidate for candidate in self.list_active_candidates(run_id) if candidate.status != "retired"]
        scores = [candidate.rolling_score for candidate in active_rows]
        metric_id = f"{run_id}-metric-{uuid4().hex}"
        self.client.call_reducer("capture_online_metrics", metric_id, run_id, utc_now_iso(), len(active_rows), max(scores) if scores else 0.0, (sum(scores) / len(scores)) if scores else 0.0, int(replacement_count), float(success_rate_window))
        result = next((metric for metric in self.list_online_metrics(run_id, limit=1000) if metric.metric_id == metric_id), None)
        if result is None:
            raise RuntimeError(f"Online metric not found after write: {metric_id}")
        return result

    def list_online_metrics(self, run_id: str, limit: int = 100) -> list[OnlineMetricRecord]:
        return queries.list_online_metrics(self.client, run_id, limit=limit)

    def save_online_state(self, run_id: str, step: int, replacement_count: int, success_window_json: str, adapter_state_blob: str) -> OnlineStateRecord:
        self.client.call_reducer("upsert_online_state", run_id, step, replacement_count, success_window_json, adapter_state_blob, utc_now_iso())
        state = self.get_online_state(run_id)
        if state is None:
            raise RuntimeError("Online state not found after upsert.")
        return state

    def get_online_state(self, run_id: str) -> OnlineStateRecord | None:
        return queries.get_online_state(self.client, run_id)

    def resume_online_run(self, run_id: str) -> RunRecord:
        self.client.call_reducer("resume_online_run", run_id, utc_now_iso())
        run = self.get_run(run_id)
        if run is None:
            raise RuntimeError("Run not found after resume.")
        return run
