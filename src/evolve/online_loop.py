from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from config import AppConfig
from db.models import (
    ActiveCandidateRecord,
    EvaluationJobRecord,
    EvaluationResultRecord,
    HallOfFameEntryRecord,
    OnlineMetricRecord,
    OnlineStateRecord,
    RunRecord,
)
from db.online_repository import OnlineRunRepository
from evolve.candidate_registry import CandidateRegistry
from evolve.evaluator import build_evaluator, score_ceiling_for_task
from evolve.replacement import choose_parent_ids
from evolve.rolling_metrics import success_rate_window, success_window_append
from evolve.rtneat_scheduler import RTNEATScheduler
from evolve.tensorneat_adapter import TensorNEATAdapter
from utils.scoring import is_success_score, resolve_success, update_exponential_rolling_score
from utils.seeds import set_global_seed
from utils.serialization import stable_json_dumps, utc_now_iso


class OnlineObserver(Protocol):
    def on_run_started(self, run: RunRecord) -> None:
        ...

    def on_job_finished(
        self,
        job: EvaluationJobRecord,
        result: EvaluationResultRecord,
        candidate: ActiveCandidateRecord,
    ) -> None:
        ...

    def on_replacement(
        self,
        retired_candidate: ActiveCandidateRecord,
        offspring_candidate: ActiveCandidateRecord,
        hall_of_fame_entry: HallOfFameEntryRecord | None,
    ) -> None:
        ...

    def on_metrics(self, metric: OnlineMetricRecord) -> None:
        ...

    def on_run_finished(self, run: RunRecord) -> None:
        ...


@dataclass(frozen=True)
class OnlineRunResult:
    run: RunRecord
    step: int
    last_metric: OnlineMetricRecord | None


ACTIVE_ONLINE_STATUSES = {"created", "queued", "evaluating", "active"}


def execute_online_run(
    config: AppConfig,
    repository: OnlineRunRepository,
    observer: OnlineObserver | None = None,
    resume_run_id: str | None = None,
) -> OnlineRunResult:
    config = AppConfig.from_dict(
        {
            **config.to_dict(),
            "evolution": {
                **config.evolution.__dict__,
                "population_size": config.online.active_population_size,
            },
            "run": {
                **config.run.__dict__,
                "mode": "online",
            },
        }
    )
    set_global_seed(config.run.seed)
    evaluator = build_evaluator(
        config.task,
        variant=config.run.variant,
        delta_w_clamp=config.plasticity.delta_w_clamp,
    )
    adapter = TensorNEATAdapter(
        config=config,
        num_inputs=evaluator.input_size,
        num_outputs=evaluator.output_size,
    )
    scheduler = RTNEATScheduler(adapter=adapter, variant=config.run.variant)
    task_payload_json = stable_json_dumps(
        {
            "task_name": config.task.name,
            "activation_steps": config.task.activation_steps,
            "temporal_delay_steps": config.task.temporal_delay_steps,
        }
    )
    score_ceiling = score_ceiling_for_task(config.task)

    if resume_run_id is None:
        run, registry, state_record = _start_online_run(config, repository, adapter)
        success_window_values: list[bool] = []
        replacement_count = 0
        step = 0
    else:
        run, registry, state_record = _resume_online_run(
            config,
            repository,
            adapter,
            resume_run_id,
            score_ceiling=score_ceiling,
        )
        success_window_values = json.loads(state_record.success_window_json)
        replacement_count = state_record.replacement_count
        step = state_record.step

    state = adapter.deserialize_state(state_record.adapter_state_blob)
    _notify_observer(observer, "on_run_started", run)

    last_metric: OnlineMetricRecord | None = None
    while step < config.online.max_steps:
        _ensure_open_jobs(repository, run.run_id, registry.list_records(), task_payload_json)
        queued_job_count = len(repository.list_evaluation_jobs(run.run_id, status="queued", limit=10000))
        _notify_observer(observer, "on_jobs_queued", run, step, queued_job_count)
        job = repository.claim_job(run.run_id, config.online.worker_id)
        if job is None:
            break

        bundle = registry.get(job.candidate_id)
        _notify_observer(observer, "on_job_claimed", job, bundle.record)
        evaluation = evaluator.evaluate(bundle.genome)
        result = repository.submit_result(
            job.job_id,
            job.candidate_id,
            evaluation.score,
            stable_json_dumps(evaluation.raw_metrics),
        )
        _notify_observer(observer, "on_result_submitted", job, result, bundle.record)
        success = _extract_success(evaluation.raw_metrics, evaluation.score, score_ceiling)
        success_window_values = success_window_append(success_window_values, success, config.online.success_window)
        updated_candidate = repository.update_candidate_rolling_score(
            bundle.record.candidate_id,
            rolling_score=update_exponential_rolling_score(
                bundle.record.rolling_score,
                evaluation.score,
                config.online.rolling_score_alpha,
                bundle.record.eval_count,
            ),
            eval_count=bundle.record.eval_count + 1,
            last_eval_at=result.created_at,
            status="active",
        )
        registry.update_record(updated_candidate)

        promoted_entry = _maybe_promote_candidate(repository, updated_candidate, score_ceiling, success)
        if promoted_entry is not None:
            _notify_observer(observer, "on_candidate_promoted", updated_candidate, promoted_entry)

        step += 1

        retired_candidate = None
        offspring_candidate = None
        if step % config.online.replacement_interval == 0:
            retired_candidate, offspring_candidate, state = _perform_replacement(
                config=config,
                repository=repository,
                registry=registry,
                scheduler=scheduler,
                state=state,
                run=run,
                birth_step=step,
                task_payload_json=task_payload_json,
                observer=observer,
            )
            if retired_candidate is not None and offspring_candidate is not None:
                replacement_count += 1

        if step % config.online.metrics_interval == 0 or step >= config.online.max_steps:
            last_metric = repository.capture_online_metrics(
                run.run_id,
                replacement_count=replacement_count,
                success_rate_window=success_rate_window(success_window_values),
            )

        state = adapter.sync_population_state(state, *registry.population_arrays(adapter.genome))
        repository.save_online_state(
            run.run_id,
            step=step,
            replacement_count=replacement_count,
            success_window_json=stable_json_dumps(success_window_values),
            adapter_state_blob=adapter.serialize_state(state),
        )
        _notify_observer(observer, "on_job_finished", job, result, updated_candidate)
        if retired_candidate is not None and offspring_candidate is not None:
            _notify_replacement_observer(observer, retired_candidate, offspring_candidate)
        if last_metric is not None and (step % config.online.metrics_interval == 0 or step >= config.online.max_steps):
            _notify_observer(observer, "on_metrics", last_metric)

    finished_run = repository.finish_run(run.run_id)
    _notify_observer(observer, "on_run_finished", finished_run)
    return OnlineRunResult(run=finished_run, step=step, last_metric=last_metric)


def _start_online_run(
    config: AppConfig,
    repository: OnlineRunRepository,
    adapter: TensorNEATAdapter,
) -> tuple[RunRecord, CandidateRegistry, OnlineStateRecord]:
    run = repository.create_online_run(config.task.name, config.run.seed, stable_json_dumps(config.to_dict()))
    seed_state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(seed_state)
    created_at = utc_now_iso()
    registry = CandidateRegistry.from_seed_population(
        genome_template=adapter.genome,
        run_id=run.run_id,
        variant=config.run.variant,
        pop_nodes=np.asarray(pop_nodes, dtype=np.float32),
        pop_conns=np.asarray(pop_conns, dtype=np.float32),
        birth_step=0,
        created_at=created_at,
    )
    repository.seed_active_population(run.run_id, registry.list_records())
    synchronized = adapter.sync_population_state(seed_state, *registry.population_arrays(adapter.genome))
    state_record = repository.save_online_state(
        run.run_id,
        step=0,
        replacement_count=0,
        success_window_json=stable_json_dumps([]),
        adapter_state_blob=adapter.serialize_state(synchronized),
    )
    return run, registry, state_record


def _resume_online_run(
    config: AppConfig,
    repository: OnlineRunRepository,
    adapter: TensorNEATAdapter,
    run_id: str,
    *,
    score_ceiling: float,
) -> tuple[RunRecord, CandidateRegistry, OnlineStateRecord]:
    run = repository.resume_online_run(run_id)
    state_record = repository.get_online_state(run_id)
    if state_record is None:
        raise RuntimeError(f"Missing online state for run {run_id}")
    _restore_missing_active_slots(config, repository, run_id)
    state_record = _reconcile_online_state(config, repository, run_id, state_record, score_ceiling)
    active_candidates = repository.list_active_candidates(run_id, statuses=sorted(ACTIVE_ONLINE_STATUSES))
    if not active_candidates:
        raise RuntimeError(f"Missing active candidates for run {run_id}")
    registry = CandidateRegistry.from_records(active_candidates)
    restored_state = adapter.deserialize_state(state_record.adapter_state_blob)
    synchronized = adapter.sync_population_state(restored_state, *registry.population_arrays(adapter.genome))
    state_record = repository.save_online_state(
        run_id,
        step=state_record.step,
        replacement_count=state_record.replacement_count,
        success_window_json=state_record.success_window_json,
        adapter_state_blob=adapter.serialize_state(synchronized),
    )
    return run, registry, state_record


def _ensure_open_jobs(
    repository: OnlineRunRepository,
    run_id: str,
    active_candidates: list[ActiveCandidateRecord],
    task_payload_json: str,
) -> None:
    existing_jobs = repository.list_evaluation_jobs(run_id, limit=10000)
    open_candidate_ids = {
        job.candidate_id
        for job in existing_jobs
        if job.status in {"queued", "claimed"}
    }
    for candidate in active_candidates:
        if candidate.status == "retired":
            continue
        if candidate.candidate_id in open_candidate_ids:
            continue
        repository.enqueue_evaluation(run_id, candidate.candidate_id, task_payload_json)


def _perform_replacement(
    *,
    config: AppConfig,
    repository: OnlineRunRepository,
    registry: CandidateRegistry,
    scheduler: RTNEATScheduler,
    state: object,
    run: RunRecord,
    birth_step: int,
    task_payload_json: str,
    observer: OnlineObserver | None = None,
) -> tuple[ActiveCandidateRecord | None, ActiveCandidateRecord | None, object]:
    target = repository.select_replacement_target(run.run_id)
    if target is None:
        return None, None, state
    parent_ids = choose_parent_ids(registry.list_records(), exclude_candidate_id=target.candidate_id)
    if not parent_ids:
        return None, None, state
    retired = repository.retire_candidate(target.candidate_id)
    _notify_observer(observer, "on_candidate_retired", retired)
    spawn = scheduler.spawn_offspring(
        state=state,
        registry=registry,
        run_id=run.run_id,
        slot_index=target.slot_index,
        parent_ids=parent_ids,
        birth_step=birth_step,
    )
    spawned = repository.spawn_offspring(run.run_id, spawn.record)
    activated = repository.activate_candidate(spawned.candidate_id)
    registry.replace_slot(target.slot_index, activated, spawn.genome_bundle.genome)
    repository.enqueue_evaluation(run.run_id, activated.candidate_id, task_payload_json)
    return retired, activated, spawn.state


def _extract_success(raw_metrics: dict[str, object], score: float, score_ceiling: float) -> bool:
    return resolve_success(raw_metrics, score, score_ceiling)


def _maybe_promote_candidate(
    repository: OnlineRunRepository,
    candidate: ActiveCandidateRecord,
    score_ceiling: float,
    success: bool,
) -> HallOfFameEntryRecord | None:
    if success or is_success_score(candidate.rolling_score, score_ceiling):
        return repository.promote_to_hall_of_fame(candidate.candidate_id, score=candidate.rolling_score)
    hall_of_fame = repository.list_hall_of_fame(candidate.run_id, limit=1)
    if not hall_of_fame:
        return None
    if candidate.rolling_score > hall_of_fame[0].score:
        return repository.promote_to_hall_of_fame(candidate.candidate_id, score=candidate.rolling_score)
    return None


def _restore_missing_active_slots(
    config: AppConfig,
    repository: OnlineRunRepository,
    run_id: str,
) -> None:
    all_candidates = repository.list_active_candidates(run_id)
    active_slots = {
        candidate.slot_index
        for candidate in all_candidates
        if candidate.status in ACTIVE_ONLINE_STATUSES
    }
    for slot_index in range(config.online.active_population_size):
        if slot_index in active_slots:
            continue
        retired_candidates = [
            candidate
            for candidate in all_candidates
            if candidate.slot_index == slot_index and candidate.status == "retired"
        ]
        if not retired_candidates:
            raise RuntimeError(f"Missing recoverable candidate for slot {slot_index} in run {run_id}")
        fallback = sorted(
            retired_candidates,
            key=lambda candidate: (candidate.birth_step, candidate.created_at, candidate.candidate_id),
            reverse=True,
        )[0]
        repository.activate_candidate(fallback.candidate_id)
        active_slots.add(slot_index)


def _reconcile_online_state(
    config: AppConfig,
    repository: OnlineRunRepository,
    run_id: str,
    state_record: OnlineStateRecord,
    score_ceiling: float,
) -> OnlineStateRecord:
    results = repository.list_evaluation_results(
        run_id,
        limit=max(config.online.max_steps * max(config.online.active_population_size, 1), 1000),
    )
    ordered_results = sorted(results, key=lambda result: (result.created_at, result.result_id))
    success_window_values: list[bool] = []
    for result in ordered_results:
        success_window_values = success_window_append(
            success_window_values,
            resolve_success(result.raw_metrics, result.score, score_ceiling),
            config.online.success_window,
        )

    results_by_candidate: dict[str, list[EvaluationResultRecord]] = {}
    for result in ordered_results:
        results_by_candidate.setdefault(result.candidate_id, []).append(result)

    all_candidates = {
        candidate.candidate_id: candidate
        for candidate in repository.list_active_candidates(run_id)
    }
    for candidate_id, candidate_results in results_by_candidate.items():
        candidate = all_candidates.get(candidate_id)
        if candidate is None:
            continue
        rolling_score = 0.0
        eval_count = 0
        last_eval_at: str | None = None
        seen_success = False
        for result in candidate_results:
            rolling_score = update_exponential_rolling_score(
                rolling_score,
                result.score,
                config.online.rolling_score_alpha,
                eval_count,
            )
            eval_count += 1
            last_eval_at = result.created_at
            seen_success = seen_success or resolve_success(result.raw_metrics, result.score, score_ceiling)

        next_status = "active" if candidate.status == "evaluating" else candidate.status
        if (
            abs(candidate.rolling_score - rolling_score) > 1e-6
            or candidate.eval_count != eval_count
            or candidate.last_eval_at != last_eval_at
            or candidate.status != next_status
        ):
            candidate = repository.update_candidate_rolling_score(
                candidate_id,
                rolling_score=rolling_score,
                eval_count=eval_count,
                last_eval_at=last_eval_at or "",
                status=next_status,
            )
            all_candidates[candidate_id] = candidate

        if candidate.status != "retired":
            _maybe_promote_candidate(repository, candidate, score_ceiling, seen_success)

    return repository.save_online_state(
        run_id,
        step=max(state_record.step, len(ordered_results)),
        replacement_count=state_record.replacement_count,
        success_window_json=stable_json_dumps(success_window_values),
        adapter_state_blob=state_record.adapter_state_blob,
    )


def _notify_observer(observer: OnlineObserver | None, method_name: str, *args: Any) -> None:
    if observer is None:
        return
    callback = getattr(observer, method_name, None)
    if callback is None:
        return
    callback(*args)


def _notify_replacement_observer(
    observer: OnlineObserver | None,
    retired_candidate: ActiveCandidateRecord,
    offspring_candidate: ActiveCandidateRecord,
) -> None:
    _notify_observer(observer, "on_replacement", retired_candidate, offspring_candidate, None)
