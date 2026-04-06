from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from config import (
    AppConfig,
    curriculum_phase_delay_labels,
    curriculum_phase_name,
    curriculum_switch_generation,
    evaluation_delay_label,
)
from db.models import GenerationCommitResult, RunRecord
from evolve.archive import build_archive_cells
from db.reducers import RunRepository
from evolve.candidate_features import CandidateFeatureContext, extract_candidate_features
from evolve.engine import EvolutionEngine, PopulationSnapshot
from evolve.evaluator import build_evaluator, score_ceiling_for_task
from evolve.plasticity import plastic_d_bounds_for_variant
from evolve.tensorneat_adapter import TensorNEATAdapter
from utils.seeds import set_global_seed
from utils.serialization import stable_json_dumps


class RunObserver(Protocol):
    def on_run_started(self, run: RunRecord) -> None:
        ...

    def on_generation_committed(self, result: GenerationCommitResult) -> None:
        ...

    def on_run_finished(self, run: RunRecord) -> None:
        ...


@dataclass(frozen=True)
class RunLoopResult:
    run: RunRecord
    final_generation: GenerationCommitResult


def execute_run(
    config: AppConfig,
    repository: RunRepository,
    observer: RunObserver | None = None,
    resume_run_id: str | None = None,
    benchmark_label: str | None = None,
) -> RunLoopResult:
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
    engine = EvolutionEngine(adapter=adapter, evaluator=evaluator)

    if resume_run_id is None:
        run, population, next_generation_id = _start_new_run(config, repository, adapter, engine)
    else:
        run, population, next_generation_id = _resume_run(config, repository, adapter, engine, resume_run_id)
        repository.append_event(
            run.run_id,
            "run_resumed",
            stable_json_dumps({"next_generation_id": next_generation_id}),
        )

    effective_benchmark_label = benchmark_label or run.run_id
    task_score_ceiling = score_ceiling_for_task(config.task)
    curriculum_phase_1_delays, curriculum_phase_2_delays = curriculum_phase_delay_labels(config.task)
    curriculum_switch = curriculum_switch_generation(config.task) if config.task.curriculum_enabled else 0
    plastic_d_lower_bound, _plastic_d_upper_bound = plastic_d_bounds_for_variant(
        config.run.variant,
        default_lower_bound=config.mutation.plastic_d_lower_bound,
        default_upper_bound=config.mutation.plastic_d_upper_bound,
    )

    if observer is not None:
        observer.on_run_started(run)

    final_commit = _latest_commit(repository, run.run_id)
    first_success_generation = _initial_first_success_generation(repository, run.run_id)
    if next_generation_id >= config.run.generations:
        finished_run = repository.finish_run(run.run_id)
        if observer is not None:
            observer.on_run_finished(finished_run)
        if final_commit is None:
            raise RuntimeError("Run has no committed generation to return.")
        return RunLoopResult(run=finished_run, final_generation=final_commit)

    for generation_id in range(next_generation_id, config.run.generations):
        if population is None:
            raise RuntimeError(f"Missing population for generation {generation_id}")
        engine.evaluator = build_evaluator(
            config.task,
            variant=config.run.variant,
            delta_w_clamp=config.plasticity.delta_w_clamp,
            generation_id=generation_id,
        )
        evaluation_started = time.perf_counter()
        fitness_records = engine.evaluate_population(population)
        evaluation_duration_ms = _elapsed_ms(evaluation_started)
        candidate_by_id = {bundle.record.candidate_id: bundle for bundle in population.candidates}

        for record in fitness_records:
            first_success_generation = _next_first_success_generation(
                current=first_success_generation,
                generation_id=record.generation_id,
                success=bool(record.raw_metrics.get("success", False)),
            )
            repository.record_fitness(
                candidate_id=record.candidate_id,
                run_id=record.run_id,
                generation_id=record.generation_id,
                score=record.score,
                raw_metrics_json=stable_json_dumps(record.raw_metrics),
            )
            candidate_bundle = candidate_by_id[record.candidate_id]
            feature_record, vector_record = extract_candidate_features(
                candidate_bundle.genome,
                record.raw_metrics,
                CandidateFeatureContext(
                    candidate_id=record.candidate_id,
                    run_id=record.run_id,
                    benchmark_label=effective_benchmark_label,
                    task_name=config.task.name,
                    delay_steps=config.task.temporal_delay_steps,
                    variant=config.run.variant,
                    seed=config.run.seed,
                    generation=record.generation_id,
                    final_max_score=record.score,
                    first_success_generation=(
                        first_success_generation if bool(record.raw_metrics.get("success", False)) else None
                    ),
                    eta_lower_bound=config.mutation.eta_lower_bound,
                    eta_upper_bound=config.mutation.eta_upper_bound,
                    plastic_d_lower_bound=plastic_d_lower_bound,
                    curriculum_enabled=config.task.curriculum_enabled,
                    curriculum_phase_1_delays=curriculum_phase_1_delays,
                    curriculum_phase_2_delays=curriculum_phase_2_delays,
                    curriculum_switch_generation=curriculum_switch,
                    curriculum_phase=curriculum_phase_name(config.task, generation_id=generation_id),
                    active_evaluation_delays=evaluation_delay_label(config.task, generation_id=generation_id),
                ),
            )
            repository.upsert_candidate_features(feature_record)
            repository.upsert_candidate_feature_vector(vector_record)
            for archive_cell in build_archive_cells(
                feature_record,
                score_ceiling=task_score_ceiling,
            ):
                repository.consider_archive_candidate(archive_cell)

        repository.mark_generation_ready(
            run.run_id,
            generation_id,
            eval_duration_ms=evaluation_duration_ms,
        )
        commit_started = time.perf_counter()
        final_commit = _commit_generation_with_timing(
            repository=repository,
            run_id=run.run_id,
            generation_id=generation_id,
            commit_started=commit_started,
        )
        if observer is not None:
            observer.on_generation_committed(final_commit)
        for elite in final_commit.elites:
            repository.mark_hof_candidate(elite.candidate_id, hof_flag=True)

        if generation_id + 1 >= config.run.generations:
            break

        population = _prepare_next_population(
            repository=repository,
            adapter=adapter,
            engine=engine,
            run_id=run.run_id,
            source_generation_id=generation_id,
            next_generation_id=generation_id + 1,
        )

    finished_run = repository.finish_run(run.run_id)
    if observer is not None:
        observer.on_run_finished(finished_run)

    if final_commit is None:
        raise RuntimeError("Run finished without committing any generation.")
    return RunLoopResult(run=finished_run, final_generation=final_commit)


def _start_new_run(
    config: AppConfig,
    repository: RunRepository,
    adapter: TensorNEATAdapter,
    engine: EvolutionEngine,
) -> tuple[RunRecord, PopulationSnapshot | None, int]:
    run = repository.create_run(
        task_name=config.task.name,
        seed=config.run.seed,
        config_json=stable_json_dumps(config.to_dict()),
    )
    state = adapter.initialize(config.run.seed)
    parent_ids: list[list[str]] = [[] for _ in range(adapter.population_size)]
    repository.save_checkpoint(
        run.run_id,
        0,
        adapter.serialize_state(state),
        stable_json_dumps(parent_ids),
    )
    population = engine.prepare_population(run.run_id, 0, state, parent_ids)
    repository.insert_population(run.run_id, 0, [bundle.record for bundle in population.candidates])
    return run, population, 0


def _resume_run(
    config: AppConfig,
    repository: RunRepository,
    adapter: TensorNEATAdapter,
    engine: EvolutionEngine,
    run_id: str,
) -> tuple[RunRecord, PopulationSnapshot | None, int]:
    run = repository.get_run(run_id)
    if run is None:
        raise ValueError(f"Run not found: {run_id}")

    generations = repository.list_generations(run_id)
    open_generations = [generation for generation in generations if generation.state != "committed"]
    if open_generations:
        generation_id = min(generation.generation_id for generation in open_generations)
        population = _restore_population(repository, adapter, engine, run_id, generation_id)
        return run, population, generation_id

    if not generations:
        population = _restore_population(repository, adapter, engine, run_id, 0)
        return run, population, 0

    last_generation = _latest_generation_record(generations)
    if last_generation.generation_id + 1 >= config.run.generations:
        return run, None, config.run.generations

    next_generation_id = last_generation.generation_id + 1
    checkpoint = repository.get_checkpoint(run_id, next_generation_id)
    candidate_records = repository.list_candidates(run_id, next_generation_id)
    if checkpoint is not None:
        population = _restore_population(repository, adapter, engine, run_id, next_generation_id)
        if not candidate_records:
            repository.create_next_generation(
                run_id,
                next_generation_id,
                [bundle.record for bundle in population.candidates],
            )
        return run, population, next_generation_id

    population = _prepare_next_population(
        repository=repository,
        adapter=adapter,
        engine=engine,
        run_id=run_id,
        source_generation_id=last_generation.generation_id,
        next_generation_id=next_generation_id,
    )
    return run, population, next_generation_id


def _prepare_next_population(
    repository: RunRepository,
    adapter: TensorNEATAdapter,
    engine: EvolutionEngine,
    run_id: str,
    source_generation_id: int,
    next_generation_id: int,
) -> PopulationSnapshot:
    checkpoint = repository.get_checkpoint(run_id, source_generation_id)
    if checkpoint is None:
        raise RuntimeError(f"Missing checkpoint for generation {source_generation_id}")

    source_state = adapter.deserialize_state(checkpoint.state_blob)
    source_candidates = repository.list_candidates(run_id, source_generation_id)
    if not source_candidates:
        raise RuntimeError(f"Missing candidates for generation {source_generation_id}")

    fitness_by_candidate = {
        record.candidate_id: record.score
        for record in repository.list_fitness(run_id, source_generation_id)
    }
    scores = _scores_for_source_candidates(
        source_candidates=source_candidates,
        fitness_by_candidate=fitness_by_candidate,
        generation_id=source_generation_id,
    )
    candidate_ids = [candidate.candidate_id for candidate in source_candidates]
    next_state, parent_ids = adapter.advance(source_state, scores, candidate_ids)
    repository.save_checkpoint(
        run_id,
        next_generation_id,
        adapter.serialize_state(next_state),
        stable_json_dumps(parent_ids),
    )
    population = engine.prepare_population(run_id, next_generation_id, next_state, parent_ids)
    repository.create_next_generation(
        run_id,
        next_generation_id,
        [bundle.record for bundle in population.candidates],
    )
    return population


def _restore_population(
    repository: RunRepository,
    adapter: TensorNEATAdapter,
    engine: EvolutionEngine,
    run_id: str,
    generation_id: int,
) -> PopulationSnapshot:
    checkpoint = repository.get_checkpoint(run_id, generation_id)
    if checkpoint is None:
        raise RuntimeError(f"Missing checkpoint for generation {generation_id}")

    candidate_records = repository.list_candidates(run_id, generation_id)
    if candidate_records:
        return engine.population_from_records(run_id, generation_id, candidate_records)

    state = adapter.deserialize_state(checkpoint.state_blob)
    parent_ids = json.loads(checkpoint.parent_ids_json)
    population = engine.prepare_population(run_id, generation_id, state, parent_ids)
    if generation_id == 0:
        repository.insert_population(run_id, generation_id, [bundle.record for bundle in population.candidates])
    else:
        repository.create_next_generation(run_id, generation_id, [bundle.record for bundle in population.candidates])
    return population


def _latest_commit(repository: RunRepository, run_id: str) -> GenerationCommitResult | None:
    generations = [generation for generation in repository.list_generations(run_id) if generation.state == "committed"]
    if not generations:
        return None
    generation = _latest_generation_record(generations)
    elites = repository.list_elites(run_id, generation_id=generation.generation_id, limit=100)
    return GenerationCommitResult(generation=generation, elites=tuple(elites))


def _initial_first_success_generation(repository: RunRepository, run_id: str) -> int | None:
    generations = [
        generation
        for generation in repository.list_generations(run_id)
        if generation.state == "committed"
    ]
    success_generations = [
        generation.generation_id
        for generation in generations
        if any(
            bool(record.raw_metrics.get("success", False))
            for record in repository.list_fitness(run_id, generation.generation_id)
        )
    ]
    if not success_generations:
        return None
    return min(success_generations)


def _next_first_success_generation(
    *,
    current: int | None,
    generation_id: int,
    success: bool,
) -> int | None:
    if not success:
        return current
    if current is None:
        return generation_id
    return min(current, generation_id)


def _latest_generation_record(generations):
    return max(generations, key=lambda generation: generation.generation_id)


def _scores_for_source_candidates(
    *,
    source_candidates,
    fitness_by_candidate: dict[str, float],
    generation_id: int,
) -> np.ndarray:
    missing_candidate_ids = [
        candidate.candidate_id
        for candidate in source_candidates
        if candidate.candidate_id not in fitness_by_candidate
    ]
    if missing_candidate_ids:
        missing_ids = ", ".join(sorted(missing_candidate_ids))
        raise RuntimeError(
            f"Missing fitness records for generation {generation_id}: {missing_ids}"
        )
    return np.array(
        [fitness_by_candidate[candidate.candidate_id] for candidate in source_candidates],
        dtype=np.float32,
    )


def _commit_generation_with_timing(
    *,
    repository: RunRepository,
    run_id: str,
    generation_id: int,
    commit_started: float,
) -> GenerationCommitResult:
    initial_commit = repository.commit_generation(run_id, generation_id)
    commit_duration_ms = _elapsed_ms(commit_started)
    if initial_commit.generation.commit_duration_ms is not None:
        return initial_commit
    # Commit timing is only known after the initial reducer call finishes, so we
    # persist it in a follow-up metadata update for repositories that support it.
    return repository.commit_generation(
        run_id,
        generation_id,
        commit_duration_ms=commit_duration_ms,
    )


def _elapsed_ms(started_at: float) -> int:
    return int(round((time.perf_counter() - started_at) * 1000.0))
