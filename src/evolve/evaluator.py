from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from config import (
    TaskConfig,
    curriculum_phase_delay_labels,
    curriculum_phase_name,
    curriculum_switch_generation,
    key_value_profile_labels,
    resolved_evaluation_delay_steps,
    resolved_key_value_profile,
)
from evolve.custom_neuron import (
    AdaptivePlasticNetworkExecutor,
    ContentGatedNetworkExecutor,
    PlasticNetworkExecutor,
    PlasticityEpisodeMetrics,
    StatefulNetworkExecutor,
    StatefulV4SlotsNetworkExecutor,
    StatefulV5AddressedSlotsNetworkExecutor,
    StatefulV6DeltaMemoryNetworkExecutor,
    StatefulV2GatedNetworkExecutor,
    StatefulV3KVNetworkExecutor,
    StatefulV2NetworkExecutor,
)
from evolve.genome_codec import GenomeModel
from evolve.plasticity import (
    is_content_gated_variant,
    is_stateful_v4_slots_variant,
    is_stateful_v5_addressed_slots_variant,
    is_stateful_v6_delta_memory_variant,
    is_stateful_v3_kv_variant,
    is_stateful_v2_gated_variant,
    is_stateful_v2_variant,
    plastic_mode_for_variant,
)
from tasks.bit_memory import BitMemoryTask
from tasks.delayed_xor import DelayedXorTask
from tasks.event_decision import EventDecisionTask
from tasks.event_memory import EventMemoryTask
from tasks.key_value_memory import KeyValueMemoryTask
from tasks.xor import XorTask


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    raw_metrics: dict[str, Any]


@dataclass(frozen=True)
class TemporalSequenceEvaluation:
    score: float
    sequence_mse: float
    final_mae: float
    final_accuracy: float
    success: bool
    raw_outputs: np.ndarray
    predictions: np.ndarray
    final_predictions: np.ndarray
    final_targets: np.ndarray
    episode_metrics: PlasticityEpisodeMetrics | None
    task_metrics: dict[str, Any] = field(default_factory=dict)


class GenomeEvaluator(Protocol):
    task_name: str
    input_size: int
    output_size: int
    score_ceiling: float

    def evaluate(self, genome: GenomeModel) -> EvaluationResult:
        ...


class XorEvaluator:
    task_name = "xor"

    def __init__(
        self,
        task: XorTask,
        activation_steps: int,
        variant: str = "stateful",
        *,
        delta_w_clamp: float = 1.0,
    ) -> None:
        self.task = task
        self.input_size = task.input_size
        self.output_size = task.output_size
        self.score_ceiling = 4.0
        self.executor = _build_executor(
            variant=variant,
            activation_steps=activation_steps,
            delta_w_clamp=delta_w_clamp,
        )

    @staticmethod
    def score_predictions(
        predictions: np.ndarray,
        targets: np.ndarray,
        score_ceiling: float = 4.0,
    ) -> float:
        absolute_error = float(np.abs(predictions - targets).sum())
        return _clamp_score(score_ceiling, absolute_error)

    def evaluate(self, genome: GenomeModel) -> EvaluationResult:
        raw_outputs, episode_metrics = _run_samples(
            executor=self.executor,
            genome=genome,
            samples=self.task.inputs,
            sequence_mode=False,
        )
        predictions = _bounded_predictions(raw_outputs)
        score = self.score_predictions(
            predictions,
            self.task.targets,
            score_ceiling=self.score_ceiling,
        )
        mse = float(np.mean((predictions - self.task.targets) ** 2))
        mae = float(np.mean(np.abs(predictions - self.task.targets)))
        classes = (predictions >= 0.5).astype(np.float32)
        accuracy = float(np.mean(classes == self.task.targets))

        return EvaluationResult(
            score=score,
            raw_metrics={
                "predictions": predictions.flatten().round(6).tolist(),
                "raw_outputs": raw_outputs.flatten().round(6).tolist(),
                "mse": mse,
                "mae": mae,
                "accuracy": accuracy,
                "score_ceiling": self.score_ceiling,
                "success": bool(np.all(classes == self.task.targets)),
                **_executor_metrics(episode_metrics),
            },
        )


class DelayedXorEvaluator:
    task_name = "delayed_xor"

    def __init__(
        self,
        task: DelayedXorTask,
        activation_steps: int,
        variant: str = "stateful",
        *,
        delta_w_clamp: float = 1.0,
    ) -> None:
        self.task = task
        self.input_size = task.input_size
        self.output_size = task.output_size
        self.score_ceiling = float(task.target_sequences.shape[0] * task.target_sequences.shape[2])
        self.executor = _build_executor(
            variant=variant,
            activation_steps=activation_steps,
            delta_w_clamp=delta_w_clamp,
        )

    def score_predictions(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        absolute_error = float(np.abs(predictions - targets).sum())
        return _clamp_score(self.score_ceiling, absolute_error)

    def evaluate(self, genome: GenomeModel) -> EvaluationResult:
        raw_outputs, episode_metrics = _run_samples(
            executor=self.executor,
            genome=genome,
            samples=self.task.input_sequences,
            sequence_mode=True,
        )
        predictions = _bounded_predictions(raw_outputs)
        final_predictions = predictions[:, -1:, :]
        final_targets = self.task.target_sequences[:, -1:, :]
        score = self.score_predictions(final_predictions, final_targets)
        sequence_mse = float(np.mean((predictions - self.task.target_sequences) ** 2))
        final_mae = float(np.mean(np.abs(final_predictions - final_targets)))
        final_classes = (final_predictions >= 0.5).astype(np.float32)
        final_accuracy = float(np.mean(final_classes == final_targets))

        return EvaluationResult(
            score=score,
            raw_metrics={
                "sequence_predictions": predictions.round(6).tolist(),
                "sequence_raw_outputs": raw_outputs.round(6).tolist(),
                "final_predictions": final_predictions.flatten().round(6).tolist(),
                "sequence_mse": sequence_mse,
                "final_mae": final_mae,
                "final_accuracy": final_accuracy,
                "score_ceiling": self.score_ceiling,
                "success": bool(np.all(final_classes == final_targets)),
                **_executor_metrics(episode_metrics),
            },
        )


class BitMemoryEvaluator:
    task_name = "bit_memory"

    def __init__(
        self,
        task: BitMemoryTask,
        activation_steps: int,
        variant: str = "stateful",
        *,
        delta_w_clamp: float = 1.0,
        evaluation_delay_steps: Sequence[int] | None = None,
        curriculum_phase: str = "static",
        curriculum_enabled: bool = False,
        curriculum_phase_1_delays: str = "",
        curriculum_phase_2_delays: str = "",
        curriculum_switch_generation: int = 0,
    ) -> None:
        self.task = task
        self.evaluation_delay_steps = tuple(int(delay_steps) for delay_steps in (evaluation_delay_steps or (task.delay_steps,)))
        self.curriculum_phase = curriculum_phase
        self.curriculum_enabled = bool(curriculum_enabled)
        self.curriculum_phase_1_delays = curriculum_phase_1_delays
        self.curriculum_phase_2_delays = curriculum_phase_2_delays
        self.curriculum_switch_generation = int(curriculum_switch_generation)
        self.active_evaluation_delays = ",".join(str(delay_steps) for delay_steps in self.evaluation_delay_steps)
        self.tasks_by_delay = {
            int(delay_steps): (task if int(delay_steps) == int(task.delay_steps) else BitMemoryTask.create(delay_steps=int(delay_steps)))
            for delay_steps in self.evaluation_delay_steps
        }
        self.input_size = task.input_size
        self.output_size = task.output_size
        self.score_ceiling = float(task.target_sequences.shape[0] * task.target_sequences.shape[2])
        self.executor = _build_executor(
            variant=variant,
            activation_steps=activation_steps,
            delta_w_clamp=delta_w_clamp,
        )

    def score_predictions(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        absolute_error = float(np.abs(predictions - targets).sum())
        return _clamp_score(self.score_ceiling, absolute_error)

    def evaluate(self, genome: GenomeModel) -> EvaluationResult:
        evaluations = {
            delay_steps: self._evaluate_task(genome, task)
            for delay_steps, task in self.tasks_by_delay.items()
        }
        if len(evaluations) == 1:
            delay_steps, evaluation = next(iter(evaluations.items()))
            return EvaluationResult(
                score=evaluation.score,
                raw_metrics={
                    "sequence_predictions": evaluation.predictions.round(6).tolist(),
                    "sequence_raw_outputs": evaluation.raw_outputs.round(6).tolist(),
                    "final_predictions": evaluation.final_predictions.flatten().round(6).tolist(),
                    "sequence_mse": evaluation.sequence_mse,
                    "recall_mae": evaluation.final_mae,
                    "final_accuracy": evaluation.final_accuracy,
                    "score_ceiling": self.score_ceiling,
                    "success": evaluation.success,
                    "joint_success": evaluation.success,
                    "evaluation_delay_steps": [int(delay_steps)],
                    "active_evaluation_delays": self.active_evaluation_delays,
                    "curriculum_enabled": self.curriculum_enabled,
                    "curriculum_phase_1_delays": self.curriculum_phase_1_delays,
                    "curriculum_phase_2_delays": self.curriculum_phase_2_delays,
                    "curriculum_switch_generation": self.curriculum_switch_generation,
                    "curriculum_phase": self.curriculum_phase,
                    "score_current_phase": evaluation.score,
                    f"score_delay_{delay_steps}": evaluation.score,
                    f"success_delay_{delay_steps}": evaluation.success,
                    "mean_score_over_delays": evaluation.score,
                    "delay_score_std": 0.0,
                    "delay_score_range": 0.0,
                    **_executor_metrics(evaluation.episode_metrics),
                },
            )

        scores = [evaluation.score for evaluation in evaluations.values()]
        success_flags = [evaluation.success for evaluation in evaluations.values()]
        aggregate_metrics = _aggregate_episode_metrics(
            [evaluation.episode_metrics for evaluation in evaluations.values() if evaluation.episode_metrics is not None]
        )
        raw_metrics: dict[str, Any] = {
            "score_ceiling": self.score_ceiling,
            "success": bool(all(success_flags)),
            "joint_success": bool(all(success_flags)),
            "evaluation_delay_steps": [int(delay_steps) for delay_steps in sorted(evaluations.keys())],
            "active_evaluation_delays": self.active_evaluation_delays,
            "curriculum_enabled": self.curriculum_enabled,
            "curriculum_phase_1_delays": self.curriculum_phase_1_delays,
            "curriculum_phase_2_delays": self.curriculum_phase_2_delays,
            "curriculum_switch_generation": self.curriculum_switch_generation,
            "curriculum_phase": self.curriculum_phase,
            "score_current_phase": float(mean(scores)),
            "mean_score_over_delays": float(mean(scores)),
            "delay_score_std": float(pstdev(scores)) if len(scores) >= 2 else 0.0,
            "delay_score_range": float(max(scores) - min(scores)) if scores else 0.0,
            "sequence_mse": float(mean([evaluation.sequence_mse for evaluation in evaluations.values()])),
            "recall_mae": float(mean([evaluation.final_mae for evaluation in evaluations.values()])),
            "final_accuracy": float(mean([evaluation.final_accuracy for evaluation in evaluations.values()])),
            "per_delay_metrics": {
                str(delay_steps): {
                    "score": round(float(evaluation.score), 10),
                    "success": bool(evaluation.success),
                    "sequence_mse": round(float(evaluation.sequence_mse), 10),
                    "recall_mae": round(float(evaluation.final_mae), 10),
                    "final_accuracy": round(float(evaluation.final_accuracy), 10),
                }
                for delay_steps, evaluation in sorted(evaluations.items())
            },
            **_executor_metrics(aggregate_metrics),
        }
        for delay_steps, evaluation in evaluations.items():
            raw_metrics[f"score_delay_{delay_steps}"] = float(evaluation.score)
            raw_metrics[f"success_delay_{delay_steps}"] = bool(evaluation.success)

        return EvaluationResult(
            score=float(mean(scores)),
            raw_metrics=raw_metrics,
        )

    def _evaluate_task(
        self,
        genome: GenomeModel,
        task: BitMemoryTask,
    ) -> TemporalSequenceEvaluation:
        raw_outputs, episode_metrics = _run_samples(
            executor=self.executor,
            genome=genome,
            samples=task.input_sequences,
            sequence_mode=True,
        )
        predictions = _bounded_predictions(raw_outputs)
        final_predictions = predictions[:, -1:, :]
        final_targets = task.target_sequences[:, -1:, :]
        score = self.score_predictions(final_predictions, final_targets)
        sequence_mse = float(np.mean((predictions - task.target_sequences) ** 2))
        recall_mae = float(np.mean(np.abs(final_predictions - final_targets)))
        recall_classes = (final_predictions >= 0.5).astype(np.float32)
        recall_accuracy = float(np.mean(recall_classes == final_targets))

        return TemporalSequenceEvaluation(
            score=score,
            sequence_mse=sequence_mse,
            final_mae=recall_mae,
            final_accuracy=recall_accuracy,
            success=bool(np.all(recall_classes == final_targets)),
            raw_outputs=raw_outputs,
            predictions=predictions,
            final_predictions=final_predictions,
            final_targets=final_targets,
            episode_metrics=episode_metrics,
        )


class KeyValueMemoryEvaluator:
    task_name = "key_value_memory"

    def __init__(
        self,
        task: KeyValueMemoryTask,
        activation_steps: int,
        variant: str = "stateful",
        *,
        delta_w_clamp: float = 1.0,
        evaluation_delay_steps: Sequence[int] | None = None,
        curriculum_phase: str = "static",
        curriculum_enabled: bool = False,
        curriculum_phase_1_delays: str = "",
        curriculum_phase_2_delays: str = "",
        curriculum_switch_generation: int = 0,
        key_value_profile: str = "kv_full",
        curriculum_phase_1_key_value_profile: str = "",
        curriculum_phase_2_key_value_profile: str = "",
    ) -> None:
        self.task = task
        self.variant = variant
        self.evaluation_delay_steps = tuple(int(delay_steps) for delay_steps in (evaluation_delay_steps or (task.delay_steps,)))
        self.curriculum_phase = curriculum_phase
        self.curriculum_enabled = bool(curriculum_enabled)
        self.curriculum_phase_1_delays = curriculum_phase_1_delays
        self.curriculum_phase_2_delays = curriculum_phase_2_delays
        self.curriculum_switch_generation = int(curriculum_switch_generation)
        self.key_value_profile = str(key_value_profile or task.profile_name)
        self.curriculum_phase_1_key_value_profile = str(curriculum_phase_1_key_value_profile or "kv_easy")
        self.curriculum_phase_2_key_value_profile = str(curriculum_phase_2_key_value_profile or self.key_value_profile)
        self.active_evaluation_delays = ",".join(str(delay_steps) for delay_steps in self.evaluation_delay_steps)
        self.tasks_by_delay = {
            int(delay_steps): (
                task
                if int(delay_steps) == int(task.delay_steps) and task.profile_name == self.key_value_profile
                else KeyValueMemoryTask.create(delay_steps=int(delay_steps), profile=self.key_value_profile)
            )
            for delay_steps in self.evaluation_delay_steps
        }
        self.input_size = task.input_size
        self.output_size = task.output_size
        self.score_ceiling = float(task.query_target_values.shape[0] * task.num_queries)
        self.executor = _build_executor(
            variant=variant,
            activation_steps=activation_steps,
            delta_w_clamp=delta_w_clamp,
        )

    def evaluate(self, genome: GenomeModel) -> EvaluationResult:
        evaluations = {
            delay_steps: self._evaluate_task(genome, task)
            for delay_steps, task in self.tasks_by_delay.items()
        }
        if len(evaluations) == 1:
            delay_steps, evaluation = next(iter(evaluations.items()))
            return EvaluationResult(
                score=evaluation.score,
                raw_metrics={
                    "sequence_predictions": evaluation.predictions.round(6).tolist(),
                    "sequence_raw_outputs": evaluation.raw_outputs.round(6).tolist(),
                    "final_predictions": evaluation.final_predictions.flatten().round(6).tolist(),
                    "sequence_mse": evaluation.sequence_mse,
                    "query_mae": evaluation.final_mae,
                    "final_accuracy": evaluation.final_accuracy,
                    "score_ceiling": self.score_ceiling,
                    "success": evaluation.success,
                    "joint_success": evaluation.success,
                    "evaluation_delay_steps": [int(delay_steps)],
                    "active_evaluation_delays": self.active_evaluation_delays,
                    "curriculum_enabled": self.curriculum_enabled,
                    "curriculum_phase_1_delays": self.curriculum_phase_1_delays,
                    "curriculum_phase_2_delays": self.curriculum_phase_2_delays,
                    "curriculum_switch_generation": self.curriculum_switch_generation,
                    "curriculum_phase": self.curriculum_phase,
                    "score_current_phase": evaluation.score,
                    f"score_delay_{delay_steps}": evaluation.score,
                    f"success_delay_{delay_steps}": evaluation.success,
                    "mean_score_over_delays": evaluation.score,
                    "delay_score_std": 0.0,
                    "delay_score_range": 0.0,
                    **evaluation.task_metrics,
                    **_executor_metrics(evaluation.episode_metrics),
                },
            )

        scores = [evaluation.score for evaluation in evaluations.values()]
        success_flags = [evaluation.success for evaluation in evaluations.values()]
        aggregate_metrics = _aggregate_episode_metrics(
            [evaluation.episode_metrics for evaluation in evaluations.values() if evaluation.episode_metrics is not None]
        )
        raw_metrics: dict[str, Any] = {
            "score_ceiling": self.score_ceiling,
            "success": bool(all(success_flags)),
            "joint_success": bool(all(success_flags)),
            "evaluation_delay_steps": [int(delay_steps) for delay_steps in sorted(evaluations.keys())],
            "active_evaluation_delays": self.active_evaluation_delays,
            "curriculum_enabled": self.curriculum_enabled,
            "curriculum_phase_1_delays": self.curriculum_phase_1_delays,
            "curriculum_phase_2_delays": self.curriculum_phase_2_delays,
            "curriculum_switch_generation": self.curriculum_switch_generation,
            "curriculum_phase": self.curriculum_phase,
            "score_current_phase": float(mean(scores)),
            "mean_score_over_delays": float(mean(scores)),
            "delay_score_std": float(pstdev(scores)) if len(scores) >= 2 else 0.0,
            "delay_score_range": float(max(scores) - min(scores)) if scores else 0.0,
            "sequence_mse": float(mean([evaluation.sequence_mse for evaluation in evaluations.values()])),
            "query_mae": float(mean([evaluation.final_mae for evaluation in evaluations.values()])),
            "final_accuracy": float(mean([evaluation.final_accuracy for evaluation in evaluations.values()])),
            "query_accuracy": float(mean([float(evaluation.task_metrics.get("query_accuracy", 0.0)) for evaluation in evaluations.values()])),
            "retrieval_score": float(mean([float(evaluation.task_metrics.get("retrieval_score", 0.0)) for evaluation in evaluations.values()])),
            "exact_match_success": bool(all(bool(evaluation.task_metrics.get("exact_match_success", False)) for evaluation in evaluations.values())),
            "mean_query_distance": float(mean([float(evaluation.task_metrics.get("mean_query_distance", 0.0)) for evaluation in evaluations.values()])),
            "distractor_load": float(mean([float(evaluation.task_metrics.get("distractor_load", 0.0)) for evaluation in evaluations.values()])),
            "num_stores": float(mean([float(evaluation.task_metrics.get("num_stores", 0.0)) for evaluation in evaluations.values()])),
            "num_queries": float(mean([float(evaluation.task_metrics.get("num_queries", 0.0)) for evaluation in evaluations.values()])),
            "num_distractors": float(mean([float(evaluation.task_metrics.get("num_distractors", 0.0)) for evaluation in evaluations.values()])),
            "retrieval_margin": float(mean([float(evaluation.task_metrics.get("retrieval_margin", 0.0)) for evaluation in evaluations.values()])),
            "retrieval_confusion_rate": float(mean([float(evaluation.task_metrics.get("retrieval_confusion_rate", 0.0)) for evaluation in evaluations.values()])),
            "query_response_margin": float(mean([float(evaluation.task_metrics.get("query_response_margin", 0.0)) for evaluation in evaluations.values()])),
            "relevant_token_retention": float(mean([float(evaluation.task_metrics.get("relevant_token_retention", 0.0)) for evaluation in evaluations.values()])),
            "distractor_suppression_ratio": float(mean([float(evaluation.task_metrics.get("distractor_suppression_ratio", 0.0)) for evaluation in evaluations.values()])),
            "correct_key_selected": float(mean([float(evaluation.task_metrics.get("correct_key_selected", 0.0)) for evaluation in evaluations.values()])),
            "correct_value_selected": float(mean([float(evaluation.task_metrics.get("correct_value_selected", 0.0)) for evaluation in evaluations.values()])),
            "query_key_match_score": float(mean([float(evaluation.task_metrics.get("query_key_match_score", 0.0)) for evaluation in evaluations.values()])),
            "value_margin": float(mean([float(evaluation.task_metrics.get("value_margin", 0.0)) for evaluation in evaluations.values()])),
            "distractor_competition_score": float(mean([float(evaluation.task_metrics.get("distractor_competition_score", 0.0)) for evaluation in evaluations.values()])),
            "delta_retrieval_selection_pressure_bonus": float(mean([float(evaluation.task_metrics.get("delta_retrieval_selection_pressure_bonus", 0.0)) for evaluation in evaluations.values()])),
            "per_delay_metrics": {
                str(delay_steps): {
                    "score": round(float(evaluation.score), 10),
                    "success": bool(evaluation.success),
                    "sequence_mse": round(float(evaluation.sequence_mse), 10),
                    "query_mae": round(float(evaluation.final_mae), 10),
                    "query_accuracy": round(float(evaluation.task_metrics.get("query_accuracy", 0.0)), 10),
                    "retrieval_score": round(float(evaluation.task_metrics.get("retrieval_score", 0.0)), 10),
                    "mean_query_distance": round(float(evaluation.task_metrics.get("mean_query_distance", 0.0)), 10),
                    "distractor_load": round(float(evaluation.task_metrics.get("distractor_load", 0.0)), 10),
                    "distractor_suppression_ratio": round(float(evaluation.task_metrics.get("distractor_suppression_ratio", 0.0)), 10),
                    "correct_key_selected": round(float(evaluation.task_metrics.get("correct_key_selected", 0.0)), 10),
                    "correct_value_selected": round(float(evaluation.task_metrics.get("correct_value_selected", 0.0)), 10),
                    "query_key_match_score": round(float(evaluation.task_metrics.get("query_key_match_score", 0.0)), 10),
                    "value_margin": round(float(evaluation.task_metrics.get("value_margin", 0.0)), 10),
                    "distractor_competition_score": round(float(evaluation.task_metrics.get("distractor_competition_score", 0.0)), 10),
                    "delta_retrieval_selection_pressure_bonus": round(float(evaluation.task_metrics.get("delta_retrieval_selection_pressure_bonus", 0.0)), 10),
                }
                for delay_steps, evaluation in sorted(evaluations.items())
            },
            **_executor_metrics(aggregate_metrics),
        }
        for delay_steps, evaluation in evaluations.items():
            raw_metrics[f"score_delay_{delay_steps}"] = float(evaluation.score)
            raw_metrics[f"success_delay_{delay_steps}"] = bool(evaluation.success)

        return EvaluationResult(
            score=float(mean(scores)),
            raw_metrics=raw_metrics,
        )

    def _evaluate_task(
        self,
        genome: GenomeModel,
        task: KeyValueMemoryTask,
    ) -> TemporalSequenceEvaluation:
        raw_outputs, episode_metrics = _run_samples(
            executor=self.executor,
            genome=genome,
            samples=task.input_sequences,
            sequence_mode=True,
            step_roles_by_sample=task.step_roles,
        )
        predictions = _bounded_predictions(raw_outputs)
        final_predictions = predictions[:, -1:, :]
        final_targets = task.target_sequences[:, -1:, :]
        query_values = final_predictions[:, -1, 0]
        target_values = task.query_target_values.astype(np.float32)
        value_levels = np.asarray(task.value_levels, dtype=np.float32)
        predicted_ids = _nearest_value_indices(query_values, value_levels=value_levels)
        exact_matches = predicted_ids == task.query_target_ids
        exact_match_success = bool(np.all(exact_matches))

        exact_scores = exact_matches.astype(np.float32)
        soft_scores = 1.0 - np.abs(query_values - target_values)
        soft_scores = np.clip(soft_scores, 0.0, 1.0)
        score = float(np.sum(exact_scores))
        query_accuracy = float(np.mean(exact_scores))
        retrieval_score = float(np.mean(soft_scores))
        sequence_mse = float(np.mean((predictions - task.target_sequences) ** 2))
        query_mae = float(np.mean(np.abs(query_values - target_values)))
        retrieval_margin, query_response_margin = _query_margin_stats(
            query_values=query_values,
            target_ids=task.query_target_ids,
            value_levels=value_levels,
        )
        (
            correct_key_selected,
            correct_value_selected,
            query_key_match_score,
            value_margin,
            distractor_competition_score,
        ) = _query_retrieval_breakdown(
            query_values=query_values,
            predicted_ids=predicted_ids,
            target_ids=task.query_target_ids,
            query_store_indices=task.query_store_indices,
            store_value_ids=task.store_value_ids,
            value_levels=value_levels,
        )
        distractor_leakage = _mean_role_output(predictions, task.step_roles, role="distractor")
        distractor_suppression_ratio = retrieval_score / max(distractor_leakage, 1e-6)
        distractor_suppression_ratio = float(min(10.0, max(0.0, distractor_suppression_ratio)))

        task_metrics = {
            "query_accuracy": query_accuracy,
            "retrieval_score": retrieval_score,
            "exact_match_success": exact_match_success,
            "mean_query_distance": float(np.mean(task.query_distances)),
            "distractor_load": float(np.mean(task.distractor_loads)),
            "num_stores": int(task.num_stores),
            "num_queries": int(task.num_queries),
            "num_distractors": float(np.mean(task.distractor_counts)),
            "retrieval_margin": retrieval_margin,
            "retrieval_confusion_rate": float(1.0 - query_accuracy),
            "query_response_margin": query_response_margin,
            "relevant_token_retention": retrieval_score,
            "distractor_suppression_ratio": distractor_suppression_ratio,
            "correct_key_selected": correct_key_selected,
            "correct_value_selected": correct_value_selected,
            "query_key_match_score": query_key_match_score,
            "value_margin": value_margin,
            "distractor_competition_score": distractor_competition_score,
        }
        if is_stateful_v6_delta_memory_variant(self.variant):
            delta_bonus_inputs = {
                **task_metrics,
                "store_vs_distractor_beta_gap": float(getattr(episode_metrics, "store_vs_distractor_beta_gap", 0.0)) if episode_metrics is not None else 0.0,
                "key_query_cosine_mean": float(getattr(episode_metrics, "key_query_cosine_mean", 0.0)) if episode_metrics is not None else 0.0,
                "key_query_cosine_at_query": float(getattr(episode_metrics, "key_query_cosine_at_query", 0.0)) if episode_metrics is not None else 0.0,
                "key_variance_mean": float(getattr(episode_metrics, "key_variance_mean", 0.0)) if episode_metrics is not None else 0.0,
                "query_variance_mean": float(getattr(episode_metrics, "query_variance_mean", 0.0)) if episode_metrics is not None else 0.0,
                "readout_selectivity": float(getattr(episode_metrics, "readout_selectivity", 0.0)) if episode_metrics is not None else 0.0,
                "query_memory_alignment": float(getattr(episode_metrics, "query_memory_alignment", 0.0)) if episode_metrics is not None else 0.0,
            }
            delta_selection_bonus = _delta_retrieval_selection_pressure_bonus(delta_bonus_inputs)
            task_metrics["delta_retrieval_selection_pressure_bonus"] = delta_selection_bonus
            score += delta_selection_bonus
        else:
            task_metrics["delta_retrieval_selection_pressure_bonus"] = 0.0

        return TemporalSequenceEvaluation(
            score=score,
            sequence_mse=sequence_mse,
            final_mae=query_mae,
            final_accuracy=query_accuracy,
            success=exact_match_success,
            raw_outputs=raw_outputs,
            predictions=predictions,
            final_predictions=final_predictions,
            final_targets=final_targets,
            episode_metrics=episode_metrics,
            task_metrics=task_metrics,
        )


class EventMemoryEvaluator:
    task_name = "event_memory"

    def __init__(
        self,
        task: EventMemoryTask,
        activation_steps: int,
        variant: str = "stateful",
        *,
        delta_w_clamp: float = 1.0,
    ) -> None:
        self.task = task
        self.input_size = task.input_size
        self.output_size = task.output_size
        self.score_ceiling = float(task.target_sequences.shape[0] * task.target_sequences.shape[2])
        self.executor = _build_executor(
            variant=variant,
            activation_steps=activation_steps,
            delta_w_clamp=delta_w_clamp,
        )

    def score_predictions(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        absolute_error = float(np.abs(predictions - targets).sum())
        return _clamp_score(self.score_ceiling, absolute_error)

    def evaluate(self, genome: GenomeModel) -> EvaluationResult:
        raw_outputs, episode_metrics = _run_samples(
            executor=self.executor,
            genome=genome,
            samples=self.task.input_sequences,
            sequence_mode=True,
        )
        predictions = _bounded_predictions(raw_outputs)
        final_predictions = predictions[:, -1:, :]
        final_targets = self.task.target_sequences[:, -1:, :]
        score = self.score_predictions(final_predictions, final_targets)
        sequence_mse = float(np.mean((predictions - self.task.target_sequences) ** 2))
        decision_mae = float(np.mean(np.abs(final_predictions - final_targets)))
        decision_classes = (final_predictions >= 0.5).astype(np.float32)
        decision_accuracy = float(np.mean(decision_classes == final_targets))

        return EvaluationResult(
            score=score,
            raw_metrics={
                "sequence_predictions": predictions.round(6).tolist(),
                "sequence_raw_outputs": raw_outputs.round(6).tolist(),
                "final_predictions": final_predictions.flatten().round(6).tolist(),
                "sequence_mse": sequence_mse,
                "decision_mae": decision_mae,
                "final_accuracy": decision_accuracy,
                "score_ceiling": self.score_ceiling,
                "success": bool(np.all(decision_classes == final_targets)),
                **_executor_metrics(episode_metrics),
            },
        )


class EventDecisionEvaluator:
    task_name = "event_decision"

    def __init__(
        self,
        task: EventDecisionTask,
        activation_steps: int,
        variant: str = "stateful",
        *,
        delta_w_clamp: float = 1.0,
    ) -> None:
        self.task = task
        self.input_size = task.input_size
        self.output_size = task.output_size
        self.score_ceiling = float(task.target_sequences.shape[0] * task.target_sequences.shape[2])
        self.executor = _build_executor(
            variant=variant,
            activation_steps=activation_steps,
            delta_w_clamp=delta_w_clamp,
        )

    def score_predictions(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        absolute_error = float(np.abs(predictions - targets).sum())
        return _clamp_score(self.score_ceiling, absolute_error)

    def evaluate(self, genome: GenomeModel) -> EvaluationResult:
        raw_outputs, episode_metrics = _run_samples(
            executor=self.executor,
            genome=genome,
            samples=self.task.input_sequences,
            sequence_mode=True,
        )
        predictions = _bounded_predictions(raw_outputs)
        final_predictions = predictions[:, -1:, :]
        final_targets = self.task.target_sequences[:, -1:, :]
        score = self.score_predictions(final_predictions, final_targets)
        sequence_mse = float(np.mean((predictions - self.task.target_sequences) ** 2))
        decision_mae = float(np.mean(np.abs(final_predictions - final_targets)))
        decision_classes = (final_predictions >= 0.5).astype(np.float32)
        decision_accuracy = float(np.mean(decision_classes == final_targets))

        return EvaluationResult(
            score=score,
            raw_metrics={
                "sequence_predictions": predictions.round(6).tolist(),
                "sequence_raw_outputs": raw_outputs.round(6).tolist(),
                "final_predictions": final_predictions.flatten().round(6).tolist(),
                "sequence_mse": sequence_mse,
                "decision_mae": decision_mae,
                "final_accuracy": decision_accuracy,
                "score_ceiling": self.score_ceiling,
                "success": bool(np.all(decision_classes == final_targets)),
                **_executor_metrics(episode_metrics),
            },
        )


def build_evaluator(
    task_config: TaskConfig,
    variant: str = "stateful",
    *,
    delta_w_clamp: float = 1.0,
    generation_id: int | None = None,
) -> GenomeEvaluator:
    if task_config.name == "xor":
        return XorEvaluator(
            task=XorTask.create(),
            activation_steps=task_config.activation_steps,
            variant=variant,
            delta_w_clamp=delta_w_clamp,
        )
    if task_config.name == "delayed_xor":
        return DelayedXorEvaluator(
            task=DelayedXorTask.create(delay_steps=task_config.temporal_delay_steps),
            activation_steps=task_config.activation_steps,
            variant=variant,
            delta_w_clamp=delta_w_clamp,
        )
    if task_config.name == "bit_memory":
        evaluation_delay_steps = resolved_evaluation_delay_steps(task_config, generation_id=generation_id)
        curriculum_phase_1_delays, curriculum_phase_2_delays = curriculum_phase_delay_labels(task_config)
        return BitMemoryEvaluator(
            task=BitMemoryTask.create(delay_steps=task_config.temporal_delay_steps),
            activation_steps=task_config.activation_steps,
            variant=variant,
            delta_w_clamp=delta_w_clamp,
            evaluation_delay_steps=evaluation_delay_steps,
            curriculum_phase=curriculum_phase_name(task_config, generation_id=generation_id),
            curriculum_enabled=task_config.curriculum_enabled,
            curriculum_phase_1_delays=curriculum_phase_1_delays,
            curriculum_phase_2_delays=curriculum_phase_2_delays,
            curriculum_switch_generation=curriculum_switch_generation(task_config) if task_config.curriculum_enabled else 0,
        )
    if task_config.name == "key_value_memory":
        evaluation_delay_steps = resolved_evaluation_delay_steps(task_config, generation_id=generation_id)
        curriculum_phase_1_delays, curriculum_phase_2_delays = curriculum_phase_delay_labels(task_config)
        curriculum_phase_1_key_value_profile, curriculum_phase_2_key_value_profile = key_value_profile_labels(task_config)
        active_key_value_profile = resolved_key_value_profile(task_config, generation_id=generation_id)
        return KeyValueMemoryEvaluator(
            task=KeyValueMemoryTask.create(
                delay_steps=task_config.temporal_delay_steps,
                profile=resolved_key_value_profile(task_config),
            ),
            activation_steps=task_config.activation_steps,
            variant=variant,
            delta_w_clamp=delta_w_clamp,
            evaluation_delay_steps=evaluation_delay_steps,
            curriculum_phase=curriculum_phase_name(task_config, generation_id=generation_id),
            curriculum_enabled=task_config.curriculum_enabled,
            curriculum_phase_1_delays=curriculum_phase_1_delays,
            curriculum_phase_2_delays=curriculum_phase_2_delays,
            curriculum_switch_generation=curriculum_switch_generation(task_config) if task_config.curriculum_enabled else 0,
            key_value_profile=active_key_value_profile,
            curriculum_phase_1_key_value_profile=curriculum_phase_1_key_value_profile,
            curriculum_phase_2_key_value_profile=curriculum_phase_2_key_value_profile,
        )
    if task_config.name == "event_memory":
        return EventMemoryEvaluator(
            task=EventMemoryTask.create(delay_steps=task_config.temporal_delay_steps),
            activation_steps=task_config.activation_steps,
            variant=variant,
            delta_w_clamp=delta_w_clamp,
        )
    if task_config.name == "event_decision":
        return EventDecisionEvaluator(
            task=EventDecisionTask.create(delay_steps=task_config.temporal_delay_steps),
            activation_steps=task_config.activation_steps,
            variant=variant,
            delta_w_clamp=delta_w_clamp,
        )
    raise ValueError(f"Unsupported task: {task_config.name}")


def score_ceiling_for_task(task_config: TaskConfig) -> float:
    if task_config.name == "xor":
        return float(XorTask.create().targets.shape[0])
    if task_config.name == "delayed_xor":
        task = DelayedXorTask.create(delay_steps=task_config.temporal_delay_steps)
        return float(task.target_sequences.shape[0] * task.target_sequences.shape[2])
    if task_config.name == "bit_memory":
        task = BitMemoryTask.create(delay_steps=task_config.temporal_delay_steps)
        return float(task.target_sequences.shape[0] * task.target_sequences.shape[2])
    if task_config.name == "key_value_memory":
        task = KeyValueMemoryTask.create(
            delay_steps=task_config.temporal_delay_steps,
            profile=resolved_key_value_profile(task_config),
        )
        return float(task.query_target_values.shape[0] * task.num_queries)
    if task_config.name == "event_memory":
        task = EventMemoryTask.create(delay_steps=task_config.temporal_delay_steps)
        return float(task.target_sequences.shape[0] * task.target_sequences.shape[2])
    if task_config.name == "event_decision":
        task = EventDecisionTask.create(delay_steps=task_config.temporal_delay_steps)
        return float(task.target_sequences.shape[0] * task.target_sequences.shape[2])
    raise ValueError(f"Unsupported task: {task_config.name}")


def _bounded_predictions(raw_outputs: np.ndarray) -> np.ndarray:
    return np.clip((raw_outputs + 1.0) / 2.0, 0.0, 1.0)


def _clamp_score(score_ceiling: float, absolute_error: float) -> float:
    return min(score_ceiling, max(0.0, score_ceiling - absolute_error))


def _build_executor(
    variant: str,
    activation_steps: int,
    *,
    delta_w_clamp: float = 1.0,
) -> StatefulNetworkExecutor:
    if is_stateful_v2_variant(variant):
        return StatefulV2NetworkExecutor(activation_steps=activation_steps)
    if is_stateful_v2_gated_variant(variant):
        return StatefulV2GatedNetworkExecutor(activation_steps=activation_steps)
    if is_content_gated_variant(variant):
        return ContentGatedNetworkExecutor(activation_steps=activation_steps)
    if is_stateful_v3_kv_variant(variant):
        return StatefulV3KVNetworkExecutor(activation_steps=activation_steps)
    if is_stateful_v4_slots_variant(variant):
        return StatefulV4SlotsNetworkExecutor(activation_steps=activation_steps)
    if is_stateful_v5_addressed_slots_variant(variant):
        return StatefulV5AddressedSlotsNetworkExecutor(activation_steps=activation_steps)
    if is_stateful_v6_delta_memory_variant(variant):
        return StatefulV6DeltaMemoryNetworkExecutor(
            activation_steps=activation_steps,
            sub_variant=variant,
        )
    plastic_mode = plastic_mode_for_variant(variant)
    if plastic_mode == "hebb":
        return PlasticNetworkExecutor(
            activation_steps=activation_steps,
            delta_w_clamp=delta_w_clamp,
        )
    if plastic_mode == "ad":
        return AdaptivePlasticNetworkExecutor(
            activation_steps=activation_steps,
            delta_w_clamp=delta_w_clamp,
        )
    return StatefulNetworkExecutor(activation_steps=activation_steps)


def _executor_metrics(metrics: PlasticityEpisodeMetrics | None) -> dict[str, object]:
    if metrics is None:
        return {
            "plasticity_enabled": False,
            "mean_abs_delta_w": 0.0,
            "max_abs_delta_w": 0.0,
            "clamp_hit_rate": 0.0,
            "plasticity_active_fraction": 0.0,
            "mean_abs_decay_term": 0.0,
            "max_abs_decay_term": 0.0,
            "decay_effect_ratio": 0.0,
            "decay_near_zero_fraction": 0.0,
            "mean_abs_fast_state": 0.0,
            "mean_abs_slow_state": 0.0,
            "slow_fast_contribution_ratio": 0.0,
            "mean_abs_fast_state_during_store": 0.0,
            "mean_abs_slow_state_during_store": 0.0,
            "mean_abs_fast_state_during_query": 0.0,
            "mean_abs_slow_state_during_query": 0.0,
            "mean_abs_fast_state_during_distractor": 0.0,
            "mean_abs_slow_state_during_distractor": 0.0,
            "gate_mean": 0.0,
            "gate_variance": 0.0,
            "gate_at_store": 0.0,
            "gate_at_distractor": 0.0,
            "gate_at_query": 0.0,
            "gate_selectivity": 0.0,
            "gate_store_minus_query": 0.0,
            "gate_query_minus_distractor": 0.0,
            "gate_role_contrast": 0.0,
            "slow_state_at_query": 0.0,
            "fast_state_at_query": 0.0,
            "match_mean": 0.0,
            "match_variance": 0.0,
            "match_at_store": 0.0,
            "match_at_distractor": 0.0,
            "match_at_query": 0.0,
            "match_selectivity": 0.0,
            "query_match_score": 0.0,
            "state_query_alignment": 0.0,
            "content_retention_gap": 0.0,
            "mean_key_state": 0.0,
            "mean_value_state": 0.0,
            "key_value_separation": 0.0,
            "query_key_alignment": 0.0,
            "query_value_read_strength": 0.0,
            "store_key_value_coupling": 0.0,
            "distractor_write_leak": 0.0,
            "readout_selectivity": 0.0,
            "mean_key_state_during_store": 0.0,
            "mean_value_state_during_store": 0.0,
            "mean_key_state_during_query": 0.0,
            "mean_value_state_during_query": 0.0,
            "write_gate_at_store": 0.0,
            "write_gate_at_distractor": 0.0,
            "write_gate_at_query": 0.0,
            "store_vs_distractor_write_gap": 0.0,
            "mean_match_signal": 0.0,
            "value_state_at_query": 0.0,
            "key_state_at_query": 0.0,
            "slot_key_separation": 0.0,
            "slot_value_separation": 0.0,
            "slot_write_focus": 0.0,
            "slot_query_focus": 0.0,
            "slot_readout_selectivity": 0.0,
            "slot_utilization": 0.0,
            "query_slot_match_max": 0.0,
            "slot_distractor_leak": 0.0,
            "mean_write_address_focus": 0.0,
            "mean_read_address_focus": 0.0,
            "write_read_address_gap": 0.0,
            "slot_write_specialization": 0.0,
            "slot_read_specialization": 0.0,
            "address_consistency": 0.0,
            "query_read_alignment": 0.0,
            "store_write_alignment": 0.0,
            "readout_address_concentration": 0.0,
            "mean_beta_write": 0.0,
            "beta_at_store": 0.0,
            "beta_at_distractor": 0.0,
            "beta_at_query": 0.0,
            "store_vs_distractor_beta_gap": 0.0,
            "mean_key_norm": 0.0,
            "mean_query_norm": 0.0,
            "mean_value_norm": 0.0,
            "mean_memory_frobenius_norm": 0.0,
            "query_memory_alignment": 0.0,
            "store_memory_update_strength": 0.0,
            "delta_correction_magnitude": 0.0,
            "memory_read_strength": 0.0,
            "key_query_cosine_mean": 0.0,
            "key_query_cosine_at_query": 0.0,
            "key_variance_mean": 0.0,
            "query_variance_mean": 0.0,
            "key_query_projection_strength": 0.0,
            "query_decoupling_magnitude": 0.0,
        }
    return {
        "plasticity_enabled": metrics.plasticity_enabled,
        "mean_abs_delta_w": metrics.mean_abs_delta_w,
        "max_abs_delta_w": metrics.max_abs_delta_w,
        "clamp_hit_rate": metrics.clamp_hit_rate,
        "plasticity_active_fraction": metrics.plasticity_active_fraction,
        "mean_abs_decay_term": metrics.mean_abs_decay_term,
        "max_abs_decay_term": metrics.max_abs_decay_term,
        "decay_effect_ratio": metrics.decay_effect_ratio,
        "decay_near_zero_fraction": metrics.decay_near_zero_fraction,
        "mean_abs_fast_state": metrics.mean_abs_fast_state,
        "mean_abs_slow_state": metrics.mean_abs_slow_state,
        "slow_fast_contribution_ratio": metrics.slow_fast_contribution_ratio,
        "mean_abs_fast_state_during_store": metrics.mean_abs_fast_state_during_store,
        "mean_abs_slow_state_during_store": metrics.mean_abs_slow_state_during_store,
        "mean_abs_fast_state_during_query": metrics.mean_abs_fast_state_during_query,
        "mean_abs_slow_state_during_query": metrics.mean_abs_slow_state_during_query,
        "mean_abs_fast_state_during_distractor": metrics.mean_abs_fast_state_during_distractor,
        "mean_abs_slow_state_during_distractor": metrics.mean_abs_slow_state_during_distractor,
        "gate_mean": metrics.gate_mean,
        "gate_variance": metrics.gate_variance,
        "gate_at_store": metrics.gate_at_store,
        "gate_at_distractor": metrics.gate_at_distractor,
        "gate_at_query": metrics.gate_at_query,
        "gate_selectivity": metrics.gate_selectivity,
        "gate_store_minus_query": metrics.gate_store_minus_query,
        "gate_query_minus_distractor": metrics.gate_query_minus_distractor,
        "gate_role_contrast": metrics.gate_role_contrast,
        "slow_state_at_query": metrics.slow_state_at_query,
        "fast_state_at_query": metrics.fast_state_at_query,
        "match_mean": metrics.match_mean,
        "match_variance": metrics.match_variance,
        "match_at_store": metrics.match_at_store,
        "match_at_distractor": metrics.match_at_distractor,
        "match_at_query": metrics.match_at_query,
        "match_selectivity": metrics.match_selectivity,
        "query_match_score": metrics.query_match_score,
        "state_query_alignment": metrics.state_query_alignment,
        "content_retention_gap": metrics.content_retention_gap,
        "mean_key_state": metrics.mean_key_state,
        "mean_value_state": metrics.mean_value_state,
        "key_value_separation": metrics.key_value_separation,
        "query_key_alignment": metrics.query_key_alignment,
        "query_value_read_strength": metrics.query_value_read_strength,
        "store_key_value_coupling": metrics.store_key_value_coupling,
        "distractor_write_leak": metrics.distractor_write_leak,
        "readout_selectivity": metrics.readout_selectivity,
        "mean_key_state_during_store": metrics.mean_key_state_during_store,
        "mean_value_state_during_store": metrics.mean_value_state_during_store,
        "mean_key_state_during_query": metrics.mean_key_state_during_query,
        "mean_value_state_during_query": metrics.mean_value_state_during_query,
        "write_gate_at_store": metrics.write_gate_at_store,
        "write_gate_at_distractor": metrics.write_gate_at_distractor,
        "write_gate_at_query": metrics.write_gate_at_query,
        "store_vs_distractor_write_gap": metrics.store_vs_distractor_write_gap,
        "mean_match_signal": metrics.mean_match_signal,
        "value_state_at_query": metrics.value_state_at_query,
        "key_state_at_query": metrics.key_state_at_query,
        "slot_key_separation": metrics.slot_key_separation,
        "slot_value_separation": metrics.slot_value_separation,
        "slot_write_focus": metrics.slot_write_focus,
        "slot_query_focus": metrics.slot_query_focus,
        "slot_readout_selectivity": metrics.slot_readout_selectivity,
        "slot_utilization": metrics.slot_utilization,
        "query_slot_match_max": metrics.query_slot_match_max,
        "slot_distractor_leak": metrics.slot_distractor_leak,
        "mean_write_address_focus": metrics.mean_write_address_focus,
        "mean_read_address_focus": metrics.mean_read_address_focus,
        "write_read_address_gap": metrics.write_read_address_gap,
        "slot_write_specialization": metrics.slot_write_specialization,
        "slot_read_specialization": metrics.slot_read_specialization,
        "address_consistency": metrics.address_consistency,
        "query_read_alignment": metrics.query_read_alignment,
        "store_write_alignment": metrics.store_write_alignment,
        "readout_address_concentration": metrics.readout_address_concentration,
        "mean_beta_write": metrics.mean_beta_write,
        "beta_at_store": metrics.beta_at_store,
        "beta_at_distractor": metrics.beta_at_distractor,
        "beta_at_query": metrics.beta_at_query,
        "store_vs_distractor_beta_gap": metrics.store_vs_distractor_beta_gap,
        "mean_key_norm": metrics.mean_key_norm,
        "mean_query_norm": metrics.mean_query_norm,
        "mean_value_norm": metrics.mean_value_norm,
        "mean_memory_frobenius_norm": metrics.mean_memory_frobenius_norm,
        "query_memory_alignment": metrics.query_memory_alignment,
        "store_memory_update_strength": metrics.store_memory_update_strength,
        "delta_correction_magnitude": metrics.delta_correction_magnitude,
        "memory_read_strength": metrics.memory_read_strength,
        "key_query_cosine_mean": metrics.key_query_cosine_mean,
        "key_query_cosine_at_query": metrics.key_query_cosine_at_query,
        "key_variance_mean": metrics.key_variance_mean,
        "query_variance_mean": metrics.query_variance_mean,
        "key_query_projection_strength": metrics.key_query_projection_strength,
        "query_decoupling_magnitude": metrics.query_decoupling_magnitude,
    }


def _run_samples(
    *,
    executor: StatefulNetworkExecutor,
    genome: GenomeModel,
    samples: Sequence[Sequence[float] | Sequence[Sequence[float]]],
    sequence_mode: bool,
    step_roles_by_sample: Sequence[Sequence[str]] | None = None,
) -> tuple[np.ndarray, PlasticityEpisodeMetrics | None]:
    outputs: list[np.ndarray] = []
    metrics: list[PlasticityEpisodeMetrics] = []
    for sample_index, sample in enumerate(samples):
        if sequence_mode:
            roles = None
            if step_roles_by_sample is not None and sample_index < len(step_roles_by_sample):
                roles = step_roles_by_sample[sample_index]
            if roles is None:
                raw_output = executor.run_sequence(genome, sample)  # type: ignore[arg-type]
            else:
                raw_output = executor.run_sequence(genome, sample, step_roles=roles)  # type: ignore[arg-type]
        else:
            raw_output = executor.run(genome, sample)  # type: ignore[arg-type]
        outputs.append(raw_output)
        if hasattr(executor, "last_episode_metrics"):
            metrics.append(executor.last_episode_metrics())
    return np.stack(outputs), _aggregate_episode_metrics(metrics)


def _aggregate_episode_metrics(metrics: Sequence[PlasticityEpisodeMetrics]) -> PlasticityEpisodeMetrics | None:
    if not metrics:
        return None
    return PlasticityEpisodeMetrics(
        plasticity_enabled=any(metric.plasticity_enabled for metric in metrics),
        mean_abs_delta_w=float(np.mean([metric.mean_abs_delta_w for metric in metrics])),
        max_abs_delta_w=float(np.max([metric.max_abs_delta_w for metric in metrics])),
        clamp_hit_rate=float(np.mean([metric.clamp_hit_rate for metric in metrics])),
        plasticity_active_fraction=float(np.mean([metric.plasticity_active_fraction for metric in metrics])),
        mean_abs_decay_term=float(np.mean([metric.mean_abs_decay_term for metric in metrics])),
        max_abs_decay_term=float(np.max([metric.max_abs_decay_term for metric in metrics])),
        decay_effect_ratio=float(np.mean([metric.decay_effect_ratio for metric in metrics])),
        decay_near_zero_fraction=float(np.mean([metric.decay_near_zero_fraction for metric in metrics])),
        mean_abs_fast_state=float(np.mean([metric.mean_abs_fast_state for metric in metrics])),
        mean_abs_slow_state=float(np.mean([metric.mean_abs_slow_state for metric in metrics])),
        slow_fast_contribution_ratio=float(np.mean([metric.slow_fast_contribution_ratio for metric in metrics])),
        mean_abs_fast_state_during_store=float(np.mean([metric.mean_abs_fast_state_during_store for metric in metrics])),
        mean_abs_slow_state_during_store=float(np.mean([metric.mean_abs_slow_state_during_store for metric in metrics])),
        mean_abs_fast_state_during_query=float(np.mean([metric.mean_abs_fast_state_during_query for metric in metrics])),
        mean_abs_slow_state_during_query=float(np.mean([metric.mean_abs_slow_state_during_query for metric in metrics])),
        mean_abs_fast_state_during_distractor=float(np.mean([metric.mean_abs_fast_state_during_distractor for metric in metrics])),
        mean_abs_slow_state_during_distractor=float(np.mean([metric.mean_abs_slow_state_during_distractor for metric in metrics])),
        gate_mean=float(np.mean([metric.gate_mean for metric in metrics])),
        gate_variance=float(np.mean([metric.gate_variance for metric in metrics])),
        gate_at_store=float(np.mean([metric.gate_at_store for metric in metrics])),
        gate_at_distractor=float(np.mean([metric.gate_at_distractor for metric in metrics])),
        gate_at_query=float(np.mean([metric.gate_at_query for metric in metrics])),
        gate_selectivity=float(np.mean([metric.gate_selectivity for metric in metrics])),
        gate_store_minus_query=float(np.mean([metric.gate_store_minus_query for metric in metrics])),
        gate_query_minus_distractor=float(np.mean([metric.gate_query_minus_distractor for metric in metrics])),
        gate_role_contrast=float(np.mean([metric.gate_role_contrast for metric in metrics])),
        slow_state_at_query=float(np.mean([metric.slow_state_at_query for metric in metrics])),
        fast_state_at_query=float(np.mean([metric.fast_state_at_query for metric in metrics])),
        match_mean=float(np.mean([metric.match_mean for metric in metrics])),
        match_variance=float(np.mean([metric.match_variance for metric in metrics])),
        match_at_store=float(np.mean([metric.match_at_store for metric in metrics])),
        match_at_distractor=float(np.mean([metric.match_at_distractor for metric in metrics])),
        match_at_query=float(np.mean([metric.match_at_query for metric in metrics])),
        match_selectivity=float(np.mean([metric.match_selectivity for metric in metrics])),
        query_match_score=float(np.mean([metric.query_match_score for metric in metrics])),
        state_query_alignment=float(np.mean([metric.state_query_alignment for metric in metrics])),
        content_retention_gap=float(np.mean([metric.content_retention_gap for metric in metrics])),
        mean_key_state=float(np.mean([metric.mean_key_state for metric in metrics])),
        mean_value_state=float(np.mean([metric.mean_value_state for metric in metrics])),
        key_value_separation=float(np.mean([metric.key_value_separation for metric in metrics])),
        query_key_alignment=float(np.mean([metric.query_key_alignment for metric in metrics])),
        query_value_read_strength=float(np.mean([metric.query_value_read_strength for metric in metrics])),
        store_key_value_coupling=float(np.mean([metric.store_key_value_coupling for metric in metrics])),
        distractor_write_leak=float(np.mean([metric.distractor_write_leak for metric in metrics])),
        readout_selectivity=float(np.mean([metric.readout_selectivity for metric in metrics])),
        mean_key_state_during_store=float(np.mean([metric.mean_key_state_during_store for metric in metrics])),
        mean_value_state_during_store=float(np.mean([metric.mean_value_state_during_store for metric in metrics])),
        mean_key_state_during_query=float(np.mean([metric.mean_key_state_during_query for metric in metrics])),
        mean_value_state_during_query=float(np.mean([metric.mean_value_state_during_query for metric in metrics])),
        write_gate_at_store=float(np.mean([metric.write_gate_at_store for metric in metrics])),
        write_gate_at_distractor=float(np.mean([metric.write_gate_at_distractor for metric in metrics])),
        write_gate_at_query=float(np.mean([metric.write_gate_at_query for metric in metrics])),
        store_vs_distractor_write_gap=float(np.mean([metric.store_vs_distractor_write_gap for metric in metrics])),
        mean_match_signal=float(np.mean([metric.mean_match_signal for metric in metrics])),
        value_state_at_query=float(np.mean([metric.value_state_at_query for metric in metrics])),
        key_state_at_query=float(np.mean([metric.key_state_at_query for metric in metrics])),
        slot_key_separation=float(np.mean([metric.slot_key_separation for metric in metrics])),
        slot_value_separation=float(np.mean([metric.slot_value_separation for metric in metrics])),
        slot_write_focus=float(np.mean([metric.slot_write_focus for metric in metrics])),
        slot_query_focus=float(np.mean([metric.slot_query_focus for metric in metrics])),
        slot_readout_selectivity=float(np.mean([metric.slot_readout_selectivity for metric in metrics])),
        slot_utilization=float(np.mean([metric.slot_utilization for metric in metrics])),
        query_slot_match_max=float(np.mean([metric.query_slot_match_max for metric in metrics])),
        slot_distractor_leak=float(np.mean([metric.slot_distractor_leak for metric in metrics])),
        mean_write_address_focus=float(np.mean([metric.mean_write_address_focus for metric in metrics])),
        mean_read_address_focus=float(np.mean([metric.mean_read_address_focus for metric in metrics])),
        write_read_address_gap=float(np.mean([metric.write_read_address_gap for metric in metrics])),
        slot_write_specialization=float(np.mean([metric.slot_write_specialization for metric in metrics])),
        slot_read_specialization=float(np.mean([metric.slot_read_specialization for metric in metrics])),
        address_consistency=float(np.mean([metric.address_consistency for metric in metrics])),
        query_read_alignment=float(np.mean([metric.query_read_alignment for metric in metrics])),
        store_write_alignment=float(np.mean([metric.store_write_alignment for metric in metrics])),
        readout_address_concentration=float(np.mean([metric.readout_address_concentration for metric in metrics])),
        mean_beta_write=float(np.mean([metric.mean_beta_write for metric in metrics])),
        beta_at_store=float(np.mean([metric.beta_at_store for metric in metrics])),
        beta_at_distractor=float(np.mean([metric.beta_at_distractor for metric in metrics])),
        beta_at_query=float(np.mean([metric.beta_at_query for metric in metrics])),
        store_vs_distractor_beta_gap=float(np.mean([metric.store_vs_distractor_beta_gap for metric in metrics])),
        mean_key_norm=float(np.mean([metric.mean_key_norm for metric in metrics])),
        mean_query_norm=float(np.mean([metric.mean_query_norm for metric in metrics])),
        mean_value_norm=float(np.mean([metric.mean_value_norm for metric in metrics])),
        mean_memory_frobenius_norm=float(np.mean([metric.mean_memory_frobenius_norm for metric in metrics])),
        query_memory_alignment=float(np.mean([metric.query_memory_alignment for metric in metrics])),
        store_memory_update_strength=float(np.mean([metric.store_memory_update_strength for metric in metrics])),
        delta_correction_magnitude=float(np.mean([metric.delta_correction_magnitude for metric in metrics])),
        memory_read_strength=float(np.mean([metric.memory_read_strength for metric in metrics])),
        key_query_cosine_mean=float(np.mean([metric.key_query_cosine_mean for metric in metrics])),
        key_query_cosine_at_query=float(np.mean([metric.key_query_cosine_at_query for metric in metrics])),
        key_variance_mean=float(np.mean([metric.key_variance_mean for metric in metrics])),
        query_variance_mean=float(np.mean([metric.query_variance_mean for metric in metrics])),
        key_query_projection_strength=float(np.mean([metric.key_query_projection_strength for metric in metrics])),
        query_decoupling_magnitude=float(np.mean([metric.query_decoupling_magnitude for metric in metrics])),
    )


def _nearest_value_indices(predictions: np.ndarray, *, value_levels: np.ndarray) -> np.ndarray:
    distances = np.abs(predictions[:, None] - value_levels[None, :])
    return np.argmin(distances, axis=1).astype(np.int32)


def _query_margin_stats(
    *,
    query_values: np.ndarray,
    target_ids: np.ndarray,
    value_levels: np.ndarray,
) -> tuple[float, float]:
    margins: list[float] = []
    response_margins: list[float] = []
    for predicted_value, target_id in zip(query_values.tolist(), target_ids.tolist(), strict=True):
        distances = np.abs(value_levels - float(predicted_value))
        sorted_indices = np.argsort(distances)
        closest_distance = float(distances[sorted_indices[0]])
        runner_up_distance = float(distances[sorted_indices[1]]) if len(sorted_indices) > 1 else closest_distance
        target_distance = float(distances[int(target_id)])
        competitor_distance = min(
            float(distances[index])
            for index in range(len(value_levels))
            if index != int(target_id)
        )
        margins.append(competitor_distance - target_distance)
        response_margins.append(runner_up_distance - closest_distance)
    return float(mean(margins)) if margins else 0.0, float(mean(response_margins)) if response_margins else 0.0


def _query_retrieval_breakdown(
    *,
    query_values: np.ndarray,
    predicted_ids: np.ndarray,
    target_ids: np.ndarray,
    query_store_indices: np.ndarray,
    store_value_ids: np.ndarray,
    value_levels: np.ndarray,
) -> tuple[float, float, float, float, float]:
    key_selected_flags: list[float] = []
    value_selected_flags: list[float] = []
    key_match_scores: list[float] = []
    value_margins: list[float] = []
    distractor_competition_scores: list[float] = []

    for sample_index, predicted_value in enumerate(query_values.tolist()):
        target_id = int(target_ids[sample_index])
        target_store_index = int(query_store_indices[sample_index])
        sample_store_value_ids = [int(value_id) for value_id in store_value_ids[sample_index].tolist()]
        target_distance = abs(float(predicted_value) - float(value_levels[target_id]))

        competitor_value_ids = [
            value_id
            for store_index, value_id in enumerate(sample_store_value_ids)
            if store_index != target_store_index
        ]
        competitor_distances = [
            abs(float(predicted_value) - float(value_levels[value_id]))
            for value_id in competitor_value_ids
        ]
        best_competitor_distance = min(competitor_distances) if competitor_distances else 1.0
        key_match_score = float(best_competitor_distance - target_distance)
        key_selected_flags.append(1.0 if key_match_score >= 0.0 else 0.0)
        value_selected_flags.append(1.0 if int(predicted_ids[sample_index]) == target_id else 0.0)
        key_match_scores.append(key_match_score)

        all_value_distances = np.abs(value_levels - float(predicted_value))
        non_target_distances = [
            float(distance)
            for value_index, distance in enumerate(all_value_distances.tolist())
            if value_index != target_id
        ]
        runner_up_distance = min(non_target_distances) if non_target_distances else target_distance
        value_margins.append(float(runner_up_distance - target_distance))
        distractor_competition_scores.append(float(max(0.0, 1.0 - best_competitor_distance)))

    return (
        float(mean(key_selected_flags)) if key_selected_flags else 0.0,
        float(mean(value_selected_flags)) if value_selected_flags else 0.0,
        float(mean(key_match_scores)) if key_match_scores else 0.0,
        float(mean(value_margins)) if value_margins else 0.0,
        float(mean(distractor_competition_scores)) if distractor_competition_scores else 0.0,
    )


def _delta_retrieval_selection_pressure_bonus(metrics: Mapping[str, float]) -> float:
    correct_key_selected = float(metrics.get("correct_key_selected", 0.0) or 0.0)
    query_key_match_score = float(metrics.get("query_key_match_score", 0.0) or 0.0)
    store_vs_distractor_beta_gap = float(metrics.get("store_vs_distractor_beta_gap", 0.0) or 0.0)
    correct_value_selected = float(metrics.get("correct_value_selected", 0.0) or 0.0)
    value_margin = float(metrics.get("value_margin", 0.0) or 0.0)

    key_query_cosine_mean = float(metrics.get("key_query_cosine_mean", 0.0) or 0.0)
    key_query_cosine_at_query = float(metrics.get("key_query_cosine_at_query", 0.0) or 0.0)
    key_variance_mean = float(metrics.get("key_variance_mean", 0.0) or 0.0)
    query_variance_mean = float(metrics.get("query_variance_mean", 0.0) or 0.0)
    readout_selectivity = float(metrics.get("readout_selectivity", 0.0) or 0.0)
    query_memory_alignment = float(metrics.get("query_memory_alignment", 0.0) or 0.0)

    # NOTE: ``correct_key_selected`` is *not* a reliable key-selection signal.
    # It is derived from value-distance ranking against in-sample competitor
    # store values in ``_query_retrieval_breakdown``: it goes to 1.0 whenever
    # the prediction is at least as close to the target value as to any
    # competitor's value, which is trivially true when two store slots happen
    # to share a value level (common for kv_easy/kv_full) and degenerate for
    # kv_trivial (no competitors at all). It must therefore not be the
    # dominant term of this selection bonus. Real fitness pressure should
    # come from ``correct_value_selected`` (exact value-id match) and the
    # margin metric ``query_key_match_score``.
    signed_query_match = float(np.tanh(query_key_match_score))
    signed_value_margin = float(np.tanh(value_margin))
    positive_beta_gap = float(np.tanh(max(store_vs_distractor_beta_gap, 0.0)))
    readout_peakiness = float(np.tanh(readout_selectivity * 4.0))

    bonus = 0.0
    bonus += 0.45 * correct_value_selected
    bonus += 0.40 * signed_query_match
    bonus += 0.25 * signed_value_margin
    bonus += 0.20 * readout_peakiness
    bonus += 0.30 * positive_beta_gap
    bonus += 0.02 * correct_key_selected

    bonus -= 0.04 * max(0.0, key_query_cosine_mean - 0.50)
    bonus -= 0.04 * max(0.0, key_query_cosine_at_query - 0.50)
    bonus -= 0.02 * max(0.0, 0.02 - key_variance_mean)
    bonus -= 0.02 * max(0.0, 0.02 - query_variance_mean)
    bonus -= 0.06 * max(0.0, 0.10 - readout_selectivity)
    bonus -= 0.08 * max(0.0, 0.05 - store_vs_distractor_beta_gap)
    bonus -= 0.08 * max(0.0, 0.20 - query_memory_alignment)

    return float(np.clip(bonus, -0.15, 0.85))


def _mean_role_output(
    predictions: np.ndarray,
    step_roles: Sequence[Sequence[str]],
    *,
    role: str,
) -> float:
    values: list[float] = []
    for sample_index, roles in enumerate(step_roles):
        for step_index, step_role in enumerate(roles):
            if str(step_role).strip().lower() != role:
                continue
            values.append(abs(float(predictions[sample_index, step_index, 0])))
    return float(mean(values)) if values else 0.0
