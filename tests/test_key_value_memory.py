from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import TaskConfig
from evolve.custom_neuron import PlasticityEpisodeMetrics
from evolve.evaluator import (
    KeyValueMemoryEvaluator,
    _delta_retrieval_selection_pressure_bonus,
    build_evaluator,
)
from tasks.key_value_memory import KeyValueMemoryTask


def test_key_value_memory_task_is_deterministic_and_targets_align() -> None:
    first = KeyValueMemoryTask.create(delay_steps=3)
    second = KeyValueMemoryTask.create(delay_steps=3)

    assert np.array_equal(first.input_sequences, second.input_sequences)
    assert np.array_equal(first.target_sequences, second.target_sequences)
    assert np.array_equal(first.query_target_ids, second.query_target_ids)
    assert np.array_equal(first.query_target_values, second.query_target_values)
    assert all(roles[-1] == "query" for roles in first.step_roles)
    assert all("separator" in roles for roles in first.step_roles)
    assert np.allclose(first.target_sequences[:, -1, 0], first.query_target_values)


def test_key_value_memory_delay_increases_context_pressure() -> None:
    easy = KeyValueMemoryTask.create(delay_steps=3, profile="kv_full")
    hard = KeyValueMemoryTask.create(delay_steps=8, profile="kv_full")

    assert easy.num_stores < hard.num_stores
    assert float(np.mean(easy.query_distances)) < float(np.mean(hard.query_distances))
    assert float(np.mean(easy.distractor_loads)) < float(np.mean(hard.distractor_loads))


def test_key_value_memory_profiles_scale_retrieval_difficulty() -> None:
    trivial = KeyValueMemoryTask.create(delay_steps=8, profile="kv_trivial")
    easy = KeyValueMemoryTask.create(delay_steps=8, profile="kv_easy")
    mid = KeyValueMemoryTask.create(delay_steps=8, profile="kv_mid")
    full = KeyValueMemoryTask.create(delay_steps=8, profile="kv_full")

    assert trivial.profile_name == "kv_trivial"
    assert easy.profile_name == "kv_easy"
    assert mid.profile_name == "kv_mid"
    assert full.profile_name == "kv_full"
    assert trivial.num_stores == 1
    assert float(np.mean(trivial.distractor_loads)) == 0.0
    assert len(easy.value_levels) < len(full.value_levels)
    assert easy.num_stores <= mid.num_stores <= full.num_stores
    assert float(np.mean(easy.query_distances)) <= float(np.mean(mid.query_distances))
    assert float(np.mean(mid.query_distances)) <= float(np.mean(full.query_distances))
    assert float(np.mean(easy.distractor_loads)) <= float(np.mean(mid.distractor_loads))
    assert float(np.mean(mid.distractor_loads)) <= float(np.mean(full.distractor_loads))


def test_key_value_memory_evaluator_reports_retrieval_metrics() -> None:
    task = KeyValueMemoryTask.create(delay_steps=3)
    evaluator = KeyValueMemoryEvaluator(task=task, activation_steps=1, key_value_profile=task.profile_name)

    class DummyExecutor:
        def __init__(self, outputs: np.ndarray) -> None:
            self._outputs = outputs
            self._index = 0

        def run_sequence(self, genome: object, sample: np.ndarray, *, step_roles: tuple[str, ...] | None = None) -> np.ndarray:
            del genome, sample, step_roles
            output = self._outputs[self._index % self._outputs.shape[0]]
            self._index += 1
            return output

        def last_episode_metrics(self) -> PlasticityEpisodeMetrics:
            return PlasticityEpisodeMetrics(
                plasticity_enabled=False,
                mean_abs_delta_w=0.0,
                max_abs_delta_w=0.0,
                clamp_hit_rate=0.0,
                plasticity_active_fraction=0.0,
                mean_abs_decay_term=0.0,
                max_abs_decay_term=0.0,
                decay_effect_ratio=0.0,
                decay_near_zero_fraction=0.0,
                mean_abs_fast_state=1.0,
                mean_abs_slow_state=2.0,
                slow_fast_contribution_ratio=2.0,
                mean_abs_fast_state_during_store=0.8,
                mean_abs_slow_state_during_store=1.6,
                mean_abs_fast_state_during_query=0.7,
                mean_abs_slow_state_during_query=1.4,
                mean_abs_fast_state_during_distractor=0.2,
                mean_abs_slow_state_during_distractor=0.3,
            )

    raw_outputs = np.asarray([(targets * 2.0) - 1.0 for targets in task.target_sequences], dtype=np.float32)
    evaluator.executor = DummyExecutor(raw_outputs)  # type: ignore[assignment]

    result = evaluator.evaluate(genome=object())  # type: ignore[arg-type]

    assert result.score == evaluator.score_ceiling
    assert result.raw_metrics["success"] is True
    assert result.raw_metrics["query_accuracy"] == 1.0
    assert result.raw_metrics["retrieval_score"] == 1.0
    assert result.raw_metrics["exact_match_success"] is True
    assert result.raw_metrics["mean_query_distance"] > 0.0
    assert result.raw_metrics["distractor_load"] > 0.0
    assert result.raw_metrics["correct_key_selected"] == 1.0
    assert result.raw_metrics["correct_value_selected"] == 1.0
    assert result.raw_metrics["query_key_match_score"] > 0.0
    assert result.raw_metrics["value_margin"] >= 0.0
    assert result.raw_metrics["distractor_competition_score"] >= 0.0
    assert result.raw_metrics["distractor_suppression_ratio"] >= 1.0
    assert result.raw_metrics["mean_abs_slow_state_during_query"] > 0.0


def test_key_value_memory_soft_base_scoring_preserves_exact_success_and_ranks_near_misses() -> None:
    task = KeyValueMemoryTask.create(delay_steps=8, profile="kv_easy")
    evaluator = KeyValueMemoryEvaluator(task=task, activation_steps=1, key_value_profile=task.profile_name)

    class DummyExecutor:
        def __init__(self, outputs: np.ndarray) -> None:
            self._outputs = outputs
            self._index = 0

        def run_sequence(self, genome: object, sample: np.ndarray, *, step_roles: tuple[str, ...] | None = None) -> np.ndarray:
            del genome, sample, step_roles
            output = self._outputs[self._index % self._outputs.shape[0]]
            self._index += 1
            return output

        def last_episode_metrics(self) -> PlasticityEpisodeMetrics:
            return PlasticityEpisodeMetrics(
                plasticity_enabled=False,
                mean_abs_delta_w=0.0,
                max_abs_delta_w=0.0,
                clamp_hit_rate=0.0,
                plasticity_active_fraction=0.0,
                mean_abs_decay_term=0.0,
                max_abs_decay_term=0.0,
                decay_effect_ratio=0.0,
                decay_near_zero_fraction=0.0,
                mean_abs_fast_state=0.0,
                mean_abs_slow_state=0.0,
                slow_fast_contribution_ratio=0.0,
                mean_abs_fast_state_during_store=0.0,
                mean_abs_slow_state_during_store=0.0,
                mean_abs_fast_state_during_query=0.0,
                mean_abs_slow_state_during_query=0.0,
                mean_abs_fast_state_during_distractor=0.0,
                mean_abs_slow_state_during_distractor=0.0,
            )

    def build_outputs(query_values: np.ndarray) -> np.ndarray:
        raw_outputs = np.zeros_like(task.target_sequences, dtype=np.float32)
        raw_outputs[:, -1, 0] = (query_values * 2.0) - 1.0
        return raw_outputs

    target_ids = task.query_target_ids.astype(np.int64)
    value_levels = np.asarray(task.value_levels, dtype=np.float32)
    target_values = value_levels[target_ids]
    wrong_ids = (target_ids + 1) % value_levels.shape[0]
    wrong_values = value_levels[wrong_ids]

    near_values = target_values.copy()
    worse_values = target_values.copy()
    half = target_values.shape[0] // 2
    near_values[half:] = wrong_values[half:] + (0.49 * (target_values[half:] - wrong_values[half:]))
    worse_values[half:] = wrong_values[half:]

    perfect_outputs = build_outputs(target_values)
    near_outputs = build_outputs(near_values)
    worse_outputs = build_outputs(worse_values)

    evaluator.executor = DummyExecutor(perfect_outputs)  # type: ignore[assignment]
    perfect = evaluator.evaluate(genome=object())  # type: ignore[arg-type]
    evaluator.executor = DummyExecutor(near_outputs)  # type: ignore[assignment]
    near = evaluator.evaluate(genome=object())  # type: ignore[arg-type]
    evaluator.executor = DummyExecutor(worse_outputs)  # type: ignore[assignment]
    worse = evaluator.evaluate(genome=object())  # type: ignore[arg-type]

    assert perfect.score == evaluator.score_ceiling
    assert perfect.raw_metrics["success"] is True
    assert near.raw_metrics["success"] is False
    assert worse.raw_metrics["success"] is False
    assert near.raw_metrics["query_accuracy"] == worse.raw_metrics["query_accuracy"]
    assert near.score > worse.score


def test_delta_retrieval_bonus_does_not_reward_correct_key_selected_dominantly() -> None:
    # A candidate that only inflates the unreliable ``correct_key_selected``
    # signal must not beat a candidate with stronger ``correct_value_selected``
    # / ``query_key_match_score`` even if its raw key-match flag is maximal.
    fake_key_only = {
        "correct_key_selected": 1.0,
        "correct_value_selected": 0.0,
        "query_key_match_score": 0.0,
        "store_vs_distractor_beta_gap": 0.0,
        "key_query_cosine_mean": 0.0,
        "key_query_cosine_at_query": 0.0,
        "key_variance_mean": 0.05,
        "query_variance_mean": 0.05,
    }
    real_value_match = {
        "correct_key_selected": 0.0,
        "correct_value_selected": 1.0,
        "query_key_match_score": 0.5,
        "store_vs_distractor_beta_gap": 0.0,
        "key_query_cosine_mean": 0.0,
        "key_query_cosine_at_query": 0.0,
        "key_variance_mean": 0.05,
        "query_variance_mean": 0.05,
    }

    bonus_key_only = _delta_retrieval_selection_pressure_bonus(fake_key_only)
    bonus_value_match = _delta_retrieval_selection_pressure_bonus(real_value_match)

    # ``correct_key_selected`` may still contribute a small auxiliary boost,
    # but it must not be the dominant component of the bonus.
    assert bonus_key_only < 0.10
    assert bonus_value_match > bonus_key_only
    assert bonus_value_match > 3.0 * bonus_key_only


def test_key_value_memory_curriculum_uses_phase_specific_delays() -> None:
    config = replace(
        TaskConfig(),
        name="key_value_memory",
        activation_steps=4,
        temporal_delay_steps=8,
        curriculum_enabled=True,
        curriculum_phase_switch_generation=6,
        curriculum_phase_1_delay_steps=(3,),
        curriculum_phase_2_delay_steps=(8,),
        curriculum_phase_1_key_value_profile="kv_easy",
        curriculum_phase_2_key_value_profile="kv_full",
    )

    phase_1 = build_evaluator(config, variant="stateful_v2", generation_id=0)
    phase_2 = build_evaluator(config, variant="stateful_v2", generation_id=6)

    assert phase_1.task_name == "key_value_memory"
    assert phase_1.evaluation_delay_steps == (3,)
    assert phase_2.evaluation_delay_steps == (8,)
    assert phase_1.curriculum_phase == "phase_1"
    assert phase_2.curriculum_phase == "phase_2"
    assert phase_1.key_value_profile == "kv_easy"
    assert phase_2.key_value_profile == "kv_full"
