from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import AppConfig, RunConfig, TaskConfig, curriculum_switch_generation
from evolve.custom_neuron import AdaptivePlasticNetworkExecutor, PlasticNetworkExecutor
from evolve.evaluator import DelayedXorEvaluator, build_evaluator
from evolve.genome_codec import arrays_to_genome_model
from evolve.tensorneat_adapter import TensorNEATAdapter
from tasks.bit_memory import BitMemoryTask
from tasks.delayed_xor import DelayedXorTask


def test_delayed_xor_task_has_expected_shape_and_targets() -> None:
    task = DelayedXorTask.create(delay_steps=2)

    assert task.input_sequences.shape == (4, 5, 3)
    assert task.target_sequences.shape == (4, 5, 1)
    assert task.target_sequences[:, -1, 0].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_bit_memory_task_has_expected_shape_and_recall_targets() -> None:
    task = BitMemoryTask.create(delay_steps=3)

    assert task.input_sequences.shape == (4, 5, 3)
    assert task.target_sequences.shape == (4, 5, 1)
    assert task.target_sequences[:, -1, 0].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_delayed_xor_evaluator_is_deterministic_for_same_genome() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="delayed_xor", activation_steps=4, temporal_delay_steps=1),
        run=replace(RunConfig(), seed=11),
    )
    evaluator = build_evaluator(config.task)
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])

    first = evaluator.evaluate(genome)
    second = evaluator.evaluate(genome)

    assert first.score == second.score
    assert first.raw_metrics == second.raw_metrics


def test_delayed_xor_scores_final_cue_only() -> None:
    task = DelayedXorTask.create(delay_steps=1)
    evaluator = DelayedXorEvaluator(task=task, activation_steps=1)

    # Intermediate outputs are intentionally wrong; only the final recall is correct.
    wrong_intermediate = np.array(
        [
            [[1.0], [1.0], [1.0], [-1.0]],
            [[1.0], [1.0], [1.0], [1.0]],
            [[1.0], [1.0], [1.0], [1.0]],
            [[1.0], [1.0], [1.0], [-1.0]],
        ],
        dtype=np.float32,
    )

    class DummyExecutor:
        def __init__(self, outputs: np.ndarray) -> None:
            self._outputs = iter(outputs)

        def run_sequence(self, genome: object, sample: np.ndarray) -> np.ndarray:
            return next(self._outputs)

    evaluator.executor = DummyExecutor(wrong_intermediate)  # type: ignore[assignment]

    result = evaluator.evaluate(genome=object())  # type: ignore[arg-type]

    assert evaluator.score_ceiling == 4.0
    assert result.score == evaluator.score_ceiling
    assert result.raw_metrics["success"] is True
    assert result.raw_metrics["sequence_mse"] > 0.0
    assert result.raw_metrics["final_accuracy"] == 1.0


def test_bit_memory_score_is_clamped_to_theoretical_maximum() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=1),
        run=replace(RunConfig(), seed=11),
    )
    evaluator = build_evaluator(config.task)
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])

    result = evaluator.evaluate(genome)

    assert result.score <= evaluator.score_ceiling
    assert result.raw_metrics["score_ceiling"] == evaluator.score_ceiling
    assert result.raw_metrics["score_delay_1"] == result.score
    assert result.raw_metrics["mean_score_over_delays"] == result.score
    assert result.raw_metrics["delay_score_std"] == 0.0
    assert result.raw_metrics["delay_score_range"] == 0.0


def test_bit_memory_multi_delay_evaluator_reports_delay_breakdown() -> None:
    config = AppConfig(
        task=replace(
            TaskConfig(),
            name="bit_memory",
            activation_steps=4,
            temporal_delay_steps=8,
            evaluation_delay_steps=(5, 8),
        ),
        run=replace(RunConfig(), seed=11, variant="stateful_v2"),
    )
    evaluator = build_evaluator(config.task, variant="stateful_v2")
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])

    result = evaluator.evaluate(genome)

    assert result.raw_metrics["evaluation_delay_steps"] == [5, 8]
    assert "score_delay_5" in result.raw_metrics
    assert "score_delay_8" in result.raw_metrics
    assert "success_delay_5" in result.raw_metrics
    assert "success_delay_8" in result.raw_metrics
    assert result.raw_metrics["mean_score_over_delays"] == result.score
    assert result.raw_metrics["delay_score_std"] >= 0.0
    assert result.raw_metrics["delay_score_range"] >= 0.0
    assert result.raw_metrics["joint_success"] == (
        bool(result.raw_metrics["success_delay_5"]) and bool(result.raw_metrics["success_delay_8"])
    )


def test_bit_memory_curriculum_switches_active_delay_set_by_generation() -> None:
    config = AppConfig(
        task=replace(
            TaskConfig(),
            name="bit_memory",
            activation_steps=4,
            temporal_delay_steps=8,
            curriculum_enabled=True,
            curriculum_phase_switch_generation=6,
            curriculum_phase_1_delay_steps=(5,),
            curriculum_phase_2_delay_steps=(5, 8),
        ),
        run=replace(RunConfig(), seed=11, variant="stateful_v2"),
    )

    phase_1_evaluator = build_evaluator(config.task, variant="stateful_v2", generation_id=0)
    phase_2_evaluator = build_evaluator(config.task, variant="stateful_v2", generation_id=6)

    assert phase_1_evaluator.evaluation_delay_steps == (5,)
    assert phase_2_evaluator.evaluation_delay_steps == (5, 8)
    assert phase_1_evaluator.curriculum_phase == "phase_1"
    assert phase_2_evaluator.curriculum_phase == "phase_2"


def test_curriculum_switch_generation_supports_4_6_and_8() -> None:
    for switch_generation in (4, 6, 8):
        config = AppConfig(
            task=replace(
                TaskConfig(),
                name="bit_memory",
                activation_steps=4,
                temporal_delay_steps=8,
                curriculum_enabled=True,
                curriculum_phase_switch_generation=switch_generation,
                curriculum_phase_1_delay_steps=(5,),
                curriculum_phase_2_delay_steps=(5, 8),
            ),
            run=replace(RunConfig(), seed=11, variant="stateful_v2"),
        )

        before = build_evaluator(config.task, variant="stateful_v2", generation_id=max(0, switch_generation - 1))
        after = build_evaluator(config.task, variant="stateful_v2", generation_id=switch_generation)

        assert curriculum_switch_generation(config.task) == switch_generation
        assert before.evaluation_delay_steps == (5,)
        assert after.evaluation_delay_steps == (5, 8)
        assert before.curriculum_phase == "phase_1"
        assert after.curriculum_phase == "phase_2"


def test_stateful_plastic_with_zero_eta_matches_stateful_evaluation() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=1),
        run=replace(RunConfig(), seed=11, variant="stateful"),
    )
    stateful_evaluator = build_evaluator(config.task, variant="stateful")
    plastic_evaluator = build_evaluator(config.task, variant="stateful_plastic")
    adapter = TensorNEATAdapter(config=config, num_inputs=stateful_evaluator.input_size, num_outputs=stateful_evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])

    stateful_result = stateful_evaluator.evaluate(genome)
    plastic_result = plastic_evaluator.evaluate(genome)

    assert stateful_result.score == plastic_result.score
    assert stateful_result.raw_metrics["sequence_predictions"] == plastic_result.raw_metrics["sequence_predictions"]
    assert plastic_result.raw_metrics["plasticity_enabled"] is True
    assert plastic_result.raw_metrics["mean_abs_delta_w"] == 0.0


def test_stateful_v2_with_disabled_slow_path_matches_stateful_evaluation() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=1),
        run=replace(RunConfig(), seed=11, variant="stateful"),
    )
    stateful_evaluator = build_evaluator(config.task, variant="stateful")
    v2_evaluator = build_evaluator(config.task, variant="stateful_v2")
    adapter = TensorNEATAdapter(config=config, num_inputs=stateful_evaluator.input_size, num_outputs=stateful_evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])

    stateful_result = stateful_evaluator.evaluate(genome)
    v2_result = v2_evaluator.evaluate(genome)

    assert stateful_result.score == v2_result.score
    assert stateful_result.raw_metrics["sequence_predictions"] == v2_result.raw_metrics["sequence_predictions"]


def test_stateful_v2_gated_evaluator_is_deterministic() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=3),
        run=replace(RunConfig(), seed=11, variant="stateful_v2_gated"),
    )
    evaluator = build_evaluator(config.task, variant="stateful_v2_gated")
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])
    first = evaluator.evaluate(genome)
    second = evaluator.evaluate(genome)
    assert first.score == second.score
    assert first.raw_metrics["gate_mean"] == second.raw_metrics["gate_mean"]
    assert "gate_selectivity" in first.raw_metrics


def test_content_gated_evaluator_is_deterministic() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=3),
        run=replace(RunConfig(), seed=11, variant="content_gated"),
    )
    evaluator = build_evaluator(config.task, variant="content_gated")
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])
    first = evaluator.evaluate(genome)
    second = evaluator.evaluate(genome)
    assert first.score == second.score
    assert first.raw_metrics["match_mean"] == second.raw_metrics["match_mean"]


def test_stateful_v3_kv_evaluator_is_deterministic() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=3),
        run=replace(RunConfig(), seed=11, variant="stateful_v3_kv"),
    )
    evaluator = build_evaluator(config.task, variant="stateful_v3_kv")
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])
    first = evaluator.evaluate(genome)
    second = evaluator.evaluate(genome)
    assert first.score == second.score
    assert first.raw_metrics["mean_key_state"] == second.raw_metrics["mean_key_state"]


def test_stateful_plastic_evaluator_uses_custom_delta_weight_clamp() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=1),
        run=replace(RunConfig(), seed=11, variant="stateful_plastic"),
    )

    evaluator = build_evaluator(
        config.task,
        variant="stateful_plastic",
        delta_w_clamp=0.25,
    )

    assert isinstance(evaluator.executor, PlasticNetworkExecutor)
    assert evaluator.executor.delta_w_clamp == 0.25


def test_stateful_plastic_ad_with_zero_eta_matches_stateful_evaluation() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=1),
        run=replace(RunConfig(), seed=11, variant="stateful"),
    )
    stateful_evaluator = build_evaluator(config.task, variant="stateful")
    adaptive_evaluator = build_evaluator(config.task, variant="stateful_plastic_ad", delta_w_clamp=0.5)
    adapter = TensorNEATAdapter(config=config, num_inputs=stateful_evaluator.input_size, num_outputs=stateful_evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])

    stateful_result = stateful_evaluator.evaluate(genome)
    adaptive_result = adaptive_evaluator.evaluate(genome)

    assert stateful_result.score == adaptive_result.score
    assert stateful_result.raw_metrics["sequence_predictions"] == adaptive_result.raw_metrics["sequence_predictions"]
    assert adaptive_result.raw_metrics["plasticity_enabled"] is True
    assert adaptive_result.raw_metrics["mean_abs_delta_w"] == 0.0


def test_stateful_plastic_ad_evaluator_uses_custom_delta_weight_clamp() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=1),
        run=replace(RunConfig(), seed=11, variant="stateful_plastic_ad"),
    )

    evaluator = build_evaluator(
        config.task,
        variant="stateful_plastic_ad",
        delta_w_clamp=0.25,
    )

    assert isinstance(evaluator.executor, AdaptivePlasticNetworkExecutor)
    assert evaluator.executor.delta_w_clamp == 0.25


def test_stateful_plastic_ad_narrow_with_zero_eta_matches_stateful_evaluation() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=1),
        run=replace(RunConfig(), seed=11, variant="stateful"),
    )
    stateful_evaluator = build_evaluator(config.task, variant="stateful")
    narrow_evaluator = build_evaluator(config.task, variant="stateful_plastic_ad_narrow", delta_w_clamp=0.5)
    adapter = TensorNEATAdapter(config=config, num_inputs=stateful_evaluator.input_size, num_outputs=stateful_evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])

    stateful_result = stateful_evaluator.evaluate(genome)
    narrow_result = narrow_evaluator.evaluate(genome)

    assert stateful_result.score == narrow_result.score
    assert stateful_result.raw_metrics["sequence_predictions"] == narrow_result.raw_metrics["sequence_predictions"]
    assert narrow_result.raw_metrics["mean_abs_decay_term"] == 0.0


def test_stateful_plastic_ad_fixed_decay_with_zero_eta_matches_stateful_evaluation() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=1),
        run=replace(RunConfig(), seed=11, variant="stateful"),
    )
    stateful_evaluator = build_evaluator(config.task, variant="stateful")
    fixed_decay_evaluator = build_evaluator(config.task, variant="stateful_plastic_ad_d01", delta_w_clamp=0.5)
    adapter = TensorNEATAdapter(config=config, num_inputs=stateful_evaluator.input_size, num_outputs=stateful_evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])

    stateful_result = stateful_evaluator.evaluate(genome)
    fixed_decay_result = fixed_decay_evaluator.evaluate(genome)

    assert stateful_result.score == fixed_decay_result.score
    assert stateful_result.raw_metrics["sequence_predictions"] == fixed_decay_result.raw_metrics["sequence_predictions"]
    assert fixed_decay_result.raw_metrics["mean_abs_decay_term"] == 0.0


def test_fixed_decay_variant_initializes_and_preserves_plastic_d_value() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=1),
        run=replace(RunConfig(), seed=11, variant="stateful_plastic_ad_d01"),
    )
    evaluator = build_evaluator(config.task, variant="stateful_plastic_ad_d01")
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])
    assert all(np.isclose(conn.plastic_d, -0.1) for conn in genome.connections if conn.enabled)

    next_state, _parent_ids = adapter.advance(
        state,
        np.zeros(adapter.population_size, dtype=np.float32),
        [f"c-{index}" for index in range(adapter.population_size)],
    )
    next_nodes, next_conns = adapter.ask(next_state)
    next_genome = arrays_to_genome_model(adapter.genome, next_nodes[0], next_conns[0])
    assert all(np.isclose(conn.plastic_d, -0.1) for conn in next_genome.connections if conn.enabled)


def test_narrow_decay_variant_initializes_within_small_negative_range() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=1),
        run=replace(RunConfig(), seed=11, variant="stateful_plastic_ad_narrow"),
    )
    evaluator = build_evaluator(config.task, variant="stateful_plastic_ad_narrow")
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)
    assert np.isclose(adapter.genome.conn_gene.plastic_d_init_mean, -0.02)
    assert np.isclose(adapter.genome.conn_gene.plastic_d_init_std, 0.01)
    assert np.isclose(adapter.genome.conn_gene.plastic_d_mutate_power, 0.01)
    assert np.isclose(adapter.genome.conn_gene.plastic_d_lower_bound, -0.1)
    assert np.isclose(adapter.genome.conn_gene.plastic_d_upper_bound, 0.0)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])
    enabled_plastic_d = [conn.plastic_d for conn in genome.connections if conn.enabled]

    assert enabled_plastic_d
    assert all(-0.1 <= value <= 0.0 for value in enabled_plastic_d)

    next_state, _parent_ids = adapter.advance(
        state,
        np.zeros(adapter.population_size, dtype=np.float32),
        [f"c-{index}" for index in range(adapter.population_size)],
    )
    next_nodes, next_conns = adapter.ask(next_state)
    next_genome = arrays_to_genome_model(adapter.genome, next_nodes[0], next_conns[0])
    next_enabled_plastic_d = [conn.plastic_d for conn in next_genome.connections if conn.enabled]

    assert next_enabled_plastic_d
    assert all(-0.1 <= value <= 0.0 for value in next_enabled_plastic_d)


def test_stateful_v2_variant_initializes_slow_parameters_within_expected_bounds() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="bit_memory", activation_steps=4, temporal_delay_steps=1),
        run=replace(RunConfig(), seed=11, variant="stateful_v2"),
    )
    evaluator = build_evaluator(config.task, variant="stateful_v2")
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)
    assert np.isclose(adapter.genome.node_gene.alpha_slow_init_mean, config.mutation.alpha_slow_init_mean)
    assert np.isclose(adapter.genome.node_gene.slow_input_gain_init_mean, config.mutation.slow_input_gain_init_mean)
    assert np.isclose(adapter.genome.node_gene.slow_output_gain_init_mean, config.mutation.slow_output_gain_init_mean)

    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])
    dynamic_nodes = [node for node in genome.nodes if not node.is_input]

    assert dynamic_nodes
    assert all(0.0 <= node.alpha_slow <= 1.0 for node in dynamic_nodes)
    assert all(
        config.mutation.slow_input_gain_lower_bound <= node.slow_input_gain <= config.mutation.slow_input_gain_upper_bound
        for node in dynamic_nodes
    )
    assert all(
        config.mutation.slow_output_gain_lower_bound <= node.slow_output_gain <= config.mutation.slow_output_gain_upper_bound
        for node in dynamic_nodes
    )


def test_stateless_variant_forces_alpha_to_zero() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="delayed_xor"),
        run=replace(RunConfig(), seed=3, variant="stateless"),
    )
    evaluator = build_evaluator(config.task)
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)
    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])

    assert np.allclose([node.alpha for node in genome.nodes], 0.0)
