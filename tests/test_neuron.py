from __future__ import annotations

import numpy as np

from evolve.custom_neuron import (
    AdaptivePlasticNetworkExecutor,
    PlasticNetworkExecutor,
    StatefulNetworkExecutor,
    StatefulV2NetworkExecutor,
    clamp_alpha,
    clamp_delta_weight,
    update_adaptive_delta_weight,
    update_memory,
)
from evolve.genome_codec import ConnectionGeneModel, GenomeModel, NodeGeneModel


def test_alpha_is_clamped_into_unit_interval() -> None:
    assert clamp_alpha(-0.25) == 0.0
    assert clamp_alpha(0.4) == 0.4
    assert clamp_alpha(1.8) == 1.0


def test_state_update_uses_clamped_alpha() -> None:
    updated = update_memory(previous_memory=0.25, alpha=1.5, weighted_sum=0.5, bias=-0.1)
    assert updated == 0.65


def test_delta_weight_is_clamped_symmetrically() -> None:
    assert clamp_delta_weight(-2.5, 1.0) == -1.0
    assert clamp_delta_weight(0.4, 1.0) == 0.4
    assert clamp_delta_weight(2.5, 1.0) == 1.0


def test_plastic_executor_matches_stateful_when_eta_is_zero() -> None:
    genome = _single_connection_genome(weight=1.0, eta=0.0)
    sequence = [[1.0], [0.0], [1.0]]

    stateful = StatefulNetworkExecutor(activation_steps=3)
    plastic = PlasticNetworkExecutor(activation_steps=3)

    stateful_outputs = stateful.run_sequence(genome, sequence)
    plastic_outputs = plastic.run_sequence(genome, sequence)

    assert np.allclose(stateful_outputs, plastic_outputs)
    assert plastic.last_episode_metrics().mean_abs_delta_w == 0.0
    assert plastic.last_episode_metrics().max_abs_delta_w == 0.0


def test_plastic_executor_resets_delta_weights_per_episode() -> None:
    genome = _single_connection_genome(weight=1.0, eta=0.4)
    sequence = [[1.0], [1.0], [0.0]]
    executor = PlasticNetworkExecutor(activation_steps=2)

    first = executor.run_sequence(genome, sequence)
    first_metrics = executor.last_episode_metrics()
    second = executor.run_sequence(genome, sequence)
    second_metrics = executor.last_episode_metrics()

    assert np.allclose(first, second)
    assert first_metrics == second_metrics


def test_plastic_executor_reports_clamped_delta_weight_metrics() -> None:
    genome = _single_connection_genome(weight=1.0, eta=10.0)
    executor = PlasticNetworkExecutor(activation_steps=1, delta_w_clamp=1.0)

    executor.run_sequence(genome, [[1.0], [1.0]])
    metrics = executor.last_episode_metrics()

    assert metrics.plasticity_enabled is True
    assert metrics.max_abs_delta_w == 1.0
    assert 0.0 < metrics.mean_abs_delta_w <= 1.0


def test_adaptive_delta_update_treats_decay_as_independent_factor() -> None:
    updated = update_adaptive_delta_weight(
        0.3,
        eta=0.2,
        plastic_a=1.0,
        plastic_b=0.0,
        plastic_c=0.0,
        plastic_d=-1.0,
        pre_value=0.5,
        post_value=0.5,
        clamp=0.5,
    )

    assert np.isclose(updated, 0.05)


def test_adaptive_executor_matches_stateful_when_eta_is_zero() -> None:
    genome = _single_connection_genome(weight=1.0, eta=0.0, plastic_a=1.5, plastic_d=-0.5)
    sequence = [[1.0], [0.0], [1.0]]

    stateful = StatefulNetworkExecutor(activation_steps=3)
    adaptive = AdaptivePlasticNetworkExecutor(activation_steps=3, delta_w_clamp=0.5)

    stateful_outputs = stateful.run_sequence(genome, sequence)
    adaptive_outputs = adaptive.run_sequence(genome, sequence)

    assert np.allclose(stateful_outputs, adaptive_outputs)
    assert adaptive.last_episode_metrics().mean_abs_delta_w == 0.0


def test_adaptive_executor_matches_hebb_when_a_one_and_d_zero() -> None:
    genome = _single_connection_genome(weight=1.0, eta=0.4, plastic_a=1.0, plastic_d=0.0)
    sequence = [[1.0], [1.0], [0.0]]

    hebb = PlasticNetworkExecutor(activation_steps=2, delta_w_clamp=0.5)
    adaptive = AdaptivePlasticNetworkExecutor(activation_steps=2, delta_w_clamp=0.5)

    hebb_outputs = hebb.run_sequence(genome, sequence)
    adaptive_outputs = adaptive.run_sequence(genome, sequence)

    assert np.allclose(hebb_outputs, adaptive_outputs)
    assert hebb.last_episode_metrics().mean_abs_delta_w == adaptive.last_episode_metrics().mean_abs_delta_w
    assert hebb.last_episode_metrics().max_abs_delta_w == adaptive.last_episode_metrics().max_abs_delta_w
    assert hebb.last_episode_metrics().clamp_hit_rate == adaptive.last_episode_metrics().clamp_hit_rate


def test_stateful_v2_executor_matches_stateful_when_slow_path_disabled() -> None:
    genome = _single_connection_genome(weight=1.0, eta=0.0, alpha=0.4, alpha_slow=0.9, slow_input_gain=0.0, slow_output_gain=0.0)
    sequence = [[1.0], [0.0], [1.0]]

    stateful = StatefulNetworkExecutor(activation_steps=3)
    stateful_v2 = StatefulV2NetworkExecutor(activation_steps=3)

    stateful_outputs = stateful.run_sequence(genome, sequence)
    v2_outputs = stateful_v2.run_sequence(genome, sequence)

    assert np.allclose(stateful_outputs, v2_outputs)
    metrics = stateful_v2.last_episode_metrics()
    assert metrics.mean_abs_fast_state > 0.0
    assert metrics.mean_abs_slow_state == 0.0
    assert metrics.slow_fast_contribution_ratio == 0.0


def test_stateful_v2_executor_reports_fast_and_slow_state_metrics() -> None:
    genome = _single_connection_genome(
        weight=1.0,
        eta=0.0,
        alpha=0.2,
        alpha_slow=0.9,
        slow_input_gain=0.5,
        slow_output_gain=1.25,
    )
    executor = StatefulV2NetworkExecutor(activation_steps=1)

    executor.run_sequence(genome, [[1.0], [0.0], [1.0]])
    metrics = executor.last_episode_metrics()

    assert metrics.mean_abs_fast_state > 0.0
    assert metrics.mean_abs_slow_state > 0.0
    assert metrics.slow_fast_contribution_ratio > 0.0


def _single_connection_genome(
    weight: float,
    eta: float,
    *,
    plastic_a: float = 1.0,
    plastic_d: float = 0.0,
    alpha: float = 0.0,
    alpha_slow: float = 0.0,
    slow_input_gain: float = 0.0,
    slow_output_gain: float = 0.0,
) -> GenomeModel:
    return GenomeModel(
        input_ids=(0,),
        output_ids=(1,),
        nodes=(
            NodeGeneModel(node_id=0, bias=0.0, alpha=0.0, is_input=True, is_output=False),
            NodeGeneModel(
                node_id=1,
                bias=0.0,
                alpha=alpha,
                alpha_slow=alpha_slow,
                slow_input_gain=slow_input_gain,
                slow_output_gain=slow_output_gain,
                is_input=False,
                is_output=True,
            ),
        ),
        connections=(
            ConnectionGeneModel(
                in_id=0,
                out_id=1,
                historical_marker=0,
                weight=weight,
                enabled=True,
                eta=eta,
                plastic_a=plastic_a,
                plastic_d=plastic_d,
            ),
        ),
    )
