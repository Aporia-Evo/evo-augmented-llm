from __future__ import annotations

import numpy as np
import pytest

from evolve.custom_neuron import (
    AdaptivePlasticNetworkExecutor,
    ContentGatedNetworkExecutor,
    PlasticNetworkExecutor,
    StatefulNetworkExecutor,
    StatefulV2GatedNetworkExecutor,
    StatefulV3KVNetworkExecutor,
    StatefulV5AddressedSlotsNetworkExecutor,
    StatefulV6DeltaMemoryNetworkExecutor,
    StatefulV4SlotsNetworkExecutor,
    StatefulV2NetworkExecutor,
    _match_conditioned_focus_sharpen,
    _positive_sum_normalize,
    _product_key_focus,
    _value_level_logit_decode,
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


def test_product_key_focus_outputs_positive_distribution() -> None:
    vec = np.array([0.2, 0.4, 0.1, 0.3, 0.5, 0.1, 0.2, 0.2], dtype=np.float64)

    focus = _product_key_focus(vec)

    assert focus.shape == vec.shape
    assert np.all(focus >= 0.0)
    assert float(np.sum(focus)) == pytest.approx(1.0, abs=1e-5)


def test_product_key_focus_is_not_equivalent_to_plain_sum_normalize() -> None:
    # A vector where the two halves carry very different total mass. The
    # product-key cross-coupling must produce a distinctly different
    # distribution from the plain positive sum normalization — otherwise the
    # operator has collapsed into a no-op (the V15j regression we are
    # guarding against).
    vec = np.array([0.9, 0.8, 0.7, 0.6, 0.05, 0.05, 0.05, 0.05], dtype=np.float64)

    plain = _positive_sum_normalize(vec)
    pk = _product_key_focus(vec)

    assert not np.allclose(plain, pk, atol=1e-4)


def test_product_key_focus_is_not_equivalent_to_per_half_normalize_with_equal_mass() -> None:
    # The V15j formulation reduced to "normalize each half independently then
    # give each half 0.5 mass". Ensure _product_key_focus is *not* equivalent
    # to that degenerate operation when the two halves carry unequal mass.
    vec = np.array([0.9, 0.8, 0.7, 0.6, 0.05, 0.05, 0.05, 0.05], dtype=np.float64)

    half_a = _positive_sum_normalize(vec[:4]) * 0.5
    half_b = _positive_sum_normalize(vec[4:]) * 0.5
    degenerate = np.concatenate((half_a, half_b))

    pk = _product_key_focus(vec)

    assert not np.allclose(degenerate, pk, atol=1e-4)


def test_product_key_focus_couples_mass_between_halves() -> None:
    # The product-key cross-coupling scales each half by the *opposite*
    # half's raw mean. That means a half with large raw mean shrinks its
    # counterpart's multiplier on its own side (it scales the opposite half
    # by a large factor and itself by a small one). The net effect is a
    # symmetry-breaking penalty against per-half imbalance: the half that
    # started dominant in raw mass ends up contributing *less* total mass
    # after renormalization, while the subordinate half gets boosted.
    vec = np.array([0.9, 0.8, 0.7, 0.6, 0.05, 0.05, 0.05, 0.05], dtype=np.float64)

    pk = _product_key_focus(vec)

    # raw mean_a = 0.75, raw mean_b = 0.05. After cross-coupling the
    # originally-heavy first half must now carry *less* than half the mass.
    assert float(np.sum(pk[:4])) < 0.5
    assert float(np.sum(pk[4:])) > 0.5
    # But the operation is symmetric: swap halves and the relation flips.
    vec_swapped = np.array(
        [0.05, 0.05, 0.05, 0.05, 0.9, 0.8, 0.7, 0.6], dtype=np.float64
    )
    pk_swapped = _product_key_focus(vec_swapped)
    assert float(np.sum(pk_swapped[:4])) > 0.5
    assert float(np.sum(pk_swapped[4:])) < 0.5


def test_product_key_focus_degrades_gracefully_for_tiny_vectors() -> None:
    single = np.array([0.4], dtype=np.float64)
    assert np.allclose(_product_key_focus(single), _positive_sum_normalize(single))

    empty_half = np.array([0.2, 0.3, 0.4, 0.0, 0.0, 0.0], dtype=np.float64)
    focus = _product_key_focus(empty_half)
    # Second half has zero mass -> fall back to plain positive sum normalize.
    assert np.allclose(focus, _positive_sum_normalize(empty_half))


def test_match_conditioned_focus_sharpen_is_noop_on_nonpositive_match() -> None:
    # v15l-B: negative or zero key/query cosine must leave q_focus untouched
    # so the new lever cannot amplify mis-aligned readouts.
    q_focus = _positive_sum_normalize(
        np.array([0.05, 0.1, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float64)
    )
    for cos in (-0.9, -0.1, 0.0):
        sharpened = _match_conditioned_focus_sharpen(q_focus, cos)
        assert np.allclose(sharpened, q_focus)


def test_match_conditioned_focus_sharpen_increases_peak_but_stays_bounded() -> None:
    # v15l-B: at maximum positive match the sharpening must (a) raise the
    # peak probability, (b) lower Shannon entropy, but (c) stay bounded so
    # an already-peaked distribution does not collapse into a Dirac spike.
    q_focus = _positive_sum_normalize(
        np.array([0.04, 0.06, 0.5, 0.04, 0.06, 0.2, 0.05, 0.05], dtype=np.float64)
    )
    sharpened = _match_conditioned_focus_sharpen(q_focus, 1.0)
    assert sharpened.max() > q_focus.max()
    assert sharpened.max() < 0.95  # bounded - no dirac collapse
    base_entropy = -float(np.sum(q_focus * np.log(q_focus + 1e-12)))
    sharp_entropy = -float(np.sum(sharpened * np.log(sharpened + 1e-12)))
    assert sharp_entropy < base_entropy
    assert np.isclose(float(np.sum(sharpened)), 1.0, atol=1e-6)


def test_match_conditioned_focus_sharpen_leaves_uniform_distribution_unchanged() -> None:
    # v15l-B: power-sharpening a uniform distribution must be a fixed point
    # (any positive exponent on a constant vector renormalises back to itself),
    # so the lever cannot inject spurious structure where there is none.
    uniform = np.full(8, 1.0 / 8.0, dtype=np.float64)
    sharpened = _match_conditioned_focus_sharpen(uniform, 1.0)
    assert np.allclose(sharpened, uniform)


def test_value_level_decode_is_symmetric_around_zero() -> None:
    # v15m: the decoder must be antisymmetric under sign flip of the
    # pre-decode scalar (default levels [-1, +1] are symmetric around 0).
    for x in (-0.9, -0.4, 0.0, 0.3, 0.7):
        assert np.isclose(
            _value_level_logit_decode(x) + _value_level_logit_decode(-x),
            0.0,
            atol=1e-12,
        )


def test_value_level_decode_stays_bounded_inside_level_hull() -> None:
    # v15m: softmax-weighted average over levels must stay inside the
    # convex hull of the levels, regardless of the pre-decode scalar.
    for x in (-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0):
        decoded = _value_level_logit_decode(x)
        assert -1.0 <= decoded <= 1.0


def test_value_level_decode_sharpens_weak_signals() -> None:
    # v15m: a weak positive pre-decode (e.g. 0.3) must map to a strictly
    # larger positive output so the decoder exerts real sharpening; the
    # default temperature 2.5 is the intervention.
    assert _value_level_logit_decode(0.3) > 0.3
    assert _value_level_logit_decode(-0.3) < -0.3
    # At the origin the decoder must be a fixed point (no drift).
    assert np.isclose(_value_level_logit_decode(0.0), 0.0, atol=1e-12)


def test_value_level_decode_supports_custom_level_sets() -> None:
    # v15m: custom value_levels must work and the weighted average must
    # stay inside the level hull for non-binary sets too.
    levels = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    decoded = _value_level_logit_decode(0.4, value_levels=levels)
    assert -1.0 <= decoded <= 1.0
    # Closer to 0.5 than to 0 when pre_decode is 0.4, so output should
    # sit between 0.0 and 0.5 rather than jumping to the extreme.
    assert 0.0 < decoded < 0.6


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


def test_stateful_v2_gated_keeps_state_when_gate_is_high() -> None:
    genome = _single_connection_genome(weight=1.0, eta=0.0, alpha=12.0, alpha_slow=0.9, slow_input_gain=5.0, slow_output_gain=0.1)
    executor = StatefulV2GatedNetworkExecutor(activation_steps=1)
    outputs = executor.run_sequence(genome, [[1.0], [0.0], [0.0]])
    metrics = executor.last_episode_metrics()
    assert float(outputs[2, 0]) > 0.001
    assert metrics.gate_mean > 0.9
    assert metrics.gate_variance < 0.1


def test_stateful_v2_gated_writes_input_when_gate_is_low() -> None:
    genome = _single_connection_genome(weight=1.0, eta=0.0, alpha=-12.0, alpha_slow=0.9, slow_input_gain=-5.0, slow_output_gain=0.1)
    executor = StatefulV2GatedNetworkExecutor(activation_steps=1)
    outputs = executor.run_sequence(genome, [[1.0], [0.0]])
    metrics = executor.last_episode_metrics()
    assert float(outputs[0, 0]) > float(outputs[1, 0])
    assert metrics.gate_mean < 0.2


def test_content_gated_executor_reports_match_metrics() -> None:
    genome = _single_connection_genome(weight=1.0, eta=0.0, alpha=0.5)
    genome = GenomeModel(
        input_ids=genome.input_ids,
        output_ids=genome.output_ids,
        nodes=(
            genome.nodes[0],
            NodeGeneModel(
                node_id=1,
                bias=0.1,
                alpha=0.6,
                content_w_key=1.0,
                content_b_key=0.0,
                content_w_query=0.8,
                content_b_query=0.0,
                content_temperature=1.2,
                content_b_match=0.0,
                is_output=True,
            ),
        ),
        connections=genome.connections,
    )
    executor = ContentGatedNetworkExecutor(activation_steps=1)
    executor.run_sequence(genome, [[1.0], [0.0], [1.0]], step_roles=["store", "distractor", "query"])
    metrics = executor.last_episode_metrics()
    assert metrics.match_mean > 0.0
    assert metrics.match_variance >= 0.0
    assert metrics.match_selectivity >= 0.0


def test_stateful_v3_kv_separates_key_and_value_state() -> None:
    genome = GenomeModel(
        input_ids=(0, 2),
        output_ids=(1,),
        nodes=(
            NodeGeneModel(node_id=0, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=2, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(
                node_id=1,
                bias=0.0,
                alpha=0.6,
                alpha_slow=0.9,
                slow_output_gain=0.3,
                content_w_key=1.0,
                content_b_key=0.0,
                content_w_query=2.0,
                content_b_query=2.0,
                content_temperature=1.2,
                content_b_match=0.0,
                is_output=True,
            ),
        ),
        connections=(
            ConnectionGeneModel(in_id=0, out_id=1, historical_marker=0, weight=1.0, enabled=True, eta=0.0),
            ConnectionGeneModel(in_id=2, out_id=1, historical_marker=1, weight=0.5, enabled=True, eta=0.0),
        ),
    )
    executor = StatefulV3KVNetworkExecutor(activation_steps=1)
    executor.run_sequence(genome, [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], step_roles=["store", "distractor", "query"])
    metrics = executor.last_episode_metrics()
    assert metrics.mean_key_state > 0.0
    assert metrics.mean_value_state > 0.0
    assert metrics.key_value_separation >= 0.0
    assert metrics.write_gate_at_store > metrics.write_gate_at_distractor
    assert metrics.store_vs_distractor_write_gap > 0.0
    assert metrics.match_at_query > 0.0
    assert metrics.query_value_read_strength > 0.0


def test_stateful_v4_slots_reports_slot_metrics() -> None:
    genome = GenomeModel(
        input_ids=(0, 1, 4, 5, 6, 7),
        output_ids=(2,),
        nodes=(
            NodeGeneModel(node_id=0, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=1, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=4, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=5, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=6, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=7, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(
                node_id=2,
                bias=0.0,
                alpha=0.6,
                alpha_slow=0.9,
                content_w_key=1.0,
                content_b_key=0.5,
                content_w_query=1.0,
                content_b_query=0.0,
                content_temperature=1.2,
                is_output=True,
            ),
        ),
        connections=(
            ConnectionGeneModel(in_id=0, out_id=2, historical_marker=0, weight=1.0, enabled=True, eta=0.0),
            ConnectionGeneModel(in_id=1, out_id=2, historical_marker=1, weight=1.0, enabled=True, eta=0.0),
            ConnectionGeneModel(in_id=7, out_id=2, historical_marker=2, weight=0.8, enabled=True, eta=0.0),
        ),
    )
    executor = StatefulV4SlotsNetworkExecutor(activation_steps=1)
    sequence = [
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.3],  # store key0
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.9],  # store key1
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # query key0
    ]
    executor.run_sequence(genome, sequence, step_roles=["store", "store", "query"])
    metrics = executor.last_episode_metrics()
    assert metrics.slot_key_separation >= 0.0
    assert metrics.slot_value_separation >= 0.0
    assert metrics.slot_query_focus >= 0.0
    assert metrics.slot_utilization > 0.0
    assert metrics.query_slot_match_max >= 0.5
    assert metrics.slot_distractor_leak >= 0.0


def test_stateful_v5_addressed_slots_breaks_symmetry_and_keeps_rw_split() -> None:
    genome = GenomeModel(
        input_ids=(0, 1, 2, 4, 5, 6),
        output_ids=(3,),
        nodes=(
            NodeGeneModel(node_id=0, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=1, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=2, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=4, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=5, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=6, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(
                node_id=3,
                bias=0.0,
                alpha=0.6,
                alpha_slow=0.9,
                content_w_key=0.35,
                content_b_key=0.25,
                content_w_query=0.8,
                content_b_query=0.1,
                content_temperature=1.4,
                content_b_match=0.15,
                slow_output_gain=0.2,
                is_output=True,
            ),
        ),
        connections=(
            ConnectionGeneModel(in_id=0, out_id=3, historical_marker=0, weight=1.0, enabled=True, eta=0.0),
            ConnectionGeneModel(in_id=1, out_id=3, historical_marker=1, weight=1.0, enabled=True, eta=0.0),
        ),
    )
    sequence = [
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    ]
    executor = StatefulV5AddressedSlotsNetworkExecutor(activation_steps=1)
    executor.run_sequence(genome, sequence, step_roles=["store", "query"])
    metrics = executor.last_episode_metrics()

    assert metrics.mean_write_address_focus > 0.0
    assert metrics.mean_read_address_focus > 0.0
    assert abs(metrics.readout_address_concentration - 0.5) > 1e-6
    assert abs(metrics.write_read_address_gap) > 0.0


def test_q_focus_sharpening_scale_is_no_longer_saturated() -> None:
    # Regression guard for PR3. The v14x sharpening block divided
    # ``q_focus_centered`` by a hardcoded 0.05 before feeding it into
    # ``tanh``. ``q_focus`` is a probability distribution over ``d_key``
    # slots with mean ``1/d_key`` (~0.125 for d_key=8), so for any
    # distribution with non-trivial spread the argument landed deep in
    # the tanh saturation region and the multiplier
    # ``1 + strength * tanh(...)`` collapsed toward an effectively
    # binary ``1 +/- strength`` sign flip — not the smooth sharpening
    # it was supposed to be. The fix replaces the constant by the
    # runtime std of ``q_focus`` so that the ``tanh`` argument is a
    # z-score, which makes the sharpening *scale invariant*: rescaling
    # the deviations of ``q_focus`` around ``1/d_key`` by any positive
    # constant leaves the new ``tanh`` argument unchanged, whereas the
    # old ``/0.05`` formulation is explicitly not scale invariant and
    # saturates more as the distribution spreads.
    q_focus = np.array([0.20, 0.18, 0.15, 0.12, 0.11, 0.10, 0.08, 0.06])
    q_focus = q_focus / q_focus.sum()
    q_focus_centered = q_focus - q_focus.mean()
    strength = 0.03

    # Build a "wider" variant by scaling the deviations around the
    # uniform ``1/d_key`` mean. This is still a valid probability
    # distribution (same sum, symmetric rescaling) and captures the
    # "more concentrated q_focus" regime that the sharpening block is
    # supposed to accentuate smoothly.
    wider_centered = 1.5 * q_focus_centered
    wider = (1.0 / q_focus.size) + wider_centered
    assert float(np.min(wider)) > 0.0

    new_scale_a = float(np.std(q_focus)) + 1e-6
    new_scale_b = float(np.std(wider)) + 1e-6
    new_tanh_a = np.tanh(q_focus_centered / new_scale_a)
    new_tanh_b = np.tanh(wider_centered / new_scale_b)

    old_tanh_a = np.tanh(q_focus_centered / 0.05)
    old_tanh_b = np.tanh(wider_centered / 0.05)

    # Core invariant: new formulation is exactly scale-invariant, so
    # rescaling the deviations produces an identical tanh shape.
    assert np.allclose(new_tanh_a, new_tanh_b, atol=1e-9)

    # Old formulation is NOT scale-invariant: doubling the spread pushes
    # the argument further into saturation, so the maximum absolute tanh
    # strictly increases and the shape changes.
    assert float(np.max(np.abs(old_tanh_b))) > float(np.max(np.abs(old_tanh_a)))
    assert not np.allclose(old_tanh_a, old_tanh_b, atol=1e-2)

    # On the wider distribution the old formulation is visibly in the
    # saturation band at the extremes, whereas the new formulation keeps
    # the largest slot's ``tanh`` strictly below the old maximum.
    assert float(np.max(np.abs(old_tanh_b))) > 0.95
    assert float(np.max(np.abs(new_tanh_b))) < float(np.max(np.abs(old_tanh_b)))

    # The actual multiplier applied to ``q_focus`` is bounded in the
    # documented ``[1 - strength, 1 + strength]`` envelope and preserves
    # the slot ordering (largest ``q_focus`` entry still gets the
    # largest amplification).
    new_multiplier = 1.0 + strength * new_tanh_a
    assert bool(np.all(new_multiplier > 1.0 - strength - 1e-9))
    assert bool(np.all(new_multiplier < 1.0 + strength + 1e-9))
    assert int(np.argmax(new_multiplier)) == int(np.argmax(q_focus))


def test_stateful_v6_delta_memory_updates_state_and_reports_metrics() -> None:
    genome = GenomeModel(
        input_ids=(0, 1, 2),
        output_ids=(3,),
        nodes=(
            NodeGeneModel(node_id=0, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=1, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=2, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(
                node_id=3,
                bias=0.0,
                alpha=0.6,
                content_w_key=0.9,
                content_b_key=0.2,
                content_w_query=1.1,
                content_b_query=0.1,
                content_temperature=1.2,
                content_b_match=0.3,
                is_output=True,
            ),
        ),
        connections=(
            ConnectionGeneModel(in_id=0, out_id=3, historical_marker=0, weight=1.0, enabled=True, eta=0.0),
            ConnectionGeneModel(in_id=1, out_id=3, historical_marker=1, weight=0.5, enabled=True, eta=0.0),
            ConnectionGeneModel(in_id=2, out_id=3, historical_marker=2, weight=-0.25, enabled=True, eta=0.0),
        ),
    )
    sequence = [[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.0, 1.0, 0.0]]
    roles = ["store", "distractor", "query"]
    executor = StatefulV6DeltaMemoryNetworkExecutor(activation_steps=1)
    first = executor.run_sequence(genome, sequence, step_roles=roles)
    metrics = executor.last_episode_metrics()
    second = executor.run_sequence(genome, sequence, step_roles=roles)
    assert np.allclose(first, second)
    assert metrics.mean_memory_frobenius_norm > 0.0
    assert metrics.delta_correction_magnitude > 0.0
    assert metrics.memory_read_strength > 0.0
    assert metrics.store_vs_distractor_beta_gap != 0.0
    # ``query_memory_alignment`` must no longer be a silent clone of
    # ``key_query_cosine_at_query`` (the historical V14a bug where both
    # slots got the same ``key_query_cos`` value).
    assert np.isfinite(metrics.query_memory_alignment)
    assert 0.0 <= metrics.query_memory_alignment <= 1.0 + 1e-6
    assert not np.isclose(
        metrics.query_memory_alignment,
        metrics.key_query_cosine_at_query,
        atol=1e-6,
    )


def _v6_delta_memory_genome(*, alpha_slow: float) -> GenomeModel:
    return GenomeModel(
        input_ids=(0, 1, 2),
        output_ids=(3,),
        nodes=(
            NodeGeneModel(node_id=0, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=1, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(node_id=2, bias=0.0, alpha=0.0, is_input=True),
            NodeGeneModel(
                node_id=3,
                bias=0.0,
                alpha=0.6,
                alpha_slow=alpha_slow,
                content_w_key=0.9,
                content_b_key=0.2,
                content_w_query=1.1,
                content_b_query=0.1,
                content_temperature=1.2,
                content_b_match=0.3,
                is_output=True,
            ),
        ),
        connections=(
            ConnectionGeneModel(in_id=0, out_id=3, historical_marker=0, weight=1.0, enabled=True, eta=0.0),
            ConnectionGeneModel(in_id=1, out_id=3, historical_marker=1, weight=0.5, enabled=True, eta=0.0),
            ConnectionGeneModel(in_id=2, out_id=3, historical_marker=2, weight=-0.25, enabled=True, eta=0.0),
        ),
    )


def test_v6_delta_memory_alpha_slow_controls_retention() -> None:
    # v15n-A: node.alpha_slow is the per-neuron memory_decay knob for the
    # V6 delta-memory executor. Verify that two genomes which are identical
    # except for ``alpha_slow`` diverge in memory-retention diagnostics over
    # a long distractor delay, with the low-``alpha_slow`` genome forgetting
    # more aggressively than the high-``alpha_slow`` genome.
    executor = StatefulV6DeltaMemoryNetworkExecutor(activation_steps=1)

    # One store step, a long sequence of distractor steps to exercise
    # memory decay, then a query. Low alpha_slow => fast decay => the
    # Frobenius norm of the write at query time should be clearly smaller
    # than with high alpha_slow.
    sequence = (
        [[1.0, 0.0, 0.0]]
        + [[0.2, 0.8, 0.0]] * 10
        + [[0.0, 1.0, 0.0]]
    )
    roles = ["store"] + (["distractor"] * 10) + ["query"]

    genome_short = _v6_delta_memory_genome(alpha_slow=0.0)   # decay = 0.90
    genome_long = _v6_delta_memory_genome(alpha_slow=1.0)    # decay = 0.99

    executor.run_sequence(genome_short, sequence, step_roles=roles)
    metrics_short = executor.last_episode_metrics()
    executor.run_sequence(genome_long, sequence, step_roles=roles)
    metrics_long = executor.last_episode_metrics()

    # High-retention genome must carry a strictly larger residual memory
    # mass at query time than the aggressively-decaying one.
    assert (
        metrics_long.mean_memory_frobenius_norm
        > metrics_short.mean_memory_frobenius_norm + 1e-3
    )
    # Baseline sanity: the hardcoded-0.97 behaviour is approximated by
    # alpha_slow ~ 0.778.  A genome at alpha_slow = 0.778 should sit
    # between the two extremes on the same metric.
    genome_mid = _v6_delta_memory_genome(alpha_slow=0.778)
    executor.run_sequence(genome_mid, sequence, step_roles=roles)
    metrics_mid = executor.last_episode_metrics()
    assert (
        metrics_short.mean_memory_frobenius_norm
        < metrics_mid.mean_memory_frobenius_norm
        < metrics_long.mean_memory_frobenius_norm
    )


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
