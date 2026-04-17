from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from evolve.genome_codec import ConnectionGeneModel, GenomeModel


def clamp_alpha(alpha: float) -> float:
    return max(0.0, min(1.0, float(alpha)))


def update_memory(previous_memory: float, alpha: float, weighted_sum: float, bias: float) -> float:
    return clamp_alpha(alpha) * previous_memory + weighted_sum + bias


def clamp_delta_weight(delta_weight: float, clamp: float) -> float:
    limit = max(0.0, float(clamp))
    return max(-limit, min(limit, float(delta_weight)))


def update_hebb_delta_weight(
    previous_delta_weight: float,
    *,
    eta: float,
    pre_value: float,
    post_value: float,
    clamp: float,
) -> float:
    updated_delta = previous_delta_weight + (eta * pre_value * post_value)
    return clamp_delta_weight(updated_delta, clamp)


def update_adaptive_delta_weight(
    previous_delta_weight: float,
    *,
    eta: float,
    plastic_a: float,
    plastic_b: float,
    plastic_c: float,
    plastic_d: float,
    pre_value: float,
    post_value: float,
    clamp: float,
) -> float:
    updated_delta = (
        previous_delta_weight
        + (eta * ((plastic_a * pre_value * post_value) + (plastic_b * pre_value) + (plastic_c * post_value)))
        + (plastic_d * previous_delta_weight)
    )
    return clamp_delta_weight(updated_delta, clamp)


def _positive_sum_normalize(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.maximum(vec, 0.0)
    return clipped / (float(np.sum(clipped)) + eps)


def _clip_vector_norm(vec: np.ndarray, max_norm: float, eps: float = 1e-9) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm:
        return vec
    return vec * (max_norm / (norm + eps))


@dataclass(frozen=True)
class NodeExecutionState:
    memory: float
    output: float


@dataclass(frozen=True)
class PlasticityEpisodeMetrics:
    plasticity_enabled: bool
    mean_abs_delta_w: float
    max_abs_delta_w: float
    clamp_hit_rate: float
    plasticity_active_fraction: float
    mean_abs_decay_term: float
    max_abs_decay_term: float
    decay_effect_ratio: float
    decay_near_zero_fraction: float
    mean_abs_fast_state: float = 0.0
    mean_abs_slow_state: float = 0.0
    slow_fast_contribution_ratio: float = 0.0
    mean_abs_fast_state_during_store: float = 0.0
    mean_abs_slow_state_during_store: float = 0.0
    mean_abs_fast_state_during_query: float = 0.0
    mean_abs_slow_state_during_query: float = 0.0
    mean_abs_fast_state_during_distractor: float = 0.0
    mean_abs_slow_state_during_distractor: float = 0.0
    gate_mean: float = 0.0
    gate_variance: float = 0.0
    gate_at_store: float = 0.0
    gate_at_distractor: float = 0.0
    gate_at_query: float = 0.0
    gate_selectivity: float = 0.0
    gate_store_minus_query: float = 0.0
    gate_query_minus_distractor: float = 0.0
    gate_role_contrast: float = 0.0
    slow_state_at_query: float = 0.0
    fast_state_at_query: float = 0.0
    match_mean: float = 0.0
    match_variance: float = 0.0
    match_at_store: float = 0.0
    match_at_distractor: float = 0.0
    match_at_query: float = 0.0
    match_selectivity: float = 0.0
    query_match_score: float = 0.0
    state_query_alignment: float = 0.0
    content_retention_gap: float = 0.0
    mean_key_state: float = 0.0
    mean_value_state: float = 0.0
    key_value_separation: float = 0.0
    query_key_alignment: float = 0.0
    query_value_read_strength: float = 0.0
    store_key_value_coupling: float = 0.0
    distractor_write_leak: float = 0.0
    readout_selectivity: float = 0.0
    mean_key_state_during_store: float = 0.0
    mean_value_state_during_store: float = 0.0
    mean_key_state_during_query: float = 0.0
    mean_value_state_during_query: float = 0.0
    write_gate_at_store: float = 0.0
    write_gate_at_distractor: float = 0.0
    write_gate_at_query: float = 0.0
    store_vs_distractor_write_gap: float = 0.0
    mean_match_signal: float = 0.0
    value_state_at_query: float = 0.0
    key_state_at_query: float = 0.0
    slot_key_separation: float = 0.0
    slot_value_separation: float = 0.0
    slot_write_focus: float = 0.0
    slot_query_focus: float = 0.0
    slot_readout_selectivity: float = 0.0
    slot_utilization: float = 0.0
    query_slot_match_max: float = 0.0
    slot_distractor_leak: float = 0.0
    mean_write_address_focus: float = 0.0
    mean_read_address_focus: float = 0.0
    write_read_address_gap: float = 0.0
    slot_write_specialization: float = 0.0
    slot_read_specialization: float = 0.0
    address_consistency: float = 0.0
    query_read_alignment: float = 0.0
    store_write_alignment: float = 0.0
    readout_address_concentration: float = 0.0
    mean_beta_write: float = 0.0
    beta_at_store: float = 0.0
    beta_at_distractor: float = 0.0
    beta_at_query: float = 0.0
    store_vs_distractor_beta_gap: float = 0.0
    mean_key_norm: float = 0.0
    mean_query_norm: float = 0.0
    mean_value_norm: float = 0.0
    mean_memory_frobenius_norm: float = 0.0
    query_memory_alignment: float = 0.0
    store_memory_update_strength: float = 0.0
    delta_correction_magnitude: float = 0.0
    memory_read_strength: float = 0.0
    key_query_cosine_mean: float = 0.0
    key_query_cosine_at_query: float = 0.0
    key_variance_mean: float = 0.0
    query_variance_mean: float = 0.0
    key_query_projection_strength: float = 0.0
    query_decoupling_magnitude: float = 0.0


def _incoming_connections_by_target(genome: GenomeModel) -> dict[int, list[ConnectionGeneModel]]:
    incoming_by_target: dict[int, list[ConnectionGeneModel]] = {}
    for conn in genome.connections:
        if conn.enabled:
            incoming_by_target.setdefault(conn.out_id, []).append(conn)
    return incoming_by_target


class StatefulNetworkExecutor:
    def __init__(self, activation_steps: int) -> None:
        self.activation_steps = activation_steps

    def run(self, genome: GenomeModel, inputs: Sequence[float]) -> np.ndarray:
        return self.run_sequence(genome, [inputs], step_roles=None)[-1]

    def run_sequence(
        self,
        genome: GenomeModel,
        input_sequence: Sequence[Sequence[float]],
        *,
        step_roles: Sequence[str] | None = None,
    ) -> np.ndarray:
        del step_roles
        incoming_by_target = _incoming_connections_by_target(genome)

        memory = {node.node_id: 0.0 for node in genome.nodes if not node.is_input}
        outputs = {node.node_id: 0.0 for node in genome.nodes}
        sequence_outputs: list[np.ndarray] = []

        for inputs in input_sequence:
            input_map = {
                node_id: float(value)
                for node_id, value in zip(genome.input_ids, inputs, strict=True)
            }
            outputs.update(input_map)

            # activation_steps is a relaxation pass: signals propagate through the graph
            # multiple times per timestep so deep paths can settle within one timestep.
            # NOTE: memory is also updated on each sub-step, not just once per timestep.
            # This deviates from the documented formula m_t = alpha*m_(t-1) + ... but is
            # intentional for multi-layer networks. For plasticity (V5a), decide explicitly
            # whether Hebbian updates fire once per timestep or once per sub-step.
            for _ in range(self.activation_steps):
                previous_outputs = dict(outputs)
                next_outputs = dict(input_map)

                for node in genome.nodes:
                    if node.is_input:
                        continue

                    incoming_sum = 0.0
                    for conn in incoming_by_target.get(node.node_id, ()):
                        incoming_sum += conn.weight * previous_outputs.get(conn.in_id, 0.0)

                    new_memory = update_memory(
                        previous_memory=memory[node.node_id],
                        alpha=node.alpha,
                        weighted_sum=incoming_sum,
                        bias=node.bias,
                    )
                    memory[node.node_id] = new_memory
                    next_outputs[node.node_id] = math.tanh(new_memory)

                outputs = next_outputs

            sequence_outputs.append(
                np.array(
                    [outputs.get(node_id, 0.0) for node_id in genome.output_ids],
                    dtype=np.float32,
                )
            )

        return np.vstack(sequence_outputs)

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


class StatefulV2NetworkExecutor(StatefulNetworkExecutor):
    def __init__(self, activation_steps: int) -> None:
        super().__init__(activation_steps=activation_steps)
        self._last_metrics = PlasticityEpisodeMetrics(
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

    def run_sequence(
        self,
        genome: GenomeModel,
        input_sequence: Sequence[Sequence[float]],
        *,
        step_roles: Sequence[str] | None = None,
    ) -> np.ndarray:
        incoming_by_target = _incoming_connections_by_target(genome)

        fast_memory = {node.node_id: 0.0 for node in genome.nodes if not node.is_input}
        slow_memory = {node.node_id: 0.0 for node in genome.nodes if not node.is_input}
        outputs = {node.node_id: 0.0 for node in genome.nodes}
        sequence_outputs: list[np.ndarray] = []
        total_abs_fast_state = 0.0
        total_abs_slow_state = 0.0
        total_abs_slow_contribution = 0.0
        total_node_updates = 0
        role_totals: dict[str, tuple[float, float, int]] = {}

        for step_index, inputs in enumerate(input_sequence):
            step_role = _step_role_at(step_roles, step_index)
            input_map = {
                node_id: float(value)
                for node_id, value in zip(genome.input_ids, inputs, strict=True)
            }
            outputs.update(input_map)

            for _ in range(self.activation_steps):
                previous_outputs = dict(outputs)
                next_outputs = dict(input_map)

                for node in genome.nodes:
                    if node.is_input:
                        continue

                    incoming_sum = 0.0
                    for conn in incoming_by_target.get(node.node_id, ()):
                        incoming_sum += conn.weight * previous_outputs.get(conn.in_id, 0.0)

                    new_fast_memory = update_memory(
                        previous_memory=fast_memory[node.node_id],
                        alpha=node.alpha,
                        weighted_sum=incoming_sum,
                        bias=node.bias,
                    )
                    new_slow_memory = (
                        clamp_alpha(node.alpha_slow) * slow_memory[node.node_id]
                        + (node.slow_input_gain * incoming_sum)
                    )
                    slow_contribution = node.slow_output_gain * new_slow_memory

                    fast_memory[node.node_id] = new_fast_memory
                    slow_memory[node.node_id] = new_slow_memory
                    next_outputs[node.node_id] = math.tanh(new_fast_memory + slow_contribution)

                    total_abs_fast_state += abs(new_fast_memory)
                    total_abs_slow_state += abs(new_slow_memory)
                    total_abs_slow_contribution += abs(slow_contribution)
                    total_node_updates += 1
                    if step_role in {"store", "query", "distractor"}:
                        fast_total, slow_total, count = role_totals.get(step_role, (0.0, 0.0, 0))
                        role_totals[step_role] = (
                            fast_total + abs(new_fast_memory),
                            slow_total + abs(new_slow_memory),
                            count + 1,
                        )

                outputs = next_outputs

            sequence_outputs.append(
                np.array(
                    [outputs.get(node_id, 0.0) for node_id in genome.output_ids],
                    dtype=np.float32,
                )
            )

        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=False,
            mean_abs_delta_w=0.0,
            max_abs_delta_w=0.0,
            clamp_hit_rate=0.0,
            plasticity_active_fraction=0.0,
            mean_abs_decay_term=0.0,
            max_abs_decay_term=0.0,
            decay_effect_ratio=0.0,
            decay_near_zero_fraction=0.0,
            mean_abs_fast_state=(total_abs_fast_state / total_node_updates) if total_node_updates else 0.0,
            mean_abs_slow_state=(total_abs_slow_state / total_node_updates) if total_node_updates else 0.0,
            slow_fast_contribution_ratio=(
                total_abs_slow_contribution / total_abs_fast_state
                if total_abs_fast_state > 1e-9
                else 0.0
            ),
            mean_abs_fast_state_during_store=_role_mean(role_totals, "store", index=0),
            mean_abs_slow_state_during_store=_role_mean(role_totals, "store", index=1),
            mean_abs_fast_state_during_query=_role_mean(role_totals, "query", index=0),
            mean_abs_slow_state_during_query=_role_mean(role_totals, "query", index=1),
            mean_abs_fast_state_during_distractor=_role_mean(role_totals, "distractor", index=0),
            mean_abs_slow_state_during_distractor=_role_mean(role_totals, "distractor", index=1),
        )
        return np.vstack(sequence_outputs)

    def last_episode_metrics(self) -> PlasticityEpisodeMetrics:
        return self._last_metrics


class StatefulV2GatedNetworkExecutor(StatefulNetworkExecutor):
    def __init__(self, activation_steps: int, *, slow_write_scale: float = 1.0) -> None:
        super().__init__(activation_steps=activation_steps)
        self.slow_write_scale = float(slow_write_scale)
        self._last_metrics = PlasticityEpisodeMetrics(plasticity_enabled=False, mean_abs_delta_w=0.0, max_abs_delta_w=0.0, clamp_hit_rate=0.0, plasticity_active_fraction=0.0, mean_abs_decay_term=0.0, max_abs_decay_term=0.0, decay_effect_ratio=0.0, decay_near_zero_fraction=0.0)

    def run_sequence(
        self,
        genome: GenomeModel,
        input_sequence: Sequence[Sequence[float]],
        *,
        step_roles: Sequence[str] | None = None,
    ) -> np.ndarray:
        incoming_by_target = _incoming_connections_by_target(genome)
        fast_memory = {node.node_id: 0.0 for node in genome.nodes if not node.is_input}
        slow_memory = {node.node_id: 0.0 for node in genome.nodes if not node.is_input}
        outputs = {node.node_id: 0.0 for node in genome.nodes}
        sequence_outputs: list[np.ndarray] = []
        total_abs_fast_state = 0.0
        total_abs_slow_state = 0.0
        total_node_updates = 0
        gate_values: list[float] = []
        role_totals: dict[str, tuple[float, float, float, int]] = {}

        for step_index, inputs in enumerate(input_sequence):
            step_role = _step_role_at(step_roles, step_index)
            input_map = {node_id: float(value) for node_id, value in zip(genome.input_ids, inputs, strict=True)}
            outputs.update(input_map)
            for _ in range(self.activation_steps):
                previous_outputs = dict(outputs)
                next_outputs = dict(input_map)
                for node in genome.nodes:
                    if node.is_input:
                        continue
                    incoming_sum = 0.0
                    for conn in incoming_by_target.get(node.node_id, ()):
                        incoming_sum += conn.weight * previous_outputs.get(conn.in_id, 0.0)
                    x_input_norm = incoming_sum / (1.0 + abs(incoming_sum))
                    gate = 1.0 / (1.0 + math.exp(-(node.alpha * x_input_norm + node.slow_input_gain)))
                    write_factor = 1.0 - gate
                    new_fast_memory = (gate * fast_memory[node.node_id]) + (write_factor * (incoming_sum + node.bias))
                    new_slow_memory = (
                        clamp_alpha(node.alpha_slow) * slow_memory[node.node_id]
                        + (self.slow_write_scale * write_factor * (incoming_sum + node.slow_output_gain))
                    )
                    fast_memory[node.node_id] = new_fast_memory
                    slow_memory[node.node_id] = new_slow_memory
                    next_outputs[node.node_id] = math.tanh(new_fast_memory + new_slow_memory)
                    total_abs_fast_state += abs(new_fast_memory)
                    total_abs_slow_state += abs(new_slow_memory)
                    total_node_updates += 1
                    gate_values.append(gate)
                    if step_role in {"store", "query", "distractor"}:
                        gate_total, fast_total, slow_total, count = role_totals.get(step_role, (0.0, 0.0, 0.0, 0))
                        role_totals[step_role] = (gate_total + gate, fast_total + abs(new_fast_memory), slow_total + abs(new_slow_memory), count + 1)
                outputs = next_outputs
            sequence_outputs.append(np.array([outputs.get(node_id, 0.0) for node_id in genome.output_ids], dtype=np.float32))

        gate_at_store = _role_mean_4(role_totals, "store", index=0)
        gate_at_distractor = _role_mean_4(role_totals, "distractor", index=0)
        gate_at_query = _role_mean_4(role_totals, "query", index=0)
        fast_at_query = _role_mean_4(role_totals, "query", index=1)
        slow_at_query = _role_mean_4(role_totals, "query", index=2)
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=False,
            mean_abs_delta_w=0.0,
            max_abs_delta_w=0.0,
            clamp_hit_rate=0.0,
            plasticity_active_fraction=0.0,
            mean_abs_decay_term=0.0,
            max_abs_decay_term=0.0,
            decay_effect_ratio=0.0,
            decay_near_zero_fraction=0.0,
            mean_abs_fast_state=(total_abs_fast_state / total_node_updates) if total_node_updates else 0.0,
            mean_abs_slow_state=(total_abs_slow_state / total_node_updates) if total_node_updates else 0.0,
            slow_fast_contribution_ratio=(total_abs_slow_state / total_abs_fast_state) if total_abs_fast_state > 1e-9 else 0.0,
            mean_abs_fast_state_during_store=_role_mean_4(role_totals, "store", index=1),
            mean_abs_slow_state_during_store=_role_mean_4(role_totals, "store", index=2),
            mean_abs_fast_state_during_query=fast_at_query,
            mean_abs_slow_state_during_query=slow_at_query,
            mean_abs_fast_state_during_distractor=_role_mean_4(role_totals, "distractor", index=1),
            mean_abs_slow_state_during_distractor=_role_mean_4(role_totals, "distractor", index=2),
            gate_mean=float(np.mean(gate_values)) if gate_values else 0.0,
            gate_variance=float(np.var(gate_values)) if gate_values else 0.0,
            gate_at_store=gate_at_store,
            gate_at_distractor=gate_at_distractor,
            gate_at_query=gate_at_query,
            gate_selectivity=abs(gate_at_store - gate_at_distractor),
            gate_store_minus_query=(gate_at_store - gate_at_query),
            gate_query_minus_distractor=(gate_at_query - gate_at_distractor),
            gate_role_contrast=abs(gate_at_store - gate_at_distractor) + abs(gate_at_query - gate_at_distractor),
            slow_state_at_query=slow_at_query,
            fast_state_at_query=fast_at_query,
        )
        return np.vstack(sequence_outputs)

    def last_episode_metrics(self) -> PlasticityEpisodeMetrics:
        return self._last_metrics


class ContentGatedNetworkExecutor(StatefulNetworkExecutor):
    def __init__(self, activation_steps: int) -> None:
        super().__init__(activation_steps=activation_steps)
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=False,
            mean_abs_delta_w=0.0,
            max_abs_delta_w=0.0,
            clamp_hit_rate=0.0,
            plasticity_active_fraction=0.0,
            mean_abs_decay_term=0.0,
            max_abs_decay_term=0.0,
            decay_effect_ratio=0.0,
            decay_near_zero_fraction=0.0,
        )

    def run_sequence(
        self,
        genome: GenomeModel,
        input_sequence: Sequence[Sequence[float]],
        *,
        step_roles: Sequence[str] | None = None,
    ) -> np.ndarray:
        incoming_by_target = _incoming_connections_by_target(genome)
        memory = {node.node_id: 0.0 for node in genome.nodes if not node.is_input}
        outputs = {node.node_id: 0.0 for node in genome.nodes}
        sequence_outputs: list[np.ndarray] = []
        match_values: list[float] = []
        role_totals: dict[str, tuple[float, float, int]] = {}
        total_abs_state = 0.0
        total_updates = 0

        for step_index, inputs in enumerate(input_sequence):
            step_role = _step_role_at(step_roles, step_index)
            input_map = {node_id: float(value) for node_id, value in zip(genome.input_ids, inputs, strict=True)}
            outputs.update(input_map)
            for _ in range(self.activation_steps):
                previous_outputs = dict(outputs)
                next_outputs = dict(input_map)
                for node in genome.nodes:
                    if node.is_input:
                        continue
                    incoming_sum = 0.0
                    for conn in incoming_by_target.get(node.node_id, ()):
                        incoming_sum += conn.weight * previous_outputs.get(conn.in_id, 0.0)
                    prev_state = memory[node.node_id]
                    pre_t = incoming_sum + node.bias
                    key_t = math.tanh((node.content_w_key * pre_t) + node.content_b_key)
                    query_t = math.tanh((node.content_w_query * prev_state) + node.content_b_query)
                    match_t = 1.0 / (1.0 + math.exp(-((node.content_temperature * key_t * query_t) + node.content_b_match)))
                    new_state = (clamp_alpha(node.alpha) * prev_state) + (match_t * pre_t)
                    node_output = math.tanh((match_t * pre_t) + ((1.0 - match_t) * prev_state))
                    memory[node.node_id] = new_state
                    next_outputs[node.node_id] = node_output
                    match_values.append(match_t)
                    total_abs_state += abs(new_state)
                    total_updates += 1
                    if step_role in {"store", "query", "distractor"}:
                        match_total, state_total, count = role_totals.get(step_role, (0.0, 0.0, 0))
                        role_totals[step_role] = (match_total + match_t, state_total + abs(new_state), count + 1)
                outputs = next_outputs
            sequence_outputs.append(np.array([outputs.get(node_id, 0.0) for node_id in genome.output_ids], dtype=np.float32))

        match_at_store = _role_mean(role_totals, "store", index=0)
        match_at_query = _role_mean(role_totals, "query", index=0)
        match_at_distractor = _role_mean(role_totals, "distractor", index=0)
        state_store = _role_mean(role_totals, "store", index=1)
        state_query = _role_mean(role_totals, "query", index=1)
        denom = abs(state_store) + abs(state_query)
        alignment = 1.0 - (abs(state_store - state_query) / denom) if denom > 1e-9 else 0.0
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=False,
            mean_abs_delta_w=0.0,
            max_abs_delta_w=0.0,
            clamp_hit_rate=0.0,
            plasticity_active_fraction=0.0,
            mean_abs_decay_term=0.0,
            max_abs_decay_term=0.0,
            decay_effect_ratio=0.0,
            decay_near_zero_fraction=0.0,
            mean_abs_fast_state=(total_abs_state / total_updates) if total_updates else 0.0,
            match_mean=float(np.mean(match_values)) if match_values else 0.0,
            match_variance=float(np.var(match_values)) if match_values else 0.0,
            match_at_store=match_at_store,
            match_at_distractor=match_at_distractor,
            match_at_query=match_at_query,
            match_selectivity=abs(match_at_store - match_at_distractor),
            query_match_score=match_at_query,
            state_query_alignment=max(0.0, min(1.0, alignment)),
            content_retention_gap=abs(state_store - state_query),
        )
        return np.vstack(sequence_outputs)

    def last_episode_metrics(self) -> PlasticityEpisodeMetrics:
        return self._last_metrics


class StatefulV3KVNetworkExecutor(StatefulNetworkExecutor):
    def __init__(self, activation_steps: int) -> None:
        super().__init__(activation_steps=activation_steps)
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=False,
            mean_abs_delta_w=0.0,
            max_abs_delta_w=0.0,
            clamp_hit_rate=0.0,
            plasticity_active_fraction=0.0,
            mean_abs_decay_term=0.0,
            max_abs_decay_term=0.0,
            decay_effect_ratio=0.0,
            decay_near_zero_fraction=0.0,
        )

    def run_sequence(
        self,
        genome: GenomeModel,
        input_sequence: Sequence[Sequence[float]],
        *,
        step_roles: Sequence[str] | None = None,
    ) -> np.ndarray:
        incoming_by_target = _incoming_connections_by_target(genome)
        key_state = {node.node_id: 0.0 for node in genome.nodes if not node.is_input}
        value_state = {node.node_id: 0.0 for node in genome.nodes if not node.is_input}
        outputs = {node.node_id: 0.0 for node in genome.nodes}
        sequence_outputs: list[np.ndarray] = []
        total_abs_key = 0.0
        total_abs_value = 0.0
        total_abs_sep = 0.0
        total_updates = 0
        query_alignment_vals: list[float] = []
        query_read_vals: list[float] = []
        store_coupling_vals: list[float] = []
        distractor_write_vals: list[float] = []
        write_gate_store_vals: list[float] = []
        write_gate_query_vals: list[float] = []
        readout_query_vals: list[float] = []
        readout_distractor_vals: list[float] = []
        match_vals: list[float] = []
        match_query_vals: list[float] = []
        key_state_query_vals: list[float] = []
        value_state_query_vals: list[float] = []
        role_totals: dict[str, tuple[float, float, int]] = {}

        for step_index, inputs in enumerate(input_sequence):
            step_role = _step_role_at(step_roles, step_index)
            input_map = {node_id: float(value) for node_id, value in zip(genome.input_ids, inputs, strict=True)}
            outputs.update(input_map)
            store_signal = float(inputs[0]) if len(inputs) > 0 else 0.0
            query_signal = float(inputs[1]) if len(inputs) > 1 else 0.0
            distractor_signal = float(inputs[2]) if len(inputs) > 2 else 0.0
            for _ in range(self.activation_steps):
                previous_outputs = dict(outputs)
                next_outputs = dict(input_map)
                for node in genome.nodes:
                    if node.is_input:
                        continue
                    summed_input = 0.0
                    for conn in incoming_by_target.get(node.node_id, ()):
                        summed_input += conn.weight * previous_outputs.get(conn.in_id, 0.0)
                    input_norm = summed_input / (1.0 + abs(summed_input))
                    write_gate = 1.0 / (
                        1.0
                        + math.exp(
                            -(
                                (node.content_w_key * input_norm)
                                + (node.content_w_query * store_signal)
                                - (node.content_b_query * distractor_signal)
                                + node.content_b_key
                            )
                        )
                    )
                    key_input = summed_input + node.bias
                    value_input = summed_input + node.slow_output_gain
                    new_key = (clamp_alpha(node.alpha) * key_state[node.node_id]) + (write_gate * key_input)
                    new_value = (clamp_alpha(node.alpha_slow) * value_state[node.node_id]) + (write_gate * value_input)
                    match_t = 1.0 / (
                        1.0
                        + math.exp(
                            -(
                                (node.content_temperature * query_signal * (new_key / (1.0 + abs(new_key))))
                                + node.content_b_match
                            )
                        )
                    )
                    readout_t = match_t * new_value
                    next_outputs[node.node_id] = math.tanh(summed_input + readout_t)
                    key_state[node.node_id] = new_key
                    value_state[node.node_id] = new_value
                    total_abs_key += abs(new_key)
                    total_abs_value += abs(new_value)
                    total_abs_sep += abs(new_key - new_value)
                    total_updates += 1
                    if step_role in {"store", "query", "distractor"}:
                        key_total, value_total, count = role_totals.get(step_role, (0.0, 0.0, 0))
                        role_totals[step_role] = (key_total + abs(new_key), value_total + abs(new_value), count + 1)
                    match_vals.append(match_t)
                    if step_role == "query":
                        query_alignment_vals.append(abs(query_signal * new_key))
                        query_read_vals.append(abs(readout_t))
                        readout_query_vals.append(abs(readout_t))
                        match_query_vals.append(match_t)
                        write_gate_query_vals.append(write_gate)
                        key_state_query_vals.append(abs(new_key))
                        value_state_query_vals.append(abs(new_value))
                    if step_role == "store":
                        store_coupling_vals.append(abs(new_key * new_value))
                        write_gate_store_vals.append(write_gate)
                    if step_role == "distractor":
                        distractor_write_vals.append(write_gate)
                        readout_distractor_vals.append(abs(readout_t))
                outputs = next_outputs
            sequence_outputs.append(np.array([outputs.get(node_id, 0.0) for node_id in genome.output_ids], dtype=np.float32))

        readout_query = float(np.mean(readout_query_vals)) if readout_query_vals else 0.0
        readout_distractor = float(np.mean(readout_distractor_vals)) if readout_distractor_vals else 0.0
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=False,
            mean_abs_delta_w=0.0,
            max_abs_delta_w=0.0,
            clamp_hit_rate=0.0,
            plasticity_active_fraction=0.0,
            mean_abs_decay_term=0.0,
            max_abs_decay_term=0.0,
            decay_effect_ratio=0.0,
            decay_near_zero_fraction=0.0,
            mean_key_state=(total_abs_key / total_updates) if total_updates else 0.0,
            mean_value_state=(total_abs_value / total_updates) if total_updates else 0.0,
            key_value_separation=(total_abs_sep / total_updates) if total_updates else 0.0,
            query_key_alignment=float(np.mean(query_alignment_vals)) if query_alignment_vals else 0.0,
            query_value_read_strength=float(np.mean(query_read_vals)) if query_read_vals else 0.0,
            store_key_value_coupling=float(np.mean(store_coupling_vals)) if store_coupling_vals else 0.0,
            distractor_write_leak=float(np.mean(distractor_write_vals)) if distractor_write_vals else 0.0,
            readout_selectivity=abs(readout_query - readout_distractor),
            mean_key_state_during_store=_role_mean(role_totals, "store", index=0),
            mean_value_state_during_store=_role_mean(role_totals, "store", index=1),
            mean_key_state_during_query=_role_mean(role_totals, "query", index=0),
            mean_value_state_during_query=_role_mean(role_totals, "query", index=1),
            write_gate_at_store=float(np.mean(write_gate_store_vals)) if write_gate_store_vals else 0.0,
            write_gate_at_distractor=float(np.mean(distractor_write_vals)) if distractor_write_vals else 0.0,
            write_gate_at_query=float(np.mean(write_gate_query_vals)) if write_gate_query_vals else 0.0,
            store_vs_distractor_write_gap=(
                float(np.mean(write_gate_store_vals)) - float(np.mean(distractor_write_vals))
                if write_gate_store_vals and distractor_write_vals
                else 0.0
            ),
            mean_match_signal=float(np.mean(match_vals)) if match_vals else 0.0,
            value_state_at_query=float(np.mean(value_state_query_vals)) if value_state_query_vals else 0.0,
            key_state_at_query=float(np.mean(key_state_query_vals)) if key_state_query_vals else 0.0,
            match_at_query=float(np.mean(match_query_vals)) if match_query_vals else 0.0,
        )
        return np.vstack(sequence_outputs)

    def last_episode_metrics(self) -> PlasticityEpisodeMetrics:
        return self._last_metrics


class StatefulV4SlotsNetworkExecutor(StatefulNetworkExecutor):
    def __init__(self, activation_steps: int) -> None:
        super().__init__(activation_steps=activation_steps)
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=False,
            mean_abs_delta_w=0.0,
            max_abs_delta_w=0.0,
            clamp_hit_rate=0.0,
            plasticity_active_fraction=0.0,
            mean_abs_decay_term=0.0,
            max_abs_decay_term=0.0,
            decay_effect_ratio=0.0,
            decay_near_zero_fraction=0.0,
        )

    def run_sequence(
        self,
        genome: GenomeModel,
        input_sequence: Sequence[Sequence[float]],
        *,
        step_roles: Sequence[str] | None = None,
    ) -> np.ndarray:
        incoming_by_target = _incoming_connections_by_target(genome)
        slot_count = 2
        slot_keys = {node.node_id: [0.0 for _ in range(slot_count)] for node in genome.nodes if not node.is_input}
        slot_values = {node.node_id: [0.0 for _ in range(slot_count)] for node in genome.nodes if not node.is_input}
        outputs = {node.node_id: 0.0 for node in genome.nodes}
        sequence_outputs: list[np.ndarray] = []
        write_focus_vals: list[float] = []
        query_focus_vals: list[float] = []
        readout_selectivity_vals: list[float] = []
        query_slot_match_max_vals: list[float] = []
        distractor_leak_vals: list[float] = []
        slot_key_sep_vals: list[float] = []
        slot_value_sep_vals: list[float] = []
        slot_hits = [0, 0]

        for step_index, inputs in enumerate(input_sequence):
            step_role = _step_role_at(step_roles, step_index)
            input_map = {node_id: float(value) for node_id, value in zip(genome.input_ids, inputs, strict=True)}
            outputs.update(input_map)
            query_signal = float(inputs[1]) if len(inputs) > 1 else 0.0
            store_signal = float(inputs[0]) if len(inputs) > 0 else 0.0
            distractor_signal = float(inputs[2]) if len(inputs) > 2 else 0.0
            key_bits = np.asarray([float(inputs[4]), float(inputs[5]), float(inputs[6])] if len(inputs) > 6 else [0.0, 0.0, 0.0])
            key_index = int(np.argmax(key_bits)) if float(np.sum(key_bits)) > 0.0 else 0
            active_slot = int(key_index % slot_count)
            slot_hits[active_slot] += 1
            for _ in range(self.activation_steps):
                previous_outputs = dict(outputs)
                next_outputs = dict(input_map)
                for node in genome.nodes:
                    if node.is_input:
                        continue
                    summed_input = 0.0
                    for conn in incoming_by_target.get(node.node_id, ()):
                        summed_input += conn.weight * previous_outputs.get(conn.in_id, 0.0)
                    input_norm = summed_input / (1.0 + abs(summed_input))
                    write_gate = 1.0 / (1.0 + math.exp(-((node.content_w_key * input_norm) + (node.content_b_key * store_signal))))
                    value_gate = 1.0 / (1.0 + math.exp(-((node.content_w_query * input_norm) + node.content_b_query)))
                    for slot_index in range(slot_count):
                        slot_mask = 1.0 if slot_index == active_slot else 0.0
                        decay = clamp_alpha(node.alpha if slot_index == 0 else node.alpha_slow)
                        slot_keys[node.node_id][slot_index] = (
                            decay * slot_keys[node.node_id][slot_index]
                            + (write_gate * store_signal * slot_mask * input_norm)
                        )
                        slot_values[node.node_id][slot_index] = (
                            decay * slot_values[node.node_id][slot_index]
                            + (value_gate * store_signal * slot_mask * (summed_input + node.slow_output_gain))
                        )
                    query_key = query_signal * (0.5 + abs(input_norm))
                    match_scale = max(0.5, float(node.content_temperature))
                    # NOTE: ``content_b_match`` was previously added as a per-logit
                    # constant here, but a constant offset on every logit cancels
                    # exactly against the subsequent ``logits - max(logits)``
                    # normalisation step and therefore had zero effect on
                    # ``weights``. Removed to stop evolution from spending search
                    # budget on a phantom parameter in this code path.
                    logits = np.asarray(
                        [
                            match_scale * query_key * slot_keys[node.node_id][slot_idx]
                            for slot_idx in range(slot_count)
                        ],
                        dtype=np.float64,
                    )
                    logits = logits - float(np.max(logits))
                    probs = np.exp(logits)
                    denom = float(np.sum(probs))
                    weights = probs / denom if denom > 0.0 else np.asarray([0.5, 0.5], dtype=np.float64)
                    slot_values_arr = np.asarray(slot_values[node.node_id], dtype=np.float64)
                    primary_readout = float(np.sum(weights * slot_values_arr))
                    residual_readout = 0.5 * float(np.mean(slot_values_arr))
                    dominant_match = float(np.max(weights))
                    focus_gate = 1.0 / (
                        1.0
                        + math.exp(
                            -(
                                (2.0 * dominant_match)
                                + (0.5 * match_scale)
                                + (0.25 * node.content_b_query)
                            )
                        )
                    )
                    leak_suppress = max(0.2, 1.0 - (0.6 * max(0.0, distractor_signal)))
                    readout = leak_suppress * ((focus_gate * primary_readout) + ((1.0 - focus_gate) * residual_readout))
                    next_outputs[node.node_id] = math.tanh(summed_input + readout)
                    if step_role == "store":
                        write_focus_vals.append(abs(slot_keys[node.node_id][0] - slot_keys[node.node_id][1]))
                    if step_role == "query":
                        query_focus_vals.append(abs(float(weights[0]) - float(weights[1])))
                        readout_selectivity_vals.append(abs(readout))
                        query_slot_match_max_vals.append(float(np.max(weights)))
                    if step_role == "distractor":
                        distractor_leak_vals.append(abs(readout))
                    slot_key_sep_vals.append(abs(slot_keys[node.node_id][0] - slot_keys[node.node_id][1]))
                    slot_value_sep_vals.append(abs(slot_values[node.node_id][0] - slot_values[node.node_id][1]))
                outputs = next_outputs
            sequence_outputs.append(np.array([outputs.get(node_id, 0.0) for node_id in genome.output_ids], dtype=np.float32))

        total_hits = max(1, slot_hits[0] + slot_hits[1])
        utilization = len([hit for hit in slot_hits if hit > 0]) / slot_count
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=False,
            mean_abs_delta_w=0.0,
            max_abs_delta_w=0.0,
            clamp_hit_rate=0.0,
            plasticity_active_fraction=0.0,
            mean_abs_decay_term=0.0,
            max_abs_decay_term=0.0,
            decay_effect_ratio=0.0,
            decay_near_zero_fraction=0.0,
            slot_key_separation=float(np.mean(slot_key_sep_vals)) if slot_key_sep_vals else 0.0,
            slot_value_separation=float(np.mean(slot_value_sep_vals)) if slot_value_sep_vals else 0.0,
            slot_write_focus=float(np.mean(write_focus_vals)) if write_focus_vals else 0.0,
            slot_query_focus=float(np.mean(query_focus_vals)) if query_focus_vals else 0.0,
            slot_readout_selectivity=float(np.mean(readout_selectivity_vals)) if readout_selectivity_vals else 0.0,
            slot_utilization=max(utilization, min(slot_hits) / total_hits),
            query_slot_match_max=float(np.mean(query_slot_match_max_vals)) if query_slot_match_max_vals else 0.0,
            slot_distractor_leak=float(np.mean(distractor_leak_vals)) if distractor_leak_vals else 0.0,
        )
        return np.vstack(sequence_outputs)

    def last_episode_metrics(self) -> PlasticityEpisodeMetrics:
        return self._last_metrics


class StatefulV5AddressedSlotsNetworkExecutor(StatefulNetworkExecutor):
    def __init__(self, activation_steps: int) -> None:
        super().__init__(activation_steps=activation_steps)
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=False,
            mean_abs_delta_w=0.0,
            max_abs_delta_w=0.0,
            clamp_hit_rate=0.0,
            plasticity_active_fraction=0.0,
            mean_abs_decay_term=0.0,
            max_abs_decay_term=0.0,
            decay_effect_ratio=0.0,
            decay_near_zero_fraction=0.0,
        )

    def run_sequence(
        self,
        genome: GenomeModel,
        input_sequence: Sequence[Sequence[float]],
        *,
        step_roles: Sequence[str] | None = None,
    ) -> np.ndarray:
        incoming_by_target = _incoming_connections_by_target(genome)
        slot_count = 2
        slot_keys = {node.node_id: [0.0 for _ in range(slot_count)] for node in genome.nodes if not node.is_input}
        slot_values = {node.node_id: [0.0 for _ in range(slot_count)] for node in genome.nodes if not node.is_input}
        outputs = {node.node_id: 0.0 for node in genome.nodes}
        sequence_outputs: list[np.ndarray] = []
        write_focus_vals: list[float] = []
        read_focus_vals: list[float] = []
        gap_vals: list[float] = []
        write_alignment_vals: list[float] = []
        read_alignment_vals: list[float] = []
        consistency_vals: list[float] = []
        leak_vals: list[float] = []
        concentration_vals: list[float] = []

        for step_index, inputs in enumerate(input_sequence):
            step_role = _step_role_at(step_roles, step_index)
            input_map = {node_id: float(value) for node_id, value in zip(genome.input_ids, inputs, strict=True)}
            outputs.update(input_map)
            store_signal = float(inputs[0]) if len(inputs) > 0 else 0.0
            query_signal = float(inputs[1]) if len(inputs) > 1 else 0.0
            distractor_signal = float(inputs[2]) if len(inputs) > 2 else 0.0
            key_bits = np.asarray([float(inputs[4]), float(inputs[5]), float(inputs[6])] if len(inputs) > 6 else [0.0, 0.0, 0.0])
            key_strength = float(np.max(key_bits)) if key_bits.size else 0.0
            for _ in range(self.activation_steps):
                previous_outputs = dict(outputs)
                next_outputs = dict(input_map)
                for node in genome.nodes:
                    if node.is_input:
                        continue
                    summed_input = 0.0
                    for conn in incoming_by_target.get(node.node_id, ()):
                        summed_input += conn.weight * previous_outputs.get(conn.in_id, 0.0)
                    input_norm = summed_input / (1.0 + abs(summed_input))
                    write_signal = (node.content_w_key * input_norm) + (node.content_b_key * store_signal)
                    read_signal = (node.content_w_query * input_norm) + (node.content_b_query * query_signal)
                    # V13b: deterministic, small asymmetry between slots so evolution can leave
                    # the symmetric addressing fixed point.
                    write_scale = max(0.6, abs(float(node.content_temperature)))
                    read_scale = max(0.6, abs(float(node.content_temperature)) + (0.3 * abs(float(node.content_b_match))) + 0.1)
                    slot_write_keys = (1.12 + node.content_w_key, -0.88 - node.content_w_key)
                    slot_read_keys = (0.9 + node.content_w_query, -1.1 - node.content_w_query)
                    slot_write_bias = (0.08, -0.08)
                    slot_read_bias = (-0.05, 0.05)
                    write_bias = node.content_b_key + (0.05 * store_signal)
                    read_bias = node.content_b_match + (0.05 * query_signal)
                    write_addr = np.asarray(
                        [
                            1.0
                            / (
                                1.0
                                + math.exp(
                                    -(
                                        (write_scale * write_signal * slot_write_keys[s])
                                        + write_bias
                                        + slot_write_bias[s]
                                    )
                                )
                            )
                            for s in range(slot_count)
                        ],
                        dtype=np.float64,
                    )
                    read_addr_raw = np.asarray(
                        [
                            1.0
                            / (
                                1.0
                                + math.exp(
                                    -(
                                        (read_scale * read_signal * slot_read_keys[s])
                                        + read_bias
                                        + slot_read_bias[s]
                                    )
                                )
                            )
                            for s in range(slot_count)
                        ],
                        dtype=np.float64,
                    )
                    read_den = float(np.sum(read_addr_raw))
                    read_addr = (read_addr_raw / read_den) if read_den > 0.0 else np.asarray([0.5, 0.5], dtype=np.float64)
                    key_input = store_signal * input_norm
                    value_input = store_signal * (summed_input + node.slow_output_gain)
                    for s in range(slot_count):
                        slot_keys[node.node_id][s] = (clamp_alpha(node.alpha) * slot_keys[node.node_id][s]) + (write_addr[s] * key_input)
                        slot_values[node.node_id][s] = (clamp_alpha(node.alpha_slow) * slot_values[node.node_id][s]) + (write_addr[s] * value_input)
                    slot_values_arr = np.asarray(slot_values[node.node_id], dtype=np.float64)
                    readout = float(np.sum(read_addr * slot_values_arr))
                    readout = (1.0 - 0.5 * max(0.0, distractor_signal)) * readout
                    next_outputs[node.node_id] = math.tanh(summed_input + readout)
                    write_focus = abs(float(write_addr[0]) - float(write_addr[1]))
                    read_focus = abs(float(read_addr[0]) - float(read_addr[1]))
                    write_focus_vals.append(write_focus)
                    read_focus_vals.append(read_focus)
                    gap_vals.append(read_focus - write_focus)
                    consistency_vals.append(1.0 - abs(write_focus - read_focus))
                    concentration_vals.append(float(np.max(read_addr)))
                    if step_role == "store":
                        write_alignment_vals.append(write_focus * key_strength)
                    if step_role == "query":
                        read_alignment_vals.append(read_focus * key_strength)
                    if step_role == "distractor":
                        leak_vals.append(abs(readout))
                outputs = next_outputs
            sequence_outputs.append(np.array([outputs.get(node_id, 0.0) for node_id in genome.output_ids], dtype=np.float32))

        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=False,
            mean_abs_delta_w=0.0,
            max_abs_delta_w=0.0,
            clamp_hit_rate=0.0,
            plasticity_active_fraction=0.0,
            mean_abs_decay_term=0.0,
            max_abs_decay_term=0.0,
            decay_effect_ratio=0.0,
            decay_near_zero_fraction=0.0,
            mean_write_address_focus=float(np.mean(write_focus_vals)) if write_focus_vals else 0.0,
            mean_read_address_focus=float(np.mean(read_focus_vals)) if read_focus_vals else 0.0,
            write_read_address_gap=float(np.mean(gap_vals)) if gap_vals else 0.0,
            slot_write_specialization=float(np.std(write_focus_vals)) if write_focus_vals else 0.0,
            slot_read_specialization=float(np.std(read_focus_vals)) if read_focus_vals else 0.0,
            address_consistency=float(np.mean(consistency_vals)) if consistency_vals else 0.0,
            query_read_alignment=float(np.mean(read_alignment_vals)) if read_alignment_vals else 0.0,
            store_write_alignment=float(np.mean(write_alignment_vals)) if write_alignment_vals else 0.0,
            distractor_write_leak=float(np.mean(leak_vals)) if leak_vals else 0.0,
            readout_address_concentration=float(np.mean(concentration_vals)) if concentration_vals else 0.0,
        )
        return np.vstack(sequence_outputs)

    def last_episode_metrics(self) -> PlasticityEpisodeMetrics:
        return self._last_metrics


class StatefulV6DeltaMemoryNetworkExecutor(StatefulNetworkExecutor):
    def __init__(
        self,
        activation_steps: int,
        *,
        sub_variant: str = "stateful_v6_delta_memory",
    ) -> None:
        super().__init__(activation_steps=activation_steps)
        self._sub_variant = sub_variant
        # v16a and v16b both use the architectural source asymmetry.
        self._source_asymmetry = sub_variant in (
            "stateful_v6_delta_memory_v16a",
            "stateful_v6_delta_memory_v16b",
        )
        # v16b additionally zeroes the key<->query crossing gains so the
        # post-hoc machinery cannot pull k_t and q_t back together.
        self._cross_kq_enabled = sub_variant != "stateful_v6_delta_memory_v16b"
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=False,
            mean_abs_delta_w=0.0,
            max_abs_delta_w=0.0,
            clamp_hit_rate=0.0,
            plasticity_active_fraction=0.0,
            mean_abs_decay_term=0.0,
            max_abs_decay_term=0.0,
            decay_effect_ratio=0.0,
            decay_near_zero_fraction=0.0,
        )

    def run_sequence(
        self,
        genome: GenomeModel,
        input_sequence: Sequence[Sequence[float]],
        *,
        step_roles: Sequence[str] | None = None,
        trace_sink: list[dict[str, object]] | None = None,
    ) -> np.ndarray:
        # ``trace_sink`` is a passive diagnostic hook. When ``None`` (the default
        # and the only value used on the production evaluation path), this
        # method behaves exactly as before: no extra state, no extra allocation,
        # no change to the math. When a list is supplied the executor appends
        # one per-step-per-node dict describing the internal state of the
        # delta-memory read/write path; this is consumed by
        # ``src/analysis/retrieval_trace.py``.
        incoming_by_target = _incoming_connections_by_target(genome)
        d_key = 8
        d_value = 8
        memory_decay = 0.97
        delta_clip_norm = 2.0
        update_clip_frob = 1.0
        read_clip_norm = 2.0
        position_axis = np.linspace(-1.0, 1.0, d_key, dtype=np.float64)
        alt_sign = np.where(np.arange(d_key) % 2 == 0, 1.0, -1.0).astype(np.float64)
        center_peak = 1.0 - np.abs(position_axis)
        centered_square = (position_axis**2) - float(np.mean(position_axis**2))
        edge_peak = np.abs(position_axis)
        harmonic_sin = np.sin(np.pi * position_axis)
        harmonic_cos = np.cos(np.pi * position_axis)
        memory_state = {
            node.node_id: np.zeros((d_value, d_key), dtype=np.float64) for node in genome.nodes if not node.is_input
        }
        outputs = {node.node_id: 0.0 for node in genome.nodes}
        sequence_outputs: list[np.ndarray] = []
        beta_vals: list[float] = []
        beta_store_vals: list[float] = []
        beta_distractor_vals: list[float] = []
        beta_query_vals: list[float] = []
        key_norm_vals: list[float] = []
        query_norm_vals: list[float] = []
        value_norm_vals: list[float] = []
        memory_norm_vals: list[float] = []
        query_alignment_vals: list[float] = []
        store_update_vals: list[float] = []
        distractor_update_vals: list[float] = []
        query_update_vals: list[float] = []
        delta_correction_vals: list[float] = []
        memory_read_vals: list[float] = []
        readout_selectivity_vals: list[float] = []
        novelty_ratio_vals: list[float] = []
        readout_contrast_vals: list[float] = []
        key_query_cosine_vals: list[float] = []
        key_query_cosine_query_vals: list[float] = []
        key_variance_vals: list[float] = []
        query_variance_vals: list[float] = []
        key_query_projection_vals: list[float] = []
        query_decoupling_vals: list[float] = []

        for step_index, inputs in enumerate(input_sequence):
            step_role = _step_role_at(step_roles, step_index)
            input_map = {node_id: float(value) for node_id, value in zip(genome.input_ids, inputs, strict=True)}
            outputs.update(input_map)
            step_capture: dict[int, dict[str, object]] | None = (
                {} if trace_sink is not None else None
            )
            for _ in range(self.activation_steps):
                previous_outputs = dict(outputs)
                next_outputs = dict(input_map)
                for node in genome.nodes:
                    if node.is_input:
                        continue
                    summed_input = 0.0
                    for conn in incoming_by_target.get(node.node_id, ()):
                        summed_input += conn.weight * previous_outputs.get(conn.in_id, 0.0)
                    x_t = summed_input + node.bias
                    x_norm = x_t / (1.0 + abs(x_t))
                    x_abs = abs(x_norm)
                    if self._source_asymmetry:
                        # v16a/v16b: disjoint source signals. Key lives on
                        # the signed axis (x_norm), query on the unsigned
                        # axis (x_abs). Evolution no longer has to discover
                        # the decoupling — it is structurally enforced.
                        key_seed = (node.content_w_key * x_norm) + (0.25 * node.content_b_key)
                        query_seed = (node.content_w_query * x_abs) + (
                            0.25 * node.content_b_query * (1.0 - x_abs)
                        )
                    else:
                        key_seed = (node.content_w_key * x_norm) + (node.content_b_key * x_abs)
                        query_seed = (node.content_w_query * x_norm) + (node.content_b_query * x_abs)
                    value_seed = summed_input + node.slow_output_gain
                    beta_logit = (node.content_temperature * x_norm) + node.content_b_match
                    beta_base = 1.0 / (1.0 + math.exp(-beta_logit))
                    store_signal = 1.0 / (
                        1.0
                        + math.exp(
                            -(
                                1.5 * (abs(key_seed) - abs(query_seed))
                                + (0.5 * node.content_b_key)
                            )
                        )
                    )
                    query_signal = 1.0 / (
                        1.0
                        + math.exp(
                            -(
                                1.5 * (abs(query_seed) - abs(key_seed))
                                + (0.5 * node.content_b_query)
                            )
                        )
                    )
                    if self._source_asymmetry:
                        # v16a/v16b: disjoint positional bases. Key uses the
                        # low-frequency / signed set {center_peak,
                        # position_axis, harmonic_cos}; query uses the
                        # high-frequency / unsigned / phase-shifted set
                        # {edge_peak, alt_sign, centered_square, harmonic_sin}.
                        # No cross-pollution: key_core does not reference
                        # query_seed and vice versa.
                        key_core = (
                            key_seed * (0.95 + (0.65 * center_peak))
                            + (0.6 * node.content_b_key * position_axis)
                            + (0.35 * node.content_w_key * harmonic_cos)
                        )
                        k_raw = (
                            1.0
                            + (0.55 * np.tanh(key_core))
                            + (0.28 * np.tanh(key_core * center_peak))
                            + (0.15 * np.tanh((key_seed + node.content_b_key) * harmonic_cos))
                        )
                        query_core = (
                            query_seed * (0.95 + (0.70 * edge_peak))
                            - (0.55 * node.content_b_query * alt_sign)
                            + (0.40 * node.content_w_query * centered_square)
                            + (0.25 * query_seed * harmonic_sin)
                        )
                        q_raw = (
                            1.0
                            + (0.55 * np.tanh(query_core))
                            + (0.30 * np.tanh(query_core * edge_peak))
                            + (0.15 * np.tanh((query_seed + node.content_b_query) * harmonic_sin))
                        )
                    else:
                        key_core = (
                            key_seed * (0.95 + (0.65 * center_peak))
                            + (0.6 * node.content_b_key * position_axis)
                            + (0.35 * node.content_w_key * alt_sign)
                            + (0.2 * key_seed * harmonic_cos)
                        )
                        key_cross = 0.18 * query_seed * (0.6 * position_axis + 0.4 * alt_sign)
                        k_raw = (
                            1.0
                            + (0.5 * np.tanh(key_core))
                            + (0.3 * np.tanh((key_core * alt_sign) + key_cross))
                            + (0.22 * np.tanh((key_seed - query_seed) * (center_peak - edge_peak)))
                            + (0.12 * np.tanh((key_seed + node.content_b_key) * harmonic_cos))
                        )
                        query_core = (
                            query_seed * (0.95 + (0.7 * edge_peak))
                            - (0.55 * node.content_b_query * position_axis)
                            + (0.4 * node.content_w_query * centered_square)
                            + (0.25 * query_seed * harmonic_sin)
                        )
                        q_raw = (
                            1.0
                            + (0.52 * np.tanh(query_core + (0.25 * alt_sign)))
                            + (0.34 * np.tanh((query_seed - key_seed) * ((1.1 * edge_peak) + (0.4 * alt_sign))))
                            + (0.24 * np.tanh((query_core * position_axis) - (0.25 * key_seed * harmonic_cos)))
                            + (0.14 * np.tanh((query_seed + node.content_b_query) * harmonic_sin))
                        )
                    k_raw = np.maximum(k_raw, 1e-3)
                    q_raw = np.maximum(q_raw, 1e-3)
                    k_t_base = _positive_sum_normalize(k_raw)
                    q_t_base = _positive_sum_normalize(q_raw)
                    key_variance_pre = float(np.var(k_t_base))
                    query_variance_pre = float(np.var(q_t_base))
                    key_query_asym = abs(key_seed) / (abs(key_seed) + abs(query_seed) + 1e-6)
                    key_query_cos_pre = float(
                        np.dot(q_t_base, k_t_base) / (np.linalg.norm(q_t_base) * np.linalg.norm(k_t_base) + 1e-9)
                    )
                    projection_magnitude_pre = abs(
                        float(np.dot(q_t_base - float(np.mean(q_t_base)), k_t_base - float(np.mean(k_t_base))))
                    ) / (float(np.linalg.norm(k_t_base - float(np.mean(k_t_base)))) + 1e-9)
                    key_sharpen_logit = (
                        (0.9 * store_signal)
                        + (0.55 * key_query_asym)
                        + (0.35 * (key_variance_pre / (key_variance_pre + 0.0025)))
                        + (0.2 * (projection_magnitude_pre / (projection_magnitude_pre + 0.1)))
                        - (0.35 * query_signal)
                        - 0.7
                    )
                    key_sharpen_gain = 0.1 * (1.0 + math.tanh(key_sharpen_logit))
                    k_centered_pre = k_t_base - float(np.mean(k_t_base))
                    k_std_pre = math.sqrt(key_variance_pre + 1e-9)
                    q_centered_pre = q_t_base - float(np.mean(q_t_base))
                    q_std_pre = math.sqrt(query_variance_pre + 1e-9)
                    k_peaking = np.tanh(k_centered_pre / (k_std_pre + 1e-6))
                    query_compat_profile = np.tanh(q_centered_pre / (q_std_pre + 1e-6))
                    congruence_logit = (
                        (0.8 * store_signal)
                        + (0.4 * key_query_asym)
                        + (0.25 * (query_variance_pre / (query_variance_pre + 0.0025)))
                        - (0.55 * query_signal)
                        - 0.55
                    )
                    congruence_gain = 0.045 * (1.0 + math.tanh(congruence_logit))
                    if not self._cross_kq_enabled:
                        congruence_gain = 0.0
                    margin_logit = (
                        (1.0 * store_signal)
                        + (0.4 * key_query_asym)
                        + (0.35 * (projection_magnitude_pre / (projection_magnitude_pre + 0.1)))
                        + (0.25 * max(0.0, key_query_cos_pre))
                        - (0.7 * query_signal)
                        - 0.7
                    )
                    margin_gain = 0.028 * (1.0 + math.tanh(margin_logit))
                    store_margin_profile = np.maximum(k_centered_pre, 0.0)
                    store_margin_profile = store_margin_profile / (float(np.sum(store_margin_profile)) + 1e-9)
                    query_deflate_profile = np.maximum(q_centered_pre, 0.0)
                    query_deflate_profile = query_deflate_profile / (float(np.sum(query_deflate_profile)) + 1e-9)
                    k_sharpened = (
                        k_t_base * (1.0 + (key_sharpen_gain * k_peaking))
                        + (congruence_gain * query_compat_profile)
                        + (margin_gain * (store_margin_profile - (0.55 * query_deflate_profile)))
                        + (0.03 * key_sharpen_gain * center_peak)
                    )
                    k_t = _positive_sum_normalize(np.maximum(k_sharpened, 1e-6))
                    k_centered = k_t - float(np.mean(k_t))
                    q_centered = q_t_base - float(np.mean(q_t_base))
                    key_center_norm = float(np.linalg.norm(k_centered))
                    key_center_energy = float(np.dot(k_centered, k_centered)) + 1e-9
                    raw_projection_coeff = float(np.dot(q_centered, k_centered)) / key_center_energy
                    bounded_projection_coeff = 0.4 * math.tanh(raw_projection_coeff / 0.4)
                    projection_magnitude = abs(float(np.dot(q_centered, k_centered))) / (key_center_norm + 1e-9)
                    key_query_cos_base = float(
                        np.dot(q_t_base, k_t) / (np.linalg.norm(q_t_base) * np.linalg.norm(k_t) + 1e-9)
                    )
                    compat_refine_logit = (
                        (0.85 * store_signal)
                        + (0.35 * key_query_asym)
                        + (0.35 * max(0.0, key_query_cos_base))
                        + (0.35 * (projection_magnitude / (projection_magnitude + 0.1)))
                        - (0.45 * query_signal)
                        - 0.65
                    )
                    compat_refine_gain = 0.03 * (1.0 + math.tanh(compat_refine_logit))
                    if not self._cross_kq_enabled:
                        compat_refine_gain = 0.0
                    k_t = _positive_sum_normalize(
                        np.maximum(
                            k_t + (compat_refine_gain * query_compat_profile),
                            1e-6,
                        )
                    )
                    k_centered = k_t - float(np.mean(k_t))
                    key_center_norm = float(np.linalg.norm(k_centered))
                    key_center_energy = float(np.dot(k_centered, k_centered)) + 1e-9
                    raw_projection_coeff = float(np.dot(q_centered, k_centered)) / key_center_energy
                    bounded_projection_coeff = 0.4 * math.tanh(raw_projection_coeff / 0.4)
                    projection_magnitude = abs(float(np.dot(q_centered, k_centered))) / (key_center_norm + 1e-9)
                    projection_signal = abs(bounded_projection_coeff) / (abs(bounded_projection_coeff) + 0.15)
                    query_collapse_signal_pre = 0.003 / (query_variance_pre + 0.003)
                    projection_selectivity_signal = projection_magnitude / (projection_magnitude + 0.12)
                    coupled_role_separation_signal = 0.5 + (
                        0.5
                        * math.tanh(
                            (0.55 * key_query_asym)
                            + (0.45 * projection_selectivity_signal)
                            + (0.35 * max(0.0, key_query_cos_base))
                            + (0.25 * store_signal)
                            - (0.3 * query_signal)
                            - 0.45
                        )
                    )
                    deflation_logit = (
                        (1.2 * max(0.0, key_query_cos_base))
                        + (0.95 * projection_signal)
                        + (0.7 * query_collapse_signal_pre)
                        + (0.35 * projection_selectivity_signal)
                        - 0.85
                    )
                    deflation_gain = 1.0 + (0.45 * (1.0 + math.tanh(deflation_logit)))
                    adaptive_deflation = 0.26 + (
                        0.12
                        * (
                            0.5
                            + 0.5
                            * math.tanh(
                                (0.9 * projection_selectivity_signal)
                                + (0.65 * max(0.0, key_query_cos_base))
                                + (0.45 * query_collapse_signal_pre)
                                - (0.35 * query_signal)
                                - 0.75
                            )
                        )
                    )
                    compatibility_relax = 1.0 - (0.22 * query_signal * projection_selectivity_signal)
                    deflation_vector = (
                        adaptive_deflation
                        * compatibility_relax
                        * deflation_gain
                        * bounded_projection_coeff
                        * k_centered
                    )
                    q_orthogonal_hint = q_centered - (raw_projection_coeff * k_centered)
                    q_orthogonal_norm = float(np.linalg.norm(q_orthogonal_hint))
                    q_orthogonal_term = 0.03 * deflation_gain * (
                        q_orthogonal_hint / (q_orthogonal_norm + 1e-9)
                    )
                    if not self._cross_kq_enabled:
                        deflation_vector = np.zeros_like(k_centered)
                        q_orthogonal_term = np.zeros_like(k_centered)
                    edge_recentered = edge_peak - float(np.mean(edge_peak))
                    edge_gain = 0.02 * deflation_gain * query_collapse_signal_pre * query_signal
                    focus_logit = (
                        (0.9 * query_signal)
                        + (0.5 * projection_selectivity_signal)
                        + (0.35 * (1.0 - query_collapse_signal_pre))
                        - (0.35 * store_signal)
                        - 0.45
                    )
                    focus_gain = 0.012 * (1.0 + math.tanh(focus_logit))
                    q_focus_profile = np.maximum(q_centered, 0.0)
                    q_focus_profile = q_focus_profile / (float(np.sum(q_focus_profile)) + 1e-9)
                    q_decoupled = (
                        q_t_base
                        - deflation_vector
                        + q_orthogonal_term
                        + (edge_gain * edge_recentered)
                        + (focus_gain * q_focus_profile)
                    )
                    q_t = _positive_sum_normalize(np.maximum(q_decoupled, 1e-6))
                    extra_deflation_vector = np.zeros_like(q_t)
                    if step_role == "query" and self._cross_kq_enabled:
                        q_post_centered = q_t - float(np.mean(q_t))
                        post_projection_coeff = float(np.dot(q_post_centered, k_centered)) / key_center_energy
                        bounded_post_projection = 0.2 * math.tanh(post_projection_coeff / 0.2)
                        query_deflation_gain = 0.025 * (
                            0.5
                            + 0.5
                            * math.tanh(
                                (0.65 * projection_selectivity_signal)
                                + (0.4 * max(0.0, key_query_cos_base))
                                + (0.3 * query_collapse_signal_pre)
                                + (0.25 * coupled_role_separation_signal)
                                - (0.45 * query_signal)
                                - 0.65
                            )
                        )
                        extra_deflation_vector = query_deflation_gain * bounded_post_projection * k_centered
                        q_t = _positive_sum_normalize(np.maximum(q_t - extra_deflation_vector, 1e-6))
                    key_variance = float(np.var(k_t))
                    if step_role == "store":
                        key_variance_signal_pre = key_variance / (key_variance + 0.003)
                        store_key_sharpen_logit = (
                            (0.75 * store_signal)
                            + (0.4 * key_query_asym)
                            + (0.3 * key_variance_signal_pre)
                            + (0.3 * coupled_role_separation_signal)
                            - (0.3 * query_signal)
                            - 0.7
                        )
                        store_key_sharpen_strength = 0.015 * (1.0 + math.tanh(store_key_sharpen_logit))
                        sharpen_exponent = 1.0 + store_key_sharpen_strength
                        k_t = _positive_sum_normalize(np.power(np.maximum(k_t, 1e-6), sharpen_exponent))
                        k_centered = k_t - float(np.mean(k_t))
                        key_center_norm = float(np.linalg.norm(k_centered))
                        key_center_energy = float(np.dot(k_centered, k_centered)) + 1e-9
                    key_variance = float(np.var(k_t))
                    query_variance = float(np.var(q_t))
                    query_variance_signal = query_variance / (query_variance + 0.0025)
                    projection_write_signal = projection_magnitude / (projection_magnitude + 0.08)
                    v_base = np.asarray(
                        [
                            math.tanh(
                                value_seed + (0.1 * node.content_temperature * (idx - (d_value - 1) / 2.0))
                            )
                            for idx in range(d_value)
                        ],
                        dtype=np.float64,
                    )
                    state = memory_state[node.node_id]
                    decayed_state = memory_decay * state
                    v_hat_t = decayed_state @ k_t
                    delta_base = v_base - v_hat_t
                    raw_delta_norm_base = float(np.linalg.norm(delta_base))
                    value_norm_base = float(np.linalg.norm(v_base))
                    novelty_ratio_base = raw_delta_norm_base / (value_norm_base + 1e-6)
                    novelty_signal_base = novelty_ratio_base / (1.0 + novelty_ratio_base)
                    value_sep_logit = (
                        (0.75 * store_signal)
                        + (0.45 * novelty_signal_base)
                        + (0.35 * key_query_asym)
                        + (0.3 * query_variance_signal)
                        + (0.25 * projection_write_signal)
                        - (0.65 * query_signal)
                    )
                    value_sep_gain = 0.06 + (0.12 * (0.5 + (0.5 * math.tanh(value_sep_logit))))
                    store_value_preference = 0.75 + (0.25 * store_signal)
                    sep_axis = np.tanh(
                        (0.85 * position_axis)
                        + (0.55 * alt_sign)
                        + (0.4 * harmonic_sin)
                        - (0.25 * harmonic_cos)
                    )
                    sep_axis = sep_axis / (float(np.linalg.norm(sep_axis)) + 1e-9)
                    value_centered = v_base - float(np.mean(v_base))
                    value_spread = math.sqrt(float(np.var(v_base)) + 1e-9)
                    value_peaking = np.tanh(value_centered / (value_spread + 1e-6))
                    value_tilt = math.tanh(value_seed) * sep_axis
                    value_modulation = value_sep_gain * store_value_preference * (
                        (0.65 * value_peaking) + (0.35 * value_tilt)
                    )
                    v_t = np.tanh(v_base + value_modulation)
                    delta_t = v_t - v_hat_t
                    raw_delta_norm = float(np.linalg.norm(delta_t))
                    value_norm = float(np.linalg.norm(v_t))
                    novelty_ratio = raw_delta_norm / (value_norm + 1e-6)
                    novelty_signal = novelty_ratio / (1.0 + novelty_ratio)
                    key_query_cos = float(np.dot(q_t, k_t) / (np.linalg.norm(q_t) * np.linalg.norm(k_t) + 1e-9))
                    key_variance_signal = key_variance / (key_variance + 0.003)
                    query_collapse_signal = 0.003 / (query_variance + 0.003)
                    beta_mix = (
                        (0.45 * beta_base)
                        + (0.25 * store_signal)
                        + (0.2 * novelty_signal)
                        + (0.1 * key_query_asym)
                    )
                    beta_base_existing = beta_mix * (1.0 - (0.3 * query_signal))
                    key_query_collapse_signal = max(0.0, key_query_cos)
                    write_selectivity_logit = (
                        (0.9 * store_signal)
                        + (0.45 * novelty_signal)
                        + (0.35 * key_query_asym)
                        + (0.25 * key_variance_signal)
                        - (0.55 * query_signal)
                        - (0.4 * key_query_collapse_signal)
                        - (0.35 * query_collapse_signal)
                    )
                    write_eligibility = 0.5 + (0.5 * math.tanh(write_selectivity_logit))
                    beta_mod = 0.08 * math.tanh(write_selectivity_logit)
                    beta_t = beta_base_existing + beta_mod
                    beta_t = min(0.95, max(0.03, beta_t))
                    delta_t = _clip_vector_norm(delta_t, max_norm=delta_clip_norm)
                    update_scale = 0.85 + (0.35 * write_eligibility)
                    write_impact_logit = (
                        (1.05 * store_signal)
                        + (0.55 * write_eligibility)
                        + (0.45 * novelty_signal)
                        + (0.32 * key_variance_signal)
                        + (0.3 * (1.0 - min(1.0, max(0.0, key_query_cos))))
                        - (0.75 * query_signal)
                        - (0.42 * projection_write_signal)
                        - (0.3 * query_collapse_signal)
                        - 0.95
                    )
                    write_impact_gate = 0.8 + (
                        0.45
                        * (0.5 + (0.5 * math.tanh(write_impact_logit)))
                    )
                    delta_selectivity_logit = (
                        (0.95 * novelty_signal)
                        + (0.52 * store_signal)
                        + (0.4 * write_eligibility)
                        + (0.32 * key_variance_signal)
                        + (0.25 * (1.0 - min(1.0, max(0.0, key_query_cos))))
                        - (0.6 * query_signal)
                        - (0.4 * projection_write_signal)
                        - (0.35 * query_collapse_signal)
                        - 0.85
                    )
                    delta_selectivity = 0.85 + (
                        0.35
                        * (0.5 + (0.5 * math.tanh(delta_selectivity_logit)))
                    )
                    delta_t = delta_selectivity * delta_t
                    update_focus_logit = (
                        (0.9 * store_signal)
                        + (0.45 * write_eligibility)
                        + (0.35 * key_variance_signal)
                        + (0.35 * max(0.0, key_query_cos_base))
                        - (0.35 * query_signal)
                        - 0.85
                    )
                    update_focus_gain = 0.08 * (1.0 + math.tanh(update_focus_logit))
                    k_update_centered = k_t - float(np.mean(k_t))
                    k_update_std = math.sqrt(key_variance + 1e-9)
                    k_update_peaking = np.tanh(k_update_centered / (k_update_std + 1e-6))
                    update_margin_gain = 0.018 * (
                        1.0
                        + math.tanh(
                            (0.95 * store_signal)
                            + (0.45 * write_eligibility)
                            + (0.3 * key_variance_signal)
                            - (0.5 * query_signal)
                            - 0.55
                        )
                    )
                    k_update = _positive_sum_normalize(
                        np.maximum(
                            k_t * (1.0 + (update_focus_gain * k_update_peaking))
                            + (update_margin_gain * store_margin_profile),
                            1e-6,
                        )
                    )
                    update_address_logit = (
                        (0.95 * store_signal)
                        + (0.45 * write_eligibility)
                        + (0.35 * key_variance_signal)
                        + (0.3 * projection_write_signal)
                        - (0.55 * query_signal)
                        - 0.8
                    )
                    update_address_gain = 0.018 * (1.0 + math.tanh(update_address_logit))
                    k_update_centered_post = k_update - float(np.mean(k_update))
                    k_update = _positive_sum_normalize(
                        np.maximum(
                            k_update + (update_address_gain * np.maximum(k_update_centered_post, 0.0)),
                            1e-6,
                        )
                    )
                    effective_write_scale = beta_t * update_scale * write_impact_gate
                    update = effective_write_scale * np.outer(delta_t, k_update)
                    update_norm = float(np.linalg.norm(update, ord="fro"))
                    if update_norm > update_clip_frob:
                        update = update * (update_clip_frob / (update_norm + 1e-9))
                    new_state = decayed_state + update
                    memory_state[node.node_id] = new_state
                    read_t = decayed_state @ q_t
                    read_t = _clip_vector_norm(read_t, max_norm=read_clip_norm)
                    q_centered = q_t - float(np.mean(q_t))
                    query_focus_quality = (
                        (0.5 * query_variance_signal)
                        + (0.3 * query_signal)
                        + (0.2 * (1.0 - query_collapse_signal))
                    )
                    q_focus_temperature = 4.0 + (1.6 * query_variance_signal) + (1.1 * query_focus_quality)
                    q_focus_logits = q_focus_temperature * q_centered
                    q_focus_logits = q_focus_logits - float(np.max(q_focus_logits))
                    q_focus = np.exp(q_focus_logits)
                    q_focus = q_focus / (float(np.sum(q_focus)) + 1e-9)
                    q_dominant = np.maximum(q_t - float(np.mean(q_t)), 0.0)
                    q_dominant = q_dominant / (float(np.sum(q_dominant)) + 1e-9)
                    focus_blend = 0.2 + (0.35 * query_focus_quality)
                    q_focus = ((1.0 - focus_blend) * q_focus) + (focus_blend * q_dominant)
                    q_focus_margin_gain = 0.03 * (
                        1.0
                        + math.tanh(
                            (0.9 * query_signal)
                            + (0.55 * query_focus_quality)
                            + (0.35 * projection_signal)
                            - (0.45 * store_signal)
                            - 0.6
                        )
                    )
                    q_focus = q_focus + (q_focus_margin_gain * q_focus_profile)
                    q_focus = q_focus / (float(np.sum(q_focus)) + 1e-9)
                    if step_role == "query":
                        read_contrast_signal = float(np.max(read_t) - np.min(read_t))
                        query_focus_sharpen_logit = (
                            (0.55 * query_signal)
                            + (0.35 * max(0.0, key_query_cos))
                            + (0.3 * math.tanh(read_contrast_signal))
                            - (0.2 * store_signal)
                            - 0.45
                        )
                        query_focus_sharpen_strength = 0.06 * (
                            0.5 + (0.5 * math.tanh(query_focus_sharpen_logit))
                        )
                        q_focus_centered = q_focus - float(np.mean(q_focus))
                        # Scale the sharpening argument by the actual spread
                        # of ``q_focus`` rather than the hardcoded 0.05 that
                        # was used in v14x. ``q_focus`` is a probability
                        # distribution over ``d_key`` slots with mean
                        # ``1/d_key`` (~0.125 for d_key=8), so a constant
                        # denominator of 0.05 drove every entry of
                        # ``q_focus_centered / 0.05`` deep into the tanh
                        # saturation region and collapsed the sharpening
                        # into an effectively binary sign multiplier. Using
                        # the runtime std (with an eps floor) makes the
                        # sharpening scale-invariant: ``tanh`` now sees a
                        # z-score in the ~[-1, 1] band and reshapes
                        # ``q_focus`` smoothly in proportion to how far
                        # each slot sits from the mean of the current
                        # distribution, instead of sign-flipping.
                        q_focus_scale = float(np.std(q_focus)) + 1e-6
                        q_focus = q_focus * (
                            1.0
                            + (
                                query_focus_sharpen_strength
                                * np.tanh(q_focus_centered / q_focus_scale)
                            )
                        )
                        q_focus = _positive_sum_normalize(np.maximum(q_focus, 1e-6))
                    read_mean = float(np.mean(read_t))
                    read_abs_mean = float(np.mean(np.abs(read_t)))
                    selective_readout = float(np.dot(read_t, q_focus))
                    read_contrast = float(np.max(read_t) - np.min(read_t))
                    read_separation = read_contrast / (read_abs_mean + 1e-6)
                    separation_gate = math.tanh(read_separation)
                    contrast_signal = read_contrast / (read_contrast + read_abs_mean + 1e-6)
                    projection_read_signal = projection_magnitude / (projection_magnitude + 0.12)
                    diffuse_penalty = 1.0 - query_variance_signal
                    value_contrast_logit = (
                        (0.7 * contrast_signal)
                        + (0.55 * separation_gate)
                        + (0.35 * query_signal)
                        + (0.25 * query_variance_signal)
                        + (0.2 * projection_read_signal)
                        - (0.3 * max(0.0, key_query_cos))
                    )
                    value_contrast_gain = 0.5 + (0.5 * math.tanh(value_contrast_logit))
                    read_centered = read_t - read_mean
                    read_center_norm = float(np.linalg.norm(read_centered))
                    contrast_direction = read_centered / (read_center_norm + 1e-9)
                    contrast_readout = float(np.dot(contrast_direction, q_focus))
                    value_contrast_logit_2 = (
                        (0.85 * query_signal)
                        + (0.6 * query_focus_quality)
                        + (0.55 * contrast_signal)
                        + (0.45 * projection_read_signal)
                        + (0.25 * (1.0 - query_collapse_signal))
                        - (0.4 * max(0.0, key_query_cos))
                        - 1.0
                    )
                    value_contrast_gate = 0.5 + (0.5 * math.tanh(value_contrast_logit_2))
                    value_contrast_readout = float(np.dot(read_centered, q_focus))
                    selective_gain = 1.0 + (
                        0.45
                        * separation_gate
                        * (
                            (0.35 + (0.65 * query_focus_quality))
                            * (0.7 + (0.3 * projection_read_signal))
                        )
                    )
                    read_eligibility_logit = (
                        (1.15 * query_focus_quality)
                        + (0.65 * contrast_signal)
                        + (0.45 * projection_read_signal)
                        - (0.7 * query_collapse_signal)
                        - (0.45 * max(0.0, key_query_cos))
                    )
                    read_eligibility = 0.5 + (0.5 * math.tanh(read_eligibility_logit))
                    read_contrast_term = math.tanh(read_contrast) * (2.0 * query_signal - 1.0)
                    selective_weight = 0.61 + (0.08 * value_contrast_gain)
                    mean_weight = 0.16 - (0.05 * value_contrast_gain)
                    contrast_term_weight = 0.2 + (0.06 * value_contrast_gain)
                    contrast_readout_weight = 0.03 + (0.06 * value_contrast_gain)
                    value_contrast_weight = 0.03 + (0.07 * value_contrast_gain)
                    readout = (
                        (mean_weight * read_mean)
                        + (selective_weight * selective_gain * selective_readout)
                        + (contrast_term_weight * read_eligibility * read_contrast_term)
                        + (contrast_readout_weight * contrast_readout)
                        + (value_contrast_weight * value_contrast_gate * value_contrast_readout)
                        - (0.1 * diffuse_penalty * read_mean)
                    )
                    read_gain = max(
                        0.25,
                        min(
                            1.5,
                            (1.0 + node.slow_input_gain)
                            * (0.5 + (0.8 * query_signal))
                            * (0.9 + (0.25 * read_eligibility)),
                        ),
                    )
                    next_outputs[node.node_id] = math.tanh(summed_input + (read_gain * readout))
                    key_norm_vals.append(float(np.linalg.norm(k_t)))
                    query_norm_vals.append(float(np.linalg.norm(q_t)))
                    key_variance_vals.append(key_variance)
                    query_variance_vals.append(query_variance)
                    key_query_projection_vals.append(projection_magnitude)
                    query_decoupling_vals.append(float(np.linalg.norm(deflation_vector) + np.linalg.norm(extra_deflation_vector)))
                    key_query_cosine_vals.append(key_query_cos)
                    value_norm_vals.append(float(np.linalg.norm(v_t)))
                    memory_norm_vals.append(float(np.linalg.norm(new_state, ord="fro")))
                    beta_vals.append(beta_t)
                    delta_mag = float(np.linalg.norm(delta_t))
                    delta_correction_vals.append(delta_mag)
                    memory_read_vals.append(float(np.linalg.norm(read_t)))
                    readout_selectivity_vals.append(abs(selective_readout - read_mean) / (read_abs_mean + 1e-6))
                    novelty_ratio_vals.append(novelty_ratio)
                    readout_contrast_vals.append(read_contrast)
                    if step_role == "store":
                        beta_store_vals.append(beta_t)
                        store_update_vals.append(float(np.linalg.norm(update, ord="fro")))
                    elif step_role == "distractor":
                        beta_distractor_vals.append(beta_t)
                        distractor_update_vals.append(float(np.linalg.norm(update, ord="fro")))
                    elif step_role == "query":
                        beta_query_vals.append(beta_t)
                        query_update_vals.append(float(np.linalg.norm(update, ord="fro")))
                        # Query<->Memory alignment diagnostic. Previously this
                        # list was populated with ``key_query_cos`` which made
                        # ``query_memory_alignment`` a silent clone of
                        # ``key_query_cosine_at_query``. The real quantity of
                        # interest is how strongly the query extracts signal
                        # from the memory being read, so we use a normalized
                        # ratio of the memory readout magnitude versus the
                        # trivial Cauchy-Schwarz ceiling
                        # ``||q_t|| * ||decayed_state||_F``. The result is
                        # bounded in [0, 1] and reuses only tensors that are
                        # already computed at query time.
                        q_t_norm = float(np.linalg.norm(q_t))
                        decayed_state_norm = float(
                            np.linalg.norm(decayed_state, ord="fro")
                        )
                        read_norm = float(np.linalg.norm(read_t))
                        query_memory_alignment_step = read_norm / (
                            (q_t_norm * decayed_state_norm) + 1e-9
                        )
                        query_alignment_vals.append(query_memory_alignment_step)
                        key_query_cosine_query_vals.append(key_query_cos)
                    if step_capture is not None:
                        # Passive per-node snapshot of the delta-memory hot
                        # path. Overwritten on each activation sub-iteration so
                        # the final entry in ``step_capture`` reflects the last
                        # sub-iteration of the outer step, matching what the
                        # evaluator sees.
                        q_t_norm_cap = float(np.linalg.norm(q_t))
                        decayed_state_norm_cap = float(
                            np.linalg.norm(decayed_state, ord="fro")
                        )
                        read_norm_cap = float(np.linalg.norm(read_t))
                        query_memory_alignment_cap = read_norm_cap / (
                            (q_t_norm_cap * decayed_state_norm_cap) + 1e-9
                        )
                        step_capture[node.node_id] = {
                            "node_id": int(node.node_id),
                            "is_output": bool(node.is_output),
                            "beta_base": float(beta_base),
                            "beta_t": float(beta_t),
                            "store_signal": float(store_signal),
                            "query_signal": float(query_signal),
                            "key_query_cos_base": float(key_query_cos_base),
                            "key_query_cos_post": float(key_query_cos),
                            "k_t": [float(value) for value in k_t],
                            "q_t": [float(value) for value in q_t],
                            "q_focus": [float(value) for value in q_focus],
                            "v_t": [float(value) for value in v_t],
                            "read_t": [float(value) for value in read_t],
                            "memory_frob_pre": decayed_state_norm_cap,
                            "memory_frob_post": float(
                                np.linalg.norm(new_state, ord="fro")
                            ),
                            "query_memory_alignment": query_memory_alignment_cap,
                            "readout_scalar": float(readout),
                            "selective_readout": float(selective_readout),
                            "read_contrast": float(read_contrast),
                            "readout_selectivity": (
                                abs(selective_readout - read_mean)
                                / (read_abs_mean + 1e-6)
                            ),
                            "update_frob": float(
                                np.linalg.norm(update, ord="fro")
                            ),
                            "delta_correction": float(
                                np.linalg.norm(delta_t)
                            ),
                            "node_output": float(next_outputs[node.node_id]),
                        }
                outputs = next_outputs
            if trace_sink is not None and step_capture is not None:
                for node_id, snapshot in step_capture.items():
                    trace_sink.append(
                        {
                            "step_index": int(step_index),
                            "step_role": str(step_role),
                            **snapshot,
                        }
                    )
            sequence_outputs.append(np.array([outputs.get(node_id, 0.0) for node_id in genome.output_ids], dtype=np.float32))
        mean_store_beta = float(np.mean(beta_store_vals)) if beta_store_vals else 0.0
        mean_distractor_beta = float(np.mean(beta_distractor_vals)) if beta_distractor_vals else 0.0
        mean_store_update = float(np.mean(store_update_vals)) if store_update_vals else 0.0
        mean_distractor_update = float(np.mean(distractor_update_vals)) if distractor_update_vals else 0.0
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=False,
            mean_abs_delta_w=0.0,
            max_abs_delta_w=0.0,
            clamp_hit_rate=0.0,
            plasticity_active_fraction=0.0,
            mean_abs_decay_term=0.0,
            max_abs_decay_term=0.0,
            decay_effect_ratio=0.0,
            decay_near_zero_fraction=0.0,
            mean_beta_write=float(np.mean(beta_vals)) if beta_vals else 0.0,
            beta_at_store=mean_store_beta,
            beta_at_distractor=mean_distractor_beta,
            beta_at_query=float(np.mean(beta_query_vals)) if beta_query_vals else 0.0,
            store_vs_distractor_beta_gap=mean_store_beta - mean_distractor_beta,
            write_gate_at_store=mean_store_update,
            write_gate_at_distractor=mean_distractor_update,
            write_gate_at_query=float(np.mean(query_update_vals)) if query_update_vals else 0.0,
            store_vs_distractor_write_gap=mean_store_update - mean_distractor_update,
            mean_key_norm=float(np.mean(key_norm_vals)) if key_norm_vals else 0.0,
            mean_query_norm=float(np.mean(query_norm_vals)) if query_norm_vals else 0.0,
            mean_value_norm=float(np.mean(value_norm_vals)) if value_norm_vals else 0.0,
            mean_memory_frobenius_norm=float(np.mean(memory_norm_vals)) if memory_norm_vals else 0.0,
            query_memory_alignment=float(np.mean(query_alignment_vals)) if query_alignment_vals else 0.0,
            store_memory_update_strength=mean_store_update,
            delta_correction_magnitude=float(np.mean(delta_correction_vals)) if delta_correction_vals else 0.0,
            memory_read_strength=float(np.mean(memory_read_vals)) if memory_read_vals else 0.0,
            key_query_cosine_mean=float(np.mean(key_query_cosine_vals)) if key_query_cosine_vals else 0.0,
            key_query_cosine_at_query=float(np.mean(key_query_cosine_query_vals)) if key_query_cosine_query_vals else 0.0,
            key_variance_mean=float(np.mean(key_variance_vals)) if key_variance_vals else 0.0,
            query_variance_mean=float(np.mean(query_variance_vals)) if query_variance_vals else 0.0,
            key_query_projection_strength=float(np.mean(key_query_projection_vals)) if key_query_projection_vals else 0.0,
            query_decoupling_magnitude=float(np.mean(query_decoupling_vals)) if query_decoupling_vals else 0.0,
            readout_selectivity=float(np.mean(readout_selectivity_vals)) if readout_selectivity_vals else 0.0,
            mean_match_signal=float(np.mean(novelty_ratio_vals)) if novelty_ratio_vals else 0.0,
            query_value_read_strength=float(np.mean(readout_contrast_vals)) if readout_contrast_vals else 0.0,
        )
        return np.vstack(sequence_outputs)

    def last_episode_metrics(self) -> PlasticityEpisodeMetrics:
        return self._last_metrics


class PlasticNetworkExecutor(StatefulNetworkExecutor):
    def __init__(self, activation_steps: int, delta_w_clamp: float = 1.0) -> None:
        super().__init__(activation_steps=activation_steps)
        self.delta_w_clamp = float(delta_w_clamp)
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=True,
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

    def run_sequence(
        self,
        genome: GenomeModel,
        input_sequence: Sequence[Sequence[float]],
        *,
        step_roles: Sequence[str] | None = None,
    ) -> np.ndarray:
        del step_roles
        incoming_by_target = _incoming_connections_by_target(genome)
        enabled_connections = tuple(conn for conn in genome.connections if conn.enabled)

        # Explicit episode reset: runtime plastic changes never persist across sequences.
        delta_weights = {
            (conn.in_id, conn.out_id, conn.historical_marker): 0.0
            for conn in enabled_connections
        }
        clamp_hits = 0
        total_updates = 0
        memory = {node.node_id: 0.0 for node in genome.nodes if not node.is_input}
        outputs = {node.node_id: 0.0 for node in genome.nodes}
        sequence_outputs: list[np.ndarray] = []

        for inputs in input_sequence:
            input_map = {
                node_id: float(value)
                for node_id, value in zip(genome.input_ids, inputs, strict=True)
            }
            outputs.update(input_map)

            for _ in range(self.activation_steps):
                previous_outputs = dict(outputs)
                next_outputs = dict(input_map)

                for node in genome.nodes:
                    if node.is_input:
                        continue

                    incoming_sum = 0.0
                    for conn in incoming_by_target.get(node.node_id, ()):
                        conn_key = (conn.in_id, conn.out_id, conn.historical_marker)
                        effective_weight = conn.weight + delta_weights[conn_key]
                        incoming_sum += effective_weight * previous_outputs.get(conn.in_id, 0.0)

                    new_memory = update_memory(
                        previous_memory=memory[node.node_id],
                        alpha=node.alpha,
                        weighted_sum=incoming_sum,
                        bias=node.bias,
                    )
                    memory[node.node_id] = new_memory
                    next_outputs[node.node_id] = math.tanh(new_memory)

                outputs = next_outputs

            # Hebbian update fires once per timestep after activation settling.
            for conn in enabled_connections:
                if conn.eta == 0.0:
                    continue
                conn_key = (conn.in_id, conn.out_id, conn.historical_marker)
                pre_value = outputs.get(conn.in_id, 0.0)
                post_value = outputs.get(conn.out_id, 0.0)
                updated_delta = update_hebb_delta_weight(
                    delta_weights[conn_key],
                    eta=conn.eta,
                    pre_value=pre_value,
                    post_value=post_value,
                    clamp=self.delta_w_clamp,
                )
                if self.delta_w_clamp > 0.0 and math.isclose(
                    abs(updated_delta),
                    self.delta_w_clamp,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    clamp_hits += 1
                total_updates += 1
                delta_weights[conn_key] = updated_delta

            sequence_outputs.append(
                np.array(
                    [outputs.get(node_id, 0.0) for node_id in genome.output_ids],
                    dtype=np.float32,
                )
            )

        abs_deltas = [abs(delta) for delta in delta_weights.values()]
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=True,
            mean_abs_delta_w=float(np.mean(abs_deltas)) if abs_deltas else 0.0,
            max_abs_delta_w=float(np.max(abs_deltas)) if abs_deltas else 0.0,
            clamp_hit_rate=(clamp_hits / total_updates) if total_updates else 0.0,
            plasticity_active_fraction=(
                sum(1 for delta in abs_deltas if delta > 1e-6) / len(abs_deltas)
                if abs_deltas
                else 0.0
            ),
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
        return np.vstack(sequence_outputs)

    def last_episode_metrics(self) -> PlasticityEpisodeMetrics:
        return self._last_metrics


class AdaptivePlasticNetworkExecutor(StatefulNetworkExecutor):
    def __init__(self, activation_steps: int, delta_w_clamp: float = 0.5) -> None:
        super().__init__(activation_steps=activation_steps)
        self.delta_w_clamp = float(delta_w_clamp)
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=True,
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

    def run_sequence(
        self,
        genome: GenomeModel,
        input_sequence: Sequence[Sequence[float]],
        *,
        step_roles: Sequence[str] | None = None,
    ) -> np.ndarray:
        del step_roles
        incoming_by_target = _incoming_connections_by_target(genome)
        enabled_connections = tuple(conn for conn in genome.connections if conn.enabled)

        delta_weights = {
            (conn.in_id, conn.out_id, conn.historical_marker): 0.0
            for conn in enabled_connections
        }
        clamp_hits = 0
        total_updates = 0
        total_abs_decay_term = 0.0
        total_abs_plastic_term = 0.0
        max_abs_decay_term = 0.0
        decay_near_zero_hits = 0
        memory = {node.node_id: 0.0 for node in genome.nodes if not node.is_input}
        outputs = {node.node_id: 0.0 for node in genome.nodes}
        sequence_outputs: list[np.ndarray] = []

        for inputs in input_sequence:
            input_map = {
                node_id: float(value)
                for node_id, value in zip(genome.input_ids, inputs, strict=True)
            }
            outputs.update(input_map)

            for _ in range(self.activation_steps):
                previous_outputs = dict(outputs)
                next_outputs = dict(input_map)

                for node in genome.nodes:
                    if node.is_input:
                        continue

                    incoming_sum = 0.0
                    for conn in incoming_by_target.get(node.node_id, ()):
                        conn_key = (conn.in_id, conn.out_id, conn.historical_marker)
                        effective_weight = conn.weight + delta_weights[conn_key]
                        incoming_sum += effective_weight * previous_outputs.get(conn.in_id, 0.0)

                    new_memory = update_memory(
                        previous_memory=memory[node.node_id],
                        alpha=node.alpha,
                        weighted_sum=incoming_sum,
                        bias=node.bias,
                    )
                    memory[node.node_id] = new_memory
                    next_outputs[node.node_id] = math.tanh(new_memory)

                outputs = next_outputs

            for conn in enabled_connections:
                conn_key = (conn.in_id, conn.out_id, conn.historical_marker)
                pre_value = outputs.get(conn.in_id, 0.0)
                post_value = outputs.get(conn.out_id, 0.0)
                previous_delta = delta_weights[conn_key]
                plastic_term = conn.eta * (
                    (conn.plastic_a * pre_value * post_value)
                    + (conn.plastic_b * pre_value)
                    + (conn.plastic_c * post_value)
                )
                decay_term = conn.plastic_d * previous_delta
                updated_delta = update_adaptive_delta_weight(
                    previous_delta,
                    eta=conn.eta,
                    plastic_a=conn.plastic_a,
                    plastic_b=conn.plastic_b,
                    plastic_c=conn.plastic_c,
                    plastic_d=conn.plastic_d,
                    pre_value=pre_value,
                    post_value=post_value,
                    clamp=self.delta_w_clamp,
                )
                total_abs_decay_term += abs(decay_term)
                total_abs_plastic_term += abs(plastic_term)
                max_abs_decay_term = max(max_abs_decay_term, abs(decay_term))
                if abs(decay_term) <= 1e-6:
                    decay_near_zero_hits += 1
                if self.delta_w_clamp > 0.0 and math.isclose(
                    abs(updated_delta),
                    self.delta_w_clamp,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    clamp_hits += 1
                total_updates += 1
                delta_weights[conn_key] = updated_delta

            sequence_outputs.append(
                np.array(
                    [outputs.get(node_id, 0.0) for node_id in genome.output_ids],
                    dtype=np.float32,
                )
            )

        abs_deltas = [abs(delta) for delta in delta_weights.values()]
        self._last_metrics = PlasticityEpisodeMetrics(
            plasticity_enabled=True,
            mean_abs_delta_w=float(np.mean(abs_deltas)) if abs_deltas else 0.0,
            max_abs_delta_w=float(np.max(abs_deltas)) if abs_deltas else 0.0,
            clamp_hit_rate=(clamp_hits / total_updates) if total_updates else 0.0,
            plasticity_active_fraction=(
                sum(1 for delta in abs_deltas if delta > 1e-6) / len(abs_deltas)
                if abs_deltas
                else 0.0
            ),
            mean_abs_decay_term=(total_abs_decay_term / total_updates) if total_updates else 0.0,
            max_abs_decay_term=max_abs_decay_term,
            decay_effect_ratio=(
                total_abs_decay_term / total_abs_plastic_term
                if total_abs_plastic_term > 1e-9
                else 0.0
            ),
            decay_near_zero_fraction=(decay_near_zero_hits / total_updates) if total_updates else 0.0,
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
        return np.vstack(sequence_outputs)

    def last_episode_metrics(self) -> PlasticityEpisodeMetrics:
        return self._last_metrics


def _step_role_at(step_roles: Sequence[str] | None, step_index: int) -> str:
    if step_roles is None:
        return ""
    if step_index < 0 or step_index >= len(step_roles):
        return ""
    return str(step_roles[step_index]).strip().lower()


def _role_mean(role_totals: dict[str, tuple[float, float, int]], role: str, *, index: int) -> float:
    total_fast, total_slow, count = role_totals.get(role, (0.0, 0.0, 0))
    if count <= 0:
        return 0.0
    if index == 0:
        return total_fast / count
    return total_slow / count


def _role_mean_4(role_totals: dict[str, tuple[float, float, float, int]], role: str, *, index: int) -> float:
    first, second, third, count = role_totals.get(role, (0.0, 0.0, 0.0, 0))
    if count <= 0:
        return 0.0
    values = (first, second, third)
    return values[index] / count
