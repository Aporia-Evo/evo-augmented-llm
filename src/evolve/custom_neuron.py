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
                    match_bias = float(node.content_b_match)
                    logits = np.asarray(
                        [
                            (match_scale * query_key * slot_keys[node.node_id][slot_idx]) + match_bias
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
        delta_correction_vals: list[float] = []
        memory_read_vals: list[float] = []

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
                    summed_input = 0.0
                    for conn in incoming_by_target.get(node.node_id, ()):
                        summed_input += conn.weight * previous_outputs.get(conn.in_id, 0.0)
                    x_t = summed_input + node.bias
                    x_norm = x_t / (1.0 + abs(x_t))
                    x_abs = abs(x_norm)
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
                    beta_t = beta_base * (0.15 + (0.85 * store_signal)) * (1.0 - (0.5 * query_signal))
                    beta_t = min(0.98, max(0.02, beta_t))
                    key_core = (
                        key_seed * (0.9 + (0.5 * center_peak))
                        + (0.6 * node.content_b_key * position_axis)
                        + (0.35 * node.content_w_key * alt_sign)
                    )
                    key_cross = 0.25 * query_seed * position_axis * alt_sign
                    k_raw = (
                        1.0
                        + (0.45 * np.tanh(key_core))
                        + (0.35 * np.tanh((key_core * alt_sign) + (0.5 * key_cross)))
                        + (0.2 * np.tanh((key_seed - query_seed) * (center_peak - 0.5)))
                    )
                    query_core = (
                        query_seed * (0.8 - (0.4 * center_peak))
                        - (0.55 * node.content_b_query * position_axis)
                        + (0.4 * node.content_w_query * centered_square)
                    )
                    q_raw = (
                        1.0
                        + (0.5 * np.tanh(query_core + (0.3 * alt_sign)))
                        + (0.3 * np.tanh((query_seed - key_seed) * (position_axis * alt_sign)))
                        + (0.25 * np.tanh((query_core * position_axis) - (0.2 * key_seed * alt_sign)))
                    )
                    k_raw = np.maximum(k_raw, 1e-3)
                    q_raw = np.maximum(q_raw, 1e-3)
                    k_t = _positive_sum_normalize(k_raw)
                    q_t = _positive_sum_normalize(q_raw)
                    v_t = np.asarray(
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
                    delta_t = v_t - v_hat_t
                    delta_t = _clip_vector_norm(delta_t, max_norm=delta_clip_norm)
                    update = beta_t * np.outer(delta_t, k_t)
                    update_norm = float(np.linalg.norm(update, ord="fro"))
                    if update_norm > update_clip_frob:
                        update = update * (update_clip_frob / (update_norm + 1e-9))
                    new_state = decayed_state + update
                    memory_state[node.node_id] = new_state
                    read_t = decayed_state @ q_t
                    read_t = _clip_vector_norm(read_t, max_norm=read_clip_norm)
                    read_gain = max(
                        0.25,
                        min(
                            1.5,
                            (1.0 + node.slow_input_gain) * (0.5 + (0.8 * query_signal)),
                        ),
                    )
                    readout = float(np.mean(read_t))
                    next_outputs[node.node_id] = math.tanh(summed_input + (read_gain * readout))
                    key_norm_vals.append(float(np.linalg.norm(k_t)))
                    query_norm_vals.append(float(np.linalg.norm(q_t)))
                    value_norm_vals.append(float(np.linalg.norm(v_t)))
                    memory_norm_vals.append(float(np.linalg.norm(new_state, ord="fro")))
                    beta_vals.append(beta_t)
                    delta_mag = float(np.linalg.norm(delta_t))
                    delta_correction_vals.append(delta_mag)
                    memory_read_vals.append(float(np.linalg.norm(read_t)))
                    if step_role == "store":
                        beta_store_vals.append(beta_t)
                        store_update_vals.append(float(np.linalg.norm(update, ord="fro")))
                    elif step_role == "distractor":
                        beta_distractor_vals.append(beta_t)
                    elif step_role == "query":
                        beta_query_vals.append(beta_t)
                        query_alignment_vals.append(float(np.dot(q_t, k_t) / (np.linalg.norm(q_t) * np.linalg.norm(k_t) + 1e-9)))
                outputs = next_outputs
            sequence_outputs.append(np.array([outputs.get(node_id, 0.0) for node_id in genome.output_ids], dtype=np.float32))
        mean_store_beta = float(np.mean(beta_store_vals)) if beta_store_vals else 0.0
        mean_distractor_beta = float(np.mean(beta_distractor_vals)) if beta_distractor_vals else 0.0
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
            mean_key_norm=float(np.mean(key_norm_vals)) if key_norm_vals else 0.0,
            mean_query_norm=float(np.mean(query_norm_vals)) if query_norm_vals else 0.0,
            mean_value_norm=float(np.mean(value_norm_vals)) if value_norm_vals else 0.0,
            mean_memory_frobenius_norm=float(np.mean(memory_norm_vals)) if memory_norm_vals else 0.0,
            query_memory_alignment=float(np.mean(query_alignment_vals)) if query_alignment_vals else 0.0,
            store_memory_update_strength=float(np.mean(store_update_vals)) if store_update_vals else 0.0,
            delta_correction_magnitude=float(np.mean(delta_correction_vals)) if delta_correction_vals else 0.0,
            memory_read_strength=float(np.mean(memory_read_vals)) if memory_read_vals else 0.0,
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
