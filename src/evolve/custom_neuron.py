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
        readout_query_vals: list[float] = []
        readout_distractor_vals: list[float] = []
        role_totals: dict[str, tuple[float, float, int]] = {}

        for step_index, inputs in enumerate(input_sequence):
            step_role = _step_role_at(step_roles, step_index)
            input_map = {node_id: float(value) for node_id, value in zip(genome.input_ids, inputs, strict=True)}
            outputs.update(input_map)
            query_signal = float(inputs[1]) if len(inputs) > 1 else 0.0
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
                    write_gate = 1.0 / (1.0 + math.exp(-((node.content_w_key * input_norm) + node.content_b_key)))
                    key_input = summed_input + node.bias
                    value_input = summed_input + node.slow_output_gain
                    new_key = (clamp_alpha(node.alpha) * key_state[node.node_id]) + (write_gate * key_input)
                    new_value = (clamp_alpha(node.alpha_slow) * value_state[node.node_id]) + (write_gate * value_input)
                    match_t = 1.0 / (1.0 + math.exp(-((node.content_temperature * query_signal * new_key) + node.content_b_match)))
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
                    if step_role == "query":
                        query_alignment_vals.append(abs(query_signal * new_key))
                        query_read_vals.append(abs(readout_t))
                        readout_query_vals.append(abs(readout_t))
                    if step_role == "store":
                        store_coupling_vals.append(abs(new_key * new_value))
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
