from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from config import AppConfig
from evolve.plasticity import (
    is_stateful_v2_variant,
    plastic_d_bounds_for_variant,
    plastic_d_init_for_variant,
    plastic_d_mutation_for_variant,
    plastic_fixed_d_for_variant,
    plastic_mode_for_variant,
)
from tensorneat.algorithm.neat import NEAT
from tensorneat.common import State, mutate_float
from tensorneat.genome.gene.base import BaseGene
from tensorneat.genome.gene.conn.base import BaseConn
from tensorneat.genome.gene.node.base import BaseNode
from tensorneat.genome.gene import OriginConn
from tensorneat.genome.operations import DefaultCrossover, DefaultDistance, DefaultMutation
from tensorneat.genome.recurrent import RecurrentGenome


class AlphaNode(BaseNode):
    custom_attrs = ["bias", "alpha", "alpha_slow", "slow_input_gain", "slow_output_gain"]

    def __init__(
        self,
        *,
        bias_init_std: float,
        bias_mutate_power: float,
        bias_mutate_rate: float,
        bias_replace_rate: float,
        bias_lower_bound: float,
        bias_upper_bound: float,
        alpha_init_mean: float,
        alpha_init_std: float,
        alpha_mutate_power: float,
        alpha_mutate_rate: float,
        alpha_replace_rate: float,
        alpha_slow_init_mean: float,
        alpha_slow_init_std: float,
        alpha_slow_mutate_power: float,
        alpha_slow_mutate_rate: float,
        alpha_slow_replace_rate: float,
        slow_input_gain_init_mean: float,
        slow_input_gain_init_std: float,
        slow_input_gain_mutate_power: float,
        slow_input_gain_mutate_rate: float,
        slow_input_gain_replace_rate: float,
        slow_input_gain_lower_bound: float,
        slow_input_gain_upper_bound: float,
        slow_output_gain_init_mean: float,
        slow_output_gain_init_std: float,
        slow_output_gain_mutate_power: float,
        slow_output_gain_mutate_rate: float,
        slow_output_gain_replace_rate: float,
        slow_output_gain_lower_bound: float,
        slow_output_gain_upper_bound: float,
        enable_two_timescale: bool = False,
        force_alpha_zero: bool = False,
    ) -> None:
        super().__init__()
        self.bias_init_std = bias_init_std
        self.bias_mutate_power = bias_mutate_power
        self.bias_mutate_rate = bias_mutate_rate
        self.bias_replace_rate = bias_replace_rate
        self.bias_lower_bound = bias_lower_bound
        self.bias_upper_bound = bias_upper_bound
        self.alpha_init_mean = alpha_init_mean
        self.alpha_init_std = alpha_init_std
        self.alpha_mutate_power = alpha_mutate_power
        self.alpha_mutate_rate = alpha_mutate_rate
        self.alpha_replace_rate = alpha_replace_rate
        self.alpha_slow_init_mean = alpha_slow_init_mean
        self.alpha_slow_init_std = alpha_slow_init_std
        self.alpha_slow_mutate_power = alpha_slow_mutate_power
        self.alpha_slow_mutate_rate = alpha_slow_mutate_rate
        self.alpha_slow_replace_rate = alpha_slow_replace_rate
        self.slow_input_gain_init_mean = slow_input_gain_init_mean
        self.slow_input_gain_init_std = slow_input_gain_init_std
        self.slow_input_gain_mutate_power = slow_input_gain_mutate_power
        self.slow_input_gain_mutate_rate = slow_input_gain_mutate_rate
        self.slow_input_gain_replace_rate = slow_input_gain_replace_rate
        self.slow_input_gain_lower_bound = slow_input_gain_lower_bound
        self.slow_input_gain_upper_bound = slow_input_gain_upper_bound
        self.slow_output_gain_init_mean = slow_output_gain_init_mean
        self.slow_output_gain_init_std = slow_output_gain_init_std
        self.slow_output_gain_mutate_power = slow_output_gain_mutate_power
        self.slow_output_gain_mutate_rate = slow_output_gain_mutate_rate
        self.slow_output_gain_replace_rate = slow_output_gain_replace_rate
        self.slow_output_gain_lower_bound = slow_output_gain_lower_bound
        self.slow_output_gain_upper_bound = slow_output_gain_upper_bound
        self.enable_two_timescale = enable_two_timescale
        self.force_alpha_zero = force_alpha_zero

    def new_identity_attrs(self, state: State) -> jnp.ndarray:
        return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    def new_random_attrs(self, state: State, randkey: jax.Array) -> jnp.ndarray:
        k1, k2, k3, k4, k5 = jax.random.split(randkey, 5)
        bias = jax.random.normal(k1, ()) * self.bias_init_std
        bias = jnp.clip(bias, self.bias_lower_bound, self.bias_upper_bound)
        if self.force_alpha_zero:
            alpha = 0.0
        else:
            alpha = jax.random.normal(k2, ()) * self.alpha_init_std + self.alpha_init_mean
            alpha = jnp.clip(alpha, 0.0, 1.0)
        if self.enable_two_timescale:
            alpha_slow = jax.random.normal(k3, ()) * self.alpha_slow_init_std + self.alpha_slow_init_mean
            alpha_slow = jnp.clip(alpha_slow, 0.0, 1.0)
            slow_input_gain = (
                jax.random.normal(k4, ()) * self.slow_input_gain_init_std
                + self.slow_input_gain_init_mean
            )
            slow_input_gain = jnp.clip(
                slow_input_gain,
                self.slow_input_gain_lower_bound,
                self.slow_input_gain_upper_bound,
            )
            slow_output_gain = (
                jax.random.normal(k5, ()) * self.slow_output_gain_init_std
                + self.slow_output_gain_init_mean
            )
            slow_output_gain = jnp.clip(
                slow_output_gain,
                self.slow_output_gain_lower_bound,
                self.slow_output_gain_upper_bound,
            )
        else:
            alpha_slow = 0.0
            slow_input_gain = 0.0
            slow_output_gain = 0.0
        return jnp.array(
            [bias, alpha, alpha_slow, slow_input_gain, slow_output_gain],
            dtype=jnp.float32,
        )

    def mutate(self, state: State, randkey: jax.Array, attrs: jnp.ndarray) -> jnp.ndarray:
        k1, k2, k3, k4, k5 = jax.random.split(randkey, 5)
        bias, alpha, alpha_slow, slow_input_gain, slow_output_gain = attrs
        bias = mutate_float(
            k1,
            bias,
            0.0,
            self.bias_init_std,
            self.bias_mutate_power,
            self.bias_mutate_rate,
            self.bias_replace_rate,
        )
        bias = jnp.clip(bias, self.bias_lower_bound, self.bias_upper_bound)
        if self.force_alpha_zero:
            alpha = 0.0
        else:
            alpha = mutate_float(
                k2,
                alpha,
                self.alpha_init_mean,
                self.alpha_init_std,
                self.alpha_mutate_power,
                self.alpha_mutate_rate,
                self.alpha_replace_rate,
            )
            alpha = jnp.clip(alpha, 0.0, 1.0)
        if self.enable_two_timescale:
            alpha_slow = mutate_float(
                k3,
                alpha_slow,
                self.alpha_slow_init_mean,
                self.alpha_slow_init_std,
                self.alpha_slow_mutate_power,
                self.alpha_slow_mutate_rate,
                self.alpha_slow_replace_rate,
            )
            alpha_slow = jnp.clip(alpha_slow, 0.0, 1.0)
            slow_input_gain = mutate_float(
                k4,
                slow_input_gain,
                self.slow_input_gain_init_mean,
                self.slow_input_gain_init_std,
                self.slow_input_gain_mutate_power,
                self.slow_input_gain_mutate_rate,
                self.slow_input_gain_replace_rate,
            )
            slow_input_gain = jnp.clip(
                slow_input_gain,
                self.slow_input_gain_lower_bound,
                self.slow_input_gain_upper_bound,
            )
            slow_output_gain = mutate_float(
                k5,
                slow_output_gain,
                self.slow_output_gain_init_mean,
                self.slow_output_gain_init_std,
                self.slow_output_gain_mutate_power,
                self.slow_output_gain_mutate_rate,
                self.slow_output_gain_replace_rate,
            )
            slow_output_gain = jnp.clip(
                slow_output_gain,
                self.slow_output_gain_lower_bound,
                self.slow_output_gain_upper_bound,
            )
        else:
            alpha_slow = 0.0
            slow_input_gain = 0.0
            slow_output_gain = 0.0
        return jnp.array(
            [bias, alpha, alpha_slow, slow_input_gain, slow_output_gain],
            dtype=jnp.float32,
        )

    def distance(self, state: State, attrs1: jnp.ndarray, attrs2: jnp.ndarray) -> jnp.ndarray:
        return (
            jnp.abs(attrs1[0] - attrs2[0])
            + jnp.abs(attrs1[1] - attrs2[1])
            + jnp.abs(attrs1[2] - attrs2[2])
            + jnp.abs(attrs1[3] - attrs2[3])
            + jnp.abs(attrs1[4] - attrs2[4])
        )

    def forward(
        self,
        state: State,
        attrs: jnp.ndarray,
        inputs: jnp.ndarray,
        is_output_node: bool = False,
        valid_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        del state, is_output_node
        bias, _alpha = attrs
        if valid_mask is None:
            valid_mask = ~jnp.isnan(inputs)
        safe_inputs = jnp.where(valid_mask, inputs, 0.0)
        return jnp.tanh(bias + jnp.sum(safe_inputs))

    def repr(self, state: State, node: jnp.ndarray, precision: int = 2, idx_width: int = 3, func_width: int = 8) -> str:
        del state, func_width
        return (
            f"AlphaNode(idx={int(node[0]):<{idx_width}}, "
            f"bias={float(node[1]):.{precision}f}, alpha={float(node[2]):.{precision}f}, "
            f"alpha_slow={float(node[3]):.{precision}f}, slow_in={float(node[4]):.{precision}f}, "
            f"slow_out={float(node[5]):.{precision}f})"
        )

    def to_dict(self, state: State, node: jnp.ndarray) -> dict[str, Any]:
        del state
        return {
            "idx": int(node[0]),
            "bias": float(node[1]),
            "alpha": float(node[2]),
            "alpha_slow": float(node[3]) if node.shape[0] > 3 else 0.0,
            "slow_input_gain": float(node[4]) if node.shape[0] > 4 else 0.0,
            "slow_output_gain": float(node[5]) if node.shape[0] > 5 else 0.0,
        }

    def sympy_func(self, state: State, node_dict: dict[str, Any], inputs: Any, is_output_node: bool = False) -> tuple[Any, dict[Any, Any]]:
        del state, inputs, is_output_node
        bias = sp.symbols(f"n_{node_dict['idx']}_b")
        alpha = sp.symbols(f"n_{node_dict['idx']}_a")
        return sp.tanh(bias + alpha), {bias: node_dict["bias"], alpha: node_dict["alpha"]}


class EnabledOriginConn(OriginConn):
    custom_attrs = ["weight", "enabled", "eta", "plastic_a", "plastic_b", "plastic_c", "plastic_d"]

    def __init__(
        self,
        *,
        weight_init_std: float,
        weight_mutate_power: float,
        weight_mutate_rate: float,
        weight_replace_rate: float,
        weight_lower_bound: float,
        weight_upper_bound: float,
        enabled_toggle_rate: float,
        eta_init_mean: float,
        eta_init_std: float,
        eta_mutate_power: float,
        eta_mutate_rate: float,
        eta_replace_rate: float,
        eta_lower_bound: float,
        eta_upper_bound: float,
        plastic_a_init_mean: float,
        plastic_a_init_std: float,
        plastic_a_mutate_power: float,
        plastic_a_mutate_rate: float,
        plastic_a_replace_rate: float,
        plastic_a_lower_bound: float,
        plastic_a_upper_bound: float,
        plastic_d_init_mean: float,
        plastic_d_init_std: float,
        plastic_d_mutate_power: float,
        plastic_d_mutate_rate: float,
        plastic_d_replace_rate: float,
        plastic_d_lower_bound: float,
        plastic_d_upper_bound: float,
        plastic_mode: str | None = None,
        fixed_plastic_d: float | None = None,
    ) -> None:
        BaseGene.__init__(self)
        self.weight_init_mean = 0.0
        self.weight_init_std = weight_init_std
        self.weight_mutate_power = weight_mutate_power
        self.weight_mutate_rate = weight_mutate_rate
        self.weight_replace_rate = weight_replace_rate
        self.weight_lower_bound = weight_lower_bound
        self.weight_upper_bound = weight_upper_bound
        self.enabled_toggle_rate = enabled_toggle_rate
        self.eta_init_mean = eta_init_mean
        self.eta_init_std = eta_init_std
        self.eta_mutate_power = eta_mutate_power
        self.eta_mutate_rate = eta_mutate_rate
        self.eta_replace_rate = eta_replace_rate
        self.eta_lower_bound = eta_lower_bound
        self.eta_upper_bound = eta_upper_bound
        self.plastic_a_init_mean = plastic_a_init_mean
        self.plastic_a_init_std = plastic_a_init_std
        self.plastic_a_mutate_power = plastic_a_mutate_power
        self.plastic_a_mutate_rate = plastic_a_mutate_rate
        self.plastic_a_replace_rate = plastic_a_replace_rate
        self.plastic_a_lower_bound = plastic_a_lower_bound
        self.plastic_a_upper_bound = plastic_a_upper_bound
        self.plastic_d_init_mean = plastic_d_init_mean
        self.plastic_d_init_std = plastic_d_init_std
        self.plastic_d_mutate_power = plastic_d_mutate_power
        self.plastic_d_mutate_rate = plastic_d_mutate_rate
        self.plastic_d_replace_rate = plastic_d_replace_rate
        self.plastic_d_lower_bound = plastic_d_lower_bound
        self.plastic_d_upper_bound = plastic_d_upper_bound
        self.plastic_mode = plastic_mode
        self.fixed_plastic_d = fixed_plastic_d

    def new_zero_attrs(self, state: State) -> jnp.ndarray:
        del state
        return jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    def new_identity_attrs(self, state: State) -> jnp.ndarray:
        del state
        return jnp.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    def new_random_attrs(self, state: State, randkey: jax.Array) -> jnp.ndarray:
        del state
        weight_key, eta_key, plastic_a_key, plastic_d_key = jax.random.split(randkey, 4)
        weight = jax.random.normal(weight_key, ()) * self.weight_init_std
        weight = jnp.clip(weight, self.weight_lower_bound, self.weight_upper_bound)
        if self.plastic_mode is None:
            eta = 0.0
        else:
            eta = jax.random.normal(eta_key, ()) * self.eta_init_std + self.eta_init_mean
            eta = jnp.clip(eta, self.eta_lower_bound, self.eta_upper_bound)
        if self.plastic_mode == "ad":
            plastic_a = jax.random.normal(plastic_a_key, ()) * self.plastic_a_init_std + self.plastic_a_init_mean
            plastic_a = jnp.clip(plastic_a, self.plastic_a_lower_bound, self.plastic_a_upper_bound)
            if self.fixed_plastic_d is None:
                plastic_d = jax.random.normal(plastic_d_key, ()) * self.plastic_d_init_std + self.plastic_d_init_mean
                plastic_d = jnp.clip(plastic_d, self.plastic_d_lower_bound, self.plastic_d_upper_bound)
            else:
                plastic_d = float(self.fixed_plastic_d)
        else:
            plastic_a = 1.0
            plastic_d = 0.0
        return jnp.array([weight, 1.0, eta, plastic_a, 0.0, 0.0, plastic_d], dtype=jnp.float32)

    def mutate(self, state: State, randkey: jax.Array, attrs: jnp.ndarray) -> jnp.ndarray:
        del state
        k1, k2, k3, k4, k5 = jax.random.split(randkey, 5)
        weight, enabled, eta, plastic_a, plastic_b, plastic_c, plastic_d = attrs
        weight = mutate_float(
            k1,
            weight,
            self.weight_init_mean,
            self.weight_init_std,
            self.weight_mutate_power,
            self.weight_mutate_rate,
            self.weight_replace_rate,
        )
        weight = jnp.clip(weight, self.weight_lower_bound, self.weight_upper_bound)
        toggle = jax.random.uniform(k2, ()) < self.enabled_toggle_rate
        enabled = jnp.where(toggle, 1.0 - enabled, enabled)
        enabled = jnp.where(enabled > 0.5, 1.0, 0.0)
        if self.plastic_mode is None:
            eta = 0.0
        else:
            eta = mutate_float(
                k3,
                eta,
                self.eta_init_mean,
                self.eta_init_std,
                self.eta_mutate_power,
                self.eta_mutate_rate,
                self.eta_replace_rate,
            )
            eta = jnp.clip(eta, self.eta_lower_bound, self.eta_upper_bound)
        if self.plastic_mode == "ad":
            plastic_a = mutate_float(
                k4,
                plastic_a,
                self.plastic_a_init_mean,
                self.plastic_a_init_std,
                self.plastic_a_mutate_power,
                self.plastic_a_mutate_rate,
                self.plastic_a_replace_rate,
            )
            plastic_a = jnp.clip(plastic_a, self.plastic_a_lower_bound, self.plastic_a_upper_bound)
            if self.fixed_plastic_d is None:
                plastic_d = mutate_float(
                    k5,
                    plastic_d,
                    self.plastic_d_init_mean,
                    self.plastic_d_init_std,
                    self.plastic_d_mutate_power,
                    self.plastic_d_mutate_rate,
                    self.plastic_d_replace_rate,
                )
                plastic_d = jnp.clip(plastic_d, self.plastic_d_lower_bound, self.plastic_d_upper_bound)
            else:
                plastic_d = float(self.fixed_plastic_d)
        else:
            plastic_a = 1.0
            plastic_d = 0.0
        plastic_b = 0.0
        plastic_c = 0.0
        return jnp.array([weight, enabled, eta, plastic_a, plastic_b, plastic_c, plastic_d], dtype=jnp.float32)

    def distance(self, state: State, attrs1: jnp.ndarray, attrs2: jnp.ndarray) -> jnp.ndarray:
        del state
        return (
            jnp.abs(attrs1[0] - attrs2[0])
            + (attrs1[1] != attrs2[1])
            + jnp.abs(attrs1[2] - attrs2[2])
            + jnp.abs(attrs1[3] - attrs2[3])
            + jnp.abs(attrs1[4] - attrs2[4])
            + jnp.abs(attrs1[5] - attrs2[5])
            + jnp.abs(attrs1[6] - attrs2[6])
        )

    def forward(self, state: State, attrs: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        del state
        weight, enabled, *_plastic = attrs
        safe_weight = jnp.where(jnp.isnan(weight), 0.0, weight)
        safe_inputs = jnp.where(jnp.isnan(inputs), 0.0, inputs)
        return safe_inputs * safe_weight * enabled

    def to_dict(self, state: State, conn: jnp.ndarray) -> dict[str, Any]:
        del state
        return {
            "in": int(conn[0]),
            "out": int(conn[1]),
            "historical_marker": int(conn[2]),
            "weight": float(conn[3]),
            "enabled": bool(conn[4] > 0.5),
            "eta": float(conn[5]) if conn.shape[0] > 5 else 0.0,
            "plastic_a": float(conn[6]) if conn.shape[0] > 6 else 1.0,
            "plastic_b": float(conn[7]) if conn.shape[0] > 7 else 0.0,
            "plastic_c": float(conn[8]) if conn.shape[0] > 8 else 0.0,
            "plastic_d": float(conn[9]) if conn.shape[0] > 9 else 0.0,
        }

    def repr(self, state: State, conn: jnp.ndarray, precision: int = 2, idx_width: int = 3, func_width: int = 8) -> str:
        del state, func_width
        return (
            f"EnabledOriginConn(in={int(conn[0]):<{idx_width}}, out={int(conn[1]):<{idx_width}}, "
            f"hist={int(conn[2]):<{idx_width}}, weight={float(conn[3]):.{precision}f}, "
            f"enabled={bool(conn[4] > 0.5)}, eta={float(conn[5]) if conn.shape[0] > 5 else 0.0:.{precision}f}, "
            f"A={float(conn[6]) if conn.shape[0] > 6 else 1.0:.{precision}f}, "
            f"D={float(conn[9]) if conn.shape[0] > 9 else 0.0:.{precision}f})"
        )

    def sympy_func(self, state: State, conn_dict: dict[str, Any], inputs: Any, precision: int | None = None) -> tuple[Any, dict[Any, Any]]:
        del state, precision
        weight = sp.symbols(f"c_{conn_dict['in']}_{conn_dict['out']}_w")
        enabled = 1 if conn_dict["enabled"] else 0
        return inputs * weight * enabled, {weight: conn_dict["weight"]}


@dataclass(frozen=True)
class LineageInfo:
    winner_indices: tuple[int, ...]
    loser_indices: tuple[int, ...]
    elite_mask: tuple[bool, ...]


class TrackedNEAT(NEAT):
    def step(self, state: State, fitness: jnp.ndarray) -> tuple[State, LineageInfo]:
        state = state.update(generation=state.generation + 1)
        state, winner, loser, elite_mask = self.species_controller.update_species(state, fitness)
        lineage = LineageInfo(
            winner_indices=tuple(int(v) for v in np.asarray(jax.device_get(winner)).tolist()),
            loser_indices=tuple(int(v) for v in np.asarray(jax.device_get(loser)).tolist()),
            elite_mask=tuple(bool(v) for v in np.asarray(jax.device_get(elite_mask)).tolist()),
        )
        state = self._create_next_generation(state, winner, loser, elite_mask)
        state = self.species_controller.speciate(state, self.genome.execute_distance)
        return state, lineage


class TensorNEATAdapter:
    def __init__(self, config: AppConfig, num_inputs: int, num_outputs: int) -> None:
        self.config = config
        mutation_cfg = config.mutation
        evolution_cfg = config.evolution
        force_alpha_zero = config.run.variant == "stateless"
        enable_two_timescale = is_stateful_v2_variant(config.run.variant)
        plastic_mode = plastic_mode_for_variant(config.run.variant)
        fixed_plastic_d = plastic_fixed_d_for_variant(config.run.variant)
        plastic_d_lower_bound, plastic_d_upper_bound = plastic_d_bounds_for_variant(
            config.run.variant,
            default_lower_bound=mutation_cfg.plastic_d_lower_bound,
            default_upper_bound=mutation_cfg.plastic_d_upper_bound,
        )
        plastic_d_init_mean, plastic_d_init_std = plastic_d_init_for_variant(
            config.run.variant,
            default_mean=mutation_cfg.plastic_d_init_mean,
            default_std=mutation_cfg.plastic_d_init_std,
        )
        plastic_d_mutate_power, plastic_d_mutate_rate, plastic_d_replace_rate = plastic_d_mutation_for_variant(
            config.run.variant,
            default_power=mutation_cfg.plastic_d_mutate_power,
            default_rate=mutation_cfg.plastic_d_mutate_rate,
            default_replace_rate=mutation_cfg.plastic_d_replace_rate,
        )

        node_gene = AlphaNode(
            bias_init_std=mutation_cfg.bias_init_std,
            bias_mutate_power=mutation_cfg.bias_mutate_power,
            bias_mutate_rate=mutation_cfg.bias_mutate_rate,
            bias_replace_rate=mutation_cfg.bias_replace_rate,
            bias_lower_bound=mutation_cfg.bias_lower_bound,
            bias_upper_bound=mutation_cfg.bias_upper_bound,
            alpha_init_mean=mutation_cfg.alpha_init_mean,
            alpha_init_std=mutation_cfg.alpha_init_std,
            alpha_mutate_power=mutation_cfg.alpha_mutate_power,
            alpha_mutate_rate=mutation_cfg.alpha_mutate_rate,
            alpha_replace_rate=mutation_cfg.alpha_replace_rate,
            alpha_slow_init_mean=mutation_cfg.alpha_slow_init_mean,
            alpha_slow_init_std=mutation_cfg.alpha_slow_init_std,
            alpha_slow_mutate_power=mutation_cfg.alpha_slow_mutate_power,
            alpha_slow_mutate_rate=mutation_cfg.alpha_slow_mutate_rate,
            alpha_slow_replace_rate=mutation_cfg.alpha_slow_replace_rate,
            slow_input_gain_init_mean=mutation_cfg.slow_input_gain_init_mean,
            slow_input_gain_init_std=mutation_cfg.slow_input_gain_init_std,
            slow_input_gain_mutate_power=mutation_cfg.slow_input_gain_mutate_power,
            slow_input_gain_mutate_rate=mutation_cfg.slow_input_gain_mutate_rate,
            slow_input_gain_replace_rate=mutation_cfg.slow_input_gain_replace_rate,
            slow_input_gain_lower_bound=mutation_cfg.slow_input_gain_lower_bound,
            slow_input_gain_upper_bound=mutation_cfg.slow_input_gain_upper_bound,
            slow_output_gain_init_mean=mutation_cfg.slow_output_gain_init_mean,
            slow_output_gain_init_std=mutation_cfg.slow_output_gain_init_std,
            slow_output_gain_mutate_power=mutation_cfg.slow_output_gain_mutate_power,
            slow_output_gain_mutate_rate=mutation_cfg.slow_output_gain_mutate_rate,
            slow_output_gain_replace_rate=mutation_cfg.slow_output_gain_replace_rate,
            slow_output_gain_lower_bound=mutation_cfg.slow_output_gain_lower_bound,
            slow_output_gain_upper_bound=mutation_cfg.slow_output_gain_upper_bound,
            enable_two_timescale=enable_two_timescale,
            force_alpha_zero=force_alpha_zero,
        )
        conn_gene = EnabledOriginConn(
            weight_init_std=mutation_cfg.weight_init_std,
            weight_mutate_power=mutation_cfg.weight_mutate_power,
            weight_mutate_rate=mutation_cfg.weight_mutate_rate,
            weight_replace_rate=mutation_cfg.weight_replace_rate,
            weight_lower_bound=mutation_cfg.weight_lower_bound,
            weight_upper_bound=mutation_cfg.weight_upper_bound,
            enabled_toggle_rate=mutation_cfg.enabled_toggle_rate,
            eta_init_mean=mutation_cfg.eta_init_mean,
            eta_init_std=mutation_cfg.eta_init_std,
            eta_mutate_power=mutation_cfg.eta_mutate_power,
            eta_mutate_rate=mutation_cfg.eta_mutate_rate,
            eta_replace_rate=mutation_cfg.eta_replace_rate,
            eta_lower_bound=mutation_cfg.eta_lower_bound,
            eta_upper_bound=mutation_cfg.eta_upper_bound,
            plastic_a_init_mean=mutation_cfg.plastic_a_init_mean,
            plastic_a_init_std=mutation_cfg.plastic_a_init_std,
            plastic_a_mutate_power=mutation_cfg.plastic_a_mutate_power,
            plastic_a_mutate_rate=mutation_cfg.plastic_a_mutate_rate,
            plastic_a_replace_rate=mutation_cfg.plastic_a_replace_rate,
            plastic_a_lower_bound=mutation_cfg.plastic_a_lower_bound,
            plastic_a_upper_bound=mutation_cfg.plastic_a_upper_bound,
            plastic_d_init_mean=plastic_d_init_mean,
            plastic_d_init_std=plastic_d_init_std,
            plastic_d_mutate_power=plastic_d_mutate_power,
            plastic_d_mutate_rate=plastic_d_mutate_rate,
            plastic_d_replace_rate=plastic_d_replace_rate,
            plastic_d_lower_bound=plastic_d_lower_bound,
            plastic_d_upper_bound=plastic_d_upper_bound,
            plastic_mode=plastic_mode,
            fixed_plastic_d=fixed_plastic_d,
        )
        mutation = DefaultMutation(
            conn_add=mutation_cfg.conn_add,
            conn_delete=mutation_cfg.conn_delete,
            node_add=mutation_cfg.node_add,
            node_delete=mutation_cfg.node_delete,
        )

        genome = RecurrentGenome(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            max_nodes=evolution_cfg.max_nodes,
            max_conns=evolution_cfg.max_conns,
            node_gene=node_gene,
            conn_gene=conn_gene,
            mutation=mutation,
            crossover=DefaultCrossover(),
            distance=DefaultDistance(),
            activate_time=config.task.activation_steps,
        )

        self.algorithm = TrackedNEAT(
            genome=genome,
            pop_size=evolution_cfg.population_size,
            species_size=evolution_cfg.species_size,
            max_stagnation=evolution_cfg.max_stagnation,
            species_elitism=evolution_cfg.species_elitism,
            spawn_number_change_rate=evolution_cfg.spawn_number_change_rate,
            genome_elitism=evolution_cfg.genome_elitism,
            survival_threshold=evolution_cfg.survival_threshold,
            min_species_size=evolution_cfg.min_species_size,
            compatibility_threshold=evolution_cfg.compatibility_threshold,
            species_number_calculate_by=evolution_cfg.species_number_calculate_by,
        )

    @property
    def genome(self) -> RecurrentGenome:
        return self.algorithm.genome

    @property
    def population_size(self) -> int:
        return self.algorithm.pop_size

    def initialize(self, seed: int) -> State:
        return self.algorithm.setup(State(randkey=jax.random.PRNGKey(seed)))

    @staticmethod
    def serialize_state(state: State) -> str:
        return base64.b64encode(pickle.dumps(state)).decode("ascii")

    @staticmethod
    def deserialize_state(blob: str) -> State:
        state = pickle.loads(base64.b64decode(blob.encode("ascii")))
        return _upgrade_legacy_conn_state(state)

    def ask(self, state: State) -> tuple[Any, Any]:
        return self.algorithm.ask(state)

    def advance(
        self,
        state: State,
        fitness_scores: np.ndarray,
        candidate_ids: list[str],
    ) -> tuple[State, list[list[str]]]:
        next_state, lineage = self.algorithm.step(
            state,
            jnp.asarray(fitness_scores, dtype=jnp.float32),
        )
        parent_ids: list[list[str]] = []
        for winner_idx, loser_idx, elite in zip(
            lineage.winner_indices,
            lineage.loser_indices,
            lineage.elite_mask,
            strict=True,
        ):
            winner_id = candidate_ids[winner_idx]
            loser_id = candidate_ids[loser_idx]
            if elite or winner_id == loser_id:
                parent_ids.append([winner_id])
            else:
                parent_ids.append([winner_id, loser_id])
        return next_state, parent_ids

    def sync_population_state(self, state: State, pop_nodes: Any, pop_conns: Any) -> State:
        return state.update(
            pop_nodes=jnp.asarray(pop_nodes, dtype=jnp.float32),
            pop_conns=jnp.asarray(pop_conns, dtype=jnp.float32),
        )

    def spawn_child(
        self,
        state: State,
        parent_one_nodes: Any,
        parent_one_conns: Any,
        parent_two_nodes: Any,
        parent_two_conns: Any,
        active_population_nodes: Any,
        active_population_conns: Any,
    ) -> tuple[State, Any, Any]:
        synced_state = self.sync_population_state(state, active_population_nodes, active_population_conns)
        k1, k2, randkey = jax.random.split(synced_state.randkey, 3)
        new_node_key = self._next_node_key(active_population_nodes)
        new_conn_keys = self._next_conn_keys(active_population_conns)
        child_nodes, child_conns = self.genome.execute_crossover(
            synced_state,
            k1,
            jnp.asarray(parent_one_nodes, dtype=jnp.float32),
            jnp.asarray(parent_one_conns, dtype=jnp.float32),
            jnp.asarray(parent_two_nodes, dtype=jnp.float32),
            jnp.asarray(parent_two_conns, dtype=jnp.float32),
        )
        child_nodes, child_conns = self.genome.execute_mutation(
            synced_state,
            k2,
            child_nodes,
            child_conns,
            new_node_key,
            new_conn_keys,
        )
        return synced_state.update(randkey=randkey), child_nodes, child_conns

    @staticmethod
    def _next_node_key(active_population_nodes: Any) -> jnp.ndarray:
        node_ids = np.asarray(active_population_nodes, dtype=np.float32)[..., 0]
        if np.isnan(node_ids).all():
            return jnp.asarray(0.0, dtype=jnp.float32)
        max_node_key = np.nanmax(node_ids)
        return jnp.asarray(float(max_node_key + 1.0), dtype=jnp.float32)

    @staticmethod
    def _next_conn_keys(active_population_conns: Any) -> jnp.ndarray:
        conn_markers = np.asarray(active_population_conns, dtype=np.float32)[..., 2]
        if np.isnan(conn_markers).all():
            start = 0.0
        else:
            start = float(np.nanmax(conn_markers) + 1.0)
        return jnp.asarray(np.arange(start, start + 3.0, dtype=np.float32))


def _upgrade_legacy_conn_state(state: State) -> State:
    updates: dict[str, Any] = {}
    for key in state.registered_keys():
        value = getattr(state, key)
        if isinstance(value, State):
            updates[key] = _upgrade_legacy_conn_state(value)
            continue
        if key.endswith("nodes"):
            updates[key] = _pad_legacy_node_array(value)
        if key.endswith("conns"):
            updates[key] = _pad_legacy_conn_array(value)
    return state.update(**updates) if updates else state


def _pad_legacy_node_array(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0 or shape[-1] != 3:
        return value
    defaults = jnp.array([0.0, 0.0, 0.0], dtype=value.dtype)
    defaults = jnp.broadcast_to(defaults, (*shape[:-1], defaults.shape[0]))
    return jnp.concatenate([value, defaults], axis=-1)


def _pad_legacy_conn_array(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0 or shape[-1] not in {5, 6}:
        return value
    if shape[-1] == 5:
        defaults = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=value.dtype)
    else:
        defaults = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=value.dtype)
    defaults = jnp.broadcast_to(defaults, (*shape[:-1], defaults.shape[0]))
    return jnp.concatenate([value, defaults], axis=-1)
