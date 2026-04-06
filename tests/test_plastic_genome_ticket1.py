from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from config import AppConfig, RunConfig
from evolve.evaluator import build_evaluator
from evolve.genome_codec import arrays_to_genome_model, genome_model_from_blob, genome_model_to_blob
from evolve.tensorneat_adapter import TensorNEATAdapter
from tensorneat.common import State


def test_legacy_genome_blob_defaults_connection_eta_to_zero() -> None:
    legacy_blob = (
        '{"connections":[{"enabled":true,"historical_marker":3,"in_id":0,"out_id":2,"weight":1.25}],'
        '"input_ids":[0,1],"nodes":[{"alpha":0.3,"bias":0.0,"is_input":true,"is_output":false,"node_id":0},'
        '{"alpha":0.3,"bias":0.0,"is_input":true,"is_output":false,"node_id":1},'
        '{"alpha":0.5,"bias":0.2,"is_input":false,"is_output":true,"node_id":2}],"output_ids":[2]}'
    )

    genome = genome_model_from_blob(legacy_blob)

    assert len(genome.connections) == 1
    assert genome.connections[0].eta == 0.0
    assert genome.connections[0].plastic_a == 1.0
    assert genome.connections[0].plastic_b == 0.0
    assert genome.connections[0].plastic_c == 0.0
    assert genome.connections[0].plastic_d == 0.0
    assert '"eta":0.0' in genome_model_to_blob(genome)


def test_adapter_deserialize_state_upgrades_legacy_connection_arrays() -> None:
    legacy_state = State(
        randkey=jax.random.PRNGKey(0),
        pop_nodes=jnp.zeros((2, 3, 3), dtype=jnp.float32),
        pop_conns=jnp.zeros((2, 4, 5), dtype=jnp.float32),
        species=State(
            center_nodes=jnp.zeros((1, 3, 3), dtype=jnp.float32),
            center_conns=jnp.zeros((1, 4, 5), dtype=jnp.float32),
        ),
    )

    restored = TensorNEATAdapter.deserialize_state(TensorNEATAdapter.serialize_state(legacy_state))

    assert restored.pop_nodes.shape[-1] == 6
    assert restored.species.center_nodes.shape[-1] == 6
    assert np.allclose(np.asarray(restored.pop_nodes)[..., 3], 0.0)
    assert np.allclose(np.asarray(restored.pop_nodes)[..., 4], 0.0)
    assert np.allclose(np.asarray(restored.pop_nodes)[..., 5], 0.0)
    assert restored.pop_conns.shape[-1] == 10
    assert restored.species.center_conns.shape[-1] == 10
    assert np.allclose(np.asarray(restored.pop_conns)[..., 5], 0.0)
    assert np.allclose(np.asarray(restored.pop_conns)[..., 6], 1.0)
    assert np.allclose(np.asarray(restored.pop_conns)[..., 7], 0.0)
    assert np.allclose(np.asarray(restored.pop_conns)[..., 8], 0.0)
    assert np.allclose(np.asarray(restored.pop_conns)[..., 9], 0.0)
    assert np.allclose(np.asarray(restored.species.center_conns)[..., 5], 0.0)
    assert np.allclose(np.asarray(restored.species.center_conns)[..., 6], 1.0)


def test_adapter_deserialize_state_upgrades_v5a_connection_arrays_to_v5b_shape() -> None:
    legacy_state = State(
        randkey=jax.random.PRNGKey(0),
        pop_nodes=jnp.zeros((2, 3, 3), dtype=jnp.float32),
        pop_conns=jnp.zeros((2, 4, 6), dtype=jnp.float32),
        species=State(
            center_nodes=jnp.zeros((1, 3, 3), dtype=jnp.float32),
            center_conns=jnp.zeros((1, 4, 6), dtype=jnp.float32),
        ),
    )

    restored = TensorNEATAdapter.deserialize_state(TensorNEATAdapter.serialize_state(legacy_state))

    assert restored.pop_nodes.shape[-1] == 6
    assert restored.pop_conns.shape[-1] == 10
    assert np.allclose(np.asarray(restored.pop_conns)[..., 6], 1.0)
    assert np.allclose(np.asarray(restored.pop_conns)[..., 7], 0.0)
    assert np.allclose(np.asarray(restored.pop_conns)[..., 8], 0.0)
    assert np.allclose(np.asarray(restored.pop_conns)[..., 9], 0.0)


def test_existing_variants_keep_connection_eta_at_zero() -> None:
    config = AppConfig(run=replace(RunConfig(), seed=7, variant="stateful"))
    evaluator = build_evaluator(config.task)
    adapter = TensorNEATAdapter(config=config, num_inputs=evaluator.input_size, num_outputs=evaluator.output_size)

    state = adapter.initialize(config.run.seed)
    pop_nodes, pop_conns = adapter.ask(state)
    genome = arrays_to_genome_model(adapter.genome, pop_nodes[0], pop_conns[0])

    assert all(conn.eta == 0.0 for conn in genome.connections)


def test_plasticity_config_loads_custom_delta_weight_clamp() -> None:
    config = AppConfig.from_dict({"plasticity": {"delta_w_clamp": 0.25}})

    assert config.plasticity.delta_w_clamp == 0.25
