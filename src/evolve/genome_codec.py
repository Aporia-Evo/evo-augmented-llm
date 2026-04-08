from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NodeGeneModel:
    node_id: int
    bias: float
    alpha: float
    is_input: bool = False
    is_output: bool = False
    alpha_slow: float = 0.0
    slow_input_gain: float = 0.0
    slow_output_gain: float = 0.0
    content_w_key: float = 0.0
    content_b_key: float = 0.0
    content_w_query: float = 0.0
    content_b_query: float = 0.0
    content_temperature: float = 1.0
    content_b_match: float = 0.0


@dataclass(frozen=True)
class ConnectionGeneModel:
    in_id: int
    out_id: int
    historical_marker: int
    weight: float
    enabled: bool
    eta: float = 0.0
    plastic_a: float = 1.0
    plastic_b: float = 0.0
    plastic_c: float = 0.0
    plastic_d: float = 0.0


@dataclass(frozen=True)
class GenomeModel:
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    nodes: tuple[NodeGeneModel, ...]
    connections: tuple[ConnectionGeneModel, ...]


def _non_nan_rows(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return array
    return array[~np.isnan(array[:, 0])]


def arrays_to_genome_model(genome_template: Any, nodes: Any, conns: Any) -> GenomeModel:
    nodes_np = _non_nan_rows(np.asarray(nodes, dtype=np.float32))
    conns_np = _non_nan_rows(np.asarray(conns, dtype=np.float32))
    input_ids = tuple(int(v) for v in np.asarray(genome_template.input_idx).tolist())
    output_ids = tuple(int(v) for v in np.asarray(genome_template.output_idx).tolist())
    input_id_set = set(input_ids)
    output_id_set = set(output_ids)

    node_models = tuple(
        sorted(
            (
                NodeGeneModel(
                    node_id=int(row[0]),
                    bias=float(row[1]),
                    alpha=float(np.clip(row[2], 0.0, 1.0)),
                    alpha_slow=float(np.clip(row[3], 0.0, 1.0)) if row.shape[0] > 3 and not np.isnan(row[3]) else 0.0,
                    slow_input_gain=float(row[4]) if row.shape[0] > 4 and not np.isnan(row[4]) else 0.0,
                    slow_output_gain=float(row[5]) if row.shape[0] > 5 and not np.isnan(row[5]) else 0.0,
                    content_w_key=float(row[6]) if row.shape[0] > 6 and not np.isnan(row[6]) else 0.0,
                    content_b_key=float(row[7]) if row.shape[0] > 7 and not np.isnan(row[7]) else 0.0,
                    content_w_query=float(row[8]) if row.shape[0] > 8 and not np.isnan(row[8]) else 0.0,
                    content_b_query=float(row[9]) if row.shape[0] > 9 and not np.isnan(row[9]) else 0.0,
                    content_temperature=float(row[10]) if row.shape[0] > 10 and not np.isnan(row[10]) else 1.0,
                    content_b_match=float(row[11]) if row.shape[0] > 11 and not np.isnan(row[11]) else 0.0,
                    is_input=int(row[0]) in input_id_set,
                    is_output=int(row[0]) in output_id_set,
                )
                for row in nodes_np
            ),
            key=lambda node: node.node_id,
        )
    )

    conn_models = tuple(
        sorted(
            (
                ConnectionGeneModel(
                    in_id=int(row[0]),
                    out_id=int(row[1]),
                    historical_marker=int(row[2]),
                    weight=float(row[3]),
                    enabled=bool(row[4] > 0.5),
                    eta=float(row[5]) if row.shape[0] > 5 and not np.isnan(row[5]) else 0.0,
                    plastic_a=float(row[6]) if row.shape[0] > 6 and not np.isnan(row[6]) else 1.0,
                    plastic_b=float(row[7]) if row.shape[0] > 7 and not np.isnan(row[7]) else 0.0,
                    plastic_c=float(row[8]) if row.shape[0] > 8 and not np.isnan(row[8]) else 0.0,
                    plastic_d=float(row[9]) if row.shape[0] > 9 and not np.isnan(row[9]) else 0.0,
                )
                for row in conns_np
            ),
            key=lambda conn: (conn.out_id, conn.in_id, conn.historical_marker),
        )
    )

    return GenomeModel(
        input_ids=input_ids,
        output_ids=output_ids,
        nodes=node_models,
        connections=conn_models,
    )


def genome_model_to_blob(genome: GenomeModel) -> str:
    return json.dumps(asdict(genome), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def genome_model_from_blob(blob: str) -> GenomeModel:
    payload = json.loads(blob)
    return GenomeModel(
        input_ids=tuple(payload["input_ids"]),
        output_ids=tuple(payload["output_ids"]),
        nodes=tuple(NodeGeneModel(**node) for node in payload["nodes"]),
        connections=tuple(
            ConnectionGeneModel(
                in_id=conn["in_id"],
                out_id=conn["out_id"],
                historical_marker=conn["historical_marker"],
                weight=conn["weight"],
                enabled=conn["enabled"],
                eta=conn.get("eta", 0.0),
                plastic_a=conn.get("plastic_a", 1.0),
                plastic_b=conn.get("plastic_b", 0.0),
                plastic_c=conn.get("plastic_c", 0.0),
                plastic_d=conn.get("plastic_d", 0.0),
            )
            for conn in payload["connections"]
        ),
    )


def genome_model_to_arrays(genome_template: Any, genome: GenomeModel) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.full((genome_template.max_nodes, genome_template.node_gene.length), np.nan, dtype=np.float32)
    conns = np.full((genome_template.max_conns, genome_template.conn_gene.length), np.nan, dtype=np.float32)

    for index, node in enumerate(sorted(genome.nodes, key=lambda item: item.node_id)):
        nodes[index, 0] = float(node.node_id)
        nodes[index, 1] = float(node.bias)
        nodes[index, 2] = float(np.clip(node.alpha, 0.0, 1.0))
        if nodes.shape[1] > 3:
            nodes[index, 3] = float(np.clip(node.alpha_slow, 0.0, 1.0))
        if nodes.shape[1] > 4:
            nodes[index, 4] = float(node.slow_input_gain)
        if nodes.shape[1] > 5:
            nodes[index, 5] = float(node.slow_output_gain)
        if nodes.shape[1] > 6:
            nodes[index, 6] = float(node.content_w_key)
        if nodes.shape[1] > 7:
            nodes[index, 7] = float(node.content_b_key)
        if nodes.shape[1] > 8:
            nodes[index, 8] = float(node.content_w_query)
        if nodes.shape[1] > 9:
            nodes[index, 9] = float(node.content_b_query)
        if nodes.shape[1] > 10:
            nodes[index, 10] = float(node.content_temperature)
        if nodes.shape[1] > 11:
            nodes[index, 11] = float(node.content_b_match)

    for index, conn in enumerate(
        sorted(
            genome.connections,
            key=lambda item: (item.out_id, item.in_id, item.historical_marker),
        )
    ):
        conns[index, 0] = float(conn.in_id)
        conns[index, 1] = float(conn.out_id)
        conns[index, 2] = float(conn.historical_marker)
        conns[index, 3] = float(conn.weight)
        conns[index, 4] = 1.0 if conn.enabled else 0.0
        if conns.shape[1] > 5:
            conns[index, 5] = float(conn.eta)
        if conns.shape[1] > 6:
            conns[index, 6] = float(conn.plastic_a)
        if conns.shape[1] > 7:
            conns[index, 7] = float(conn.plastic_b)
        if conns.shape[1] > 8:
            conns[index, 8] = float(conn.plastic_c)
        if conns.shape[1] > 9:
            conns[index, 9] = float(conn.plastic_d)

    return nodes, conns
