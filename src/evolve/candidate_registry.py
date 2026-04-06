from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from db.models import ActiveCandidateRecord
from evolve.genome_codec import GenomeModel, arrays_to_genome_model, genome_model_from_blob, genome_model_to_arrays


@dataclass(frozen=True)
class ActiveCandidateBundle:
    record: ActiveCandidateRecord
    genome: GenomeModel


class CandidateRegistry:
    def __init__(self, bundles: list[ActiveCandidateBundle]) -> None:
        self._bundles_by_candidate_id = {bundle.record.candidate_id: bundle for bundle in bundles}
        self._bundles_by_slot = {bundle.record.slot_index: bundle for bundle in bundles}

    @classmethod
    def from_seed_population(
        cls,
        *,
        genome_template: object,
        run_id: str,
        variant: str,
        pop_nodes: np.ndarray,
        pop_conns: np.ndarray,
        birth_step: int,
        created_at: str,
    ) -> "CandidateRegistry":
        bundles: list[ActiveCandidateBundle] = []
        for slot_index in range(pop_nodes.shape[0]):
            genome = arrays_to_genome_model(genome_template, pop_nodes[slot_index], pop_conns[slot_index])
            record = ActiveCandidateRecord(
                candidate_id=f"{run_id}-online-s{birth_step:06d}-c{slot_index:04d}",
                run_id=run_id,
                slot_index=slot_index,
                variant=variant,
                genome_blob="",
                status="created",
                rolling_score=0.0,
                eval_count=0,
                birth_step=birth_step,
                last_eval_at=None,
                parent_ids=[],
                created_at=created_at,
            )
            record = ActiveCandidateRecord(**{**record.__dict__, "genome_blob": _blob_for_genome(genome)})
            bundles.append(ActiveCandidateBundle(record=record, genome=genome))
        return cls(bundles)

    @classmethod
    def from_records(cls, records: list[ActiveCandidateRecord]) -> "CandidateRegistry":
        bundles = [
            ActiveCandidateBundle(
                record=record,
                genome=genome_model_from_blob(record.genome_blob),
            )
            for record in sorted(records, key=lambda item: item.slot_index)
        ]
        return cls(bundles)

    @property
    def bundles(self) -> list[ActiveCandidateBundle]:
        return [self._bundles_by_slot[index] for index in sorted(self._bundles_by_slot)]

    def list_records(self) -> list[ActiveCandidateRecord]:
        return [bundle.record for bundle in self.bundles]

    def get(self, candidate_id: str) -> ActiveCandidateBundle:
        return self._bundles_by_candidate_id[candidate_id]

    def get_slot(self, slot_index: int) -> ActiveCandidateBundle:
        return self._bundles_by_slot[slot_index]

    def update_record(self, record: ActiveCandidateRecord) -> None:
        bundle = self._bundles_by_candidate_id[record.candidate_id]
        updated = ActiveCandidateBundle(record=record, genome=bundle.genome)
        self._bundles_by_candidate_id[record.candidate_id] = updated
        self._bundles_by_slot[record.slot_index] = updated

    def replace_slot(self, slot_index: int, record: ActiveCandidateRecord, genome: GenomeModel) -> None:
        previous = self._bundles_by_slot.get(slot_index)
        if previous is not None:
            self._bundles_by_candidate_id.pop(previous.record.candidate_id, None)
        bundle = ActiveCandidateBundle(record=record, genome=genome)
        self._bundles_by_slot[slot_index] = bundle
        self._bundles_by_candidate_id[record.candidate_id] = bundle

    def population_arrays(self, genome_template: object) -> tuple[np.ndarray, np.ndarray]:
        slots = sorted(self._bundles_by_slot)
        node_arrays: list[np.ndarray] = []
        conn_arrays: list[np.ndarray] = []
        for slot_index in slots:
            nodes, conns = genome_model_to_arrays(genome_template, self._bundles_by_slot[slot_index].genome)
            node_arrays.append(nodes)
            conn_arrays.append(conns)
        return np.stack(node_arrays), np.stack(conn_arrays)


def _blob_for_genome(genome: GenomeModel) -> str:
    from evolve.genome_codec import genome_model_to_blob

    return genome_model_to_blob(genome)
