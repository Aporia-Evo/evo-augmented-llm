from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from db.models import ActiveCandidateRecord
from evolve.candidate_registry import ActiveCandidateBundle, CandidateRegistry
from evolve.genome_codec import arrays_to_genome_model, genome_model_to_blob
from evolve.tensorneat_adapter import TensorNEATAdapter
from utils.serialization import utc_now_iso


@dataclass(frozen=True)
class OffspringSpawn:
    record: ActiveCandidateRecord
    genome_bundle: ActiveCandidateBundle
    state: object


class RTNEATScheduler:
    def __init__(self, adapter: TensorNEATAdapter, variant: str) -> None:
        self.adapter = adapter
        self.variant = variant

    def spawn_offspring(
        self,
        *,
        state: object,
        registry: CandidateRegistry,
        run_id: str,
        slot_index: int,
        parent_ids: list[str],
        birth_step: int,
    ) -> OffspringSpawn:
        if not parent_ids:
            raise ValueError("At least one parent is required to spawn offspring.")

        first_parent = registry.get(parent_ids[0])
        second_parent = registry.get(parent_ids[1] if len(parent_ids) > 1 else parent_ids[0])
        active_nodes, active_conns = registry.population_arrays(self.adapter.genome)
        parent_one_nodes, parent_one_conns = self._bundle_arrays(registry, first_parent.record.candidate_id)
        parent_two_nodes, parent_two_conns = self._bundle_arrays(registry, second_parent.record.candidate_id)
        next_state, child_nodes, child_conns = self.adapter.spawn_child(
            state=state,
            parent_one_nodes=parent_one_nodes,
            parent_one_conns=parent_one_conns,
            parent_two_nodes=parent_two_nodes,
            parent_two_conns=parent_two_conns,
            active_population_nodes=active_nodes,
            active_population_conns=active_conns,
        )
        genome = arrays_to_genome_model(self.adapter.genome, child_nodes, child_conns)
        created_at = utc_now_iso()
        candidate_id = f"{run_id}-online-s{birth_step:06d}-c{slot_index:04d}"
        record = ActiveCandidateRecord(
            candidate_id=candidate_id,
            run_id=run_id,
            slot_index=slot_index,
            variant=self.variant,
            genome_blob=genome_model_to_blob(genome),
            status="created",
            rolling_score=0.0,
            eval_count=0,
            birth_step=birth_step,
            last_eval_at=None,
            parent_ids=list(parent_ids),
            created_at=created_at,
        )
        bundle = ActiveCandidateBundle(record=record, genome=genome)
        return OffspringSpawn(record=record, genome_bundle=bundle, state=next_state)

    def _bundle_arrays(
        self,
        registry: CandidateRegistry,
        candidate_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        bundle = registry.get(candidate_id)
        active_nodes, active_conns = registry.population_arrays(self.adapter.genome)
        slot_index = bundle.record.slot_index
        return active_nodes[slot_index], active_conns[slot_index]
