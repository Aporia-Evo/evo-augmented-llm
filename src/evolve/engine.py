from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from db.models import CandidateRecord, FitnessRecord
from evolve.evaluator import EvaluationResult
from evolve.genome_codec import GenomeModel, arrays_to_genome_model, genome_model_from_blob, genome_model_to_blob
from evolve.tensorneat_adapter import TensorNEATAdapter
from utils.serialization import utc_now_iso


class GenomeEvaluator(Protocol):
    input_size: int
    output_size: int

    def evaluate(self, genome: GenomeModel) -> EvaluationResult:
        ...


@dataclass(frozen=True)
class CandidateBundle:
    record: CandidateRecord
    genome: GenomeModel


@dataclass(frozen=True)
class PopulationSnapshot:
    run_id: str
    generation_id: int
    candidates: tuple[CandidateBundle, ...]

    @property
    def candidate_ids(self) -> list[str]:
        return [item.record.candidate_id for item in self.candidates]


class EvolutionEngine:
    def __init__(self, adapter: TensorNEATAdapter, evaluator: GenomeEvaluator) -> None:
        self.adapter = adapter
        self.evaluator = evaluator

    def prepare_population(
        self,
        run_id: str,
        generation_id: int,
        state: object,
        parent_ids: list[list[str]],
    ) -> PopulationSnapshot:
        pop_nodes, pop_conns = self.adapter.ask(state)
        bundles: list[CandidateBundle] = []
        created_at = utc_now_iso()

        for population_index in range(self.adapter.population_size):
            genome = arrays_to_genome_model(
                self.adapter.genome,
                pop_nodes[population_index],
                pop_conns[population_index],
            )
            candidate_id = f"{run_id}-g{generation_id:04d}-c{population_index:04d}"
            record = CandidateRecord(
                candidate_id=candidate_id,
                run_id=run_id,
                generation_id=generation_id,
                genome_blob=genome_model_to_blob(genome),
                status="created",
                parent_ids=list(parent_ids[population_index]) if parent_ids else [],
                created_at=created_at,
            )
            bundles.append(CandidateBundle(record=record, genome=genome))

        return PopulationSnapshot(
            run_id=run_id,
            generation_id=generation_id,
            candidates=tuple(bundles),
        )

    def population_from_records(
        self,
        run_id: str,
        generation_id: int,
        records: list[CandidateRecord],
    ) -> PopulationSnapshot:
        bundles = [
            CandidateBundle(
                record=record,
                genome=genome_model_from_blob(record.genome_blob),
            )
            for record in sorted(records, key=lambda item: item.candidate_id)
        ]
        return PopulationSnapshot(
            run_id=run_id,
            generation_id=generation_id,
            candidates=tuple(bundles),
        )

    def evaluate_population(self, population: PopulationSnapshot) -> list[FitnessRecord]:
        fitness_records: list[FitnessRecord] = []
        evaluated_at = utc_now_iso()

        for bundle in population.candidates:
            result = self.evaluator.evaluate(bundle.genome)
            fitness_records.append(
                FitnessRecord(
                    candidate_id=bundle.record.candidate_id,
                    run_id=bundle.record.run_id,
                    generation_id=bundle.record.generation_id,
                    score=result.score,
                    raw_metrics=result.raw_metrics,
                    evaluated_at=evaluated_at,
                )
            )

        return fitness_records
