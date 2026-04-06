from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from db.models import CandidateFeatureRecord, CandidateFeatureVectorRecord
from evolve.genome_codec import GenomeModel
from utils.serialization import stable_json_dumps


FEATURE_VERSION = "candidate-features-v8"
BOUND_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CandidateFeatureContext:
    candidate_id: str
    run_id: str
    benchmark_label: str
    task_name: str
    delay_steps: int
    variant: str
    seed: int
    generation: int
    final_max_score: float
    first_success_generation: int | None
    eta_lower_bound: float
    eta_upper_bound: float
    plastic_d_lower_bound: float
    curriculum_enabled: bool = False
    curriculum_phase_1_delays: str = ""
    curriculum_phase_2_delays: str = ""
    curriculum_switch_generation: int = 0
    curriculum_phase: str = "static"
    active_evaluation_delays: str = ""


def extract_candidate_features(
    genome: GenomeModel,
    raw_metrics: dict[str, Any] | None,
    context: CandidateFeatureContext,
) -> tuple[CandidateFeatureRecord, CandidateFeatureVectorRecord]:
    metrics = raw_metrics or {}
    dynamic_nodes = [node for node in genome.nodes if not node.is_input]
    enabled_connections = [conn for conn in genome.connections if conn.enabled]

    alpha_values = [float(node.alpha) for node in dynamic_nodes]
    eta_values = [float(conn.eta) for conn in enabled_connections]
    plastic_d_values = [float(conn.plastic_d) for conn in enabled_connections]

    mean_alpha, std_alpha = _mean_std(alpha_values)
    mean_eta, std_eta = _mean_std(eta_values)
    mean_plastic_d, std_plastic_d = _mean_std(plastic_d_values)

    feature = CandidateFeatureRecord(
        candidate_id=context.candidate_id,
        run_id=context.run_id,
        benchmark_label=context.benchmark_label,
        task_name=context.task_name,
        delay_steps=int(context.delay_steps),
        variant=context.variant,
        seed=int(context.seed),
        generation=int(context.generation),
        hof_flag=False,
        success=bool(metrics.get("success", False)),
        final_max_score=float(context.final_max_score),
        first_success_generation=context.first_success_generation,
        mean_alpha=mean_alpha,
        std_alpha=std_alpha,
        mean_eta=mean_eta,
        std_eta=std_eta,
        mean_plastic_d=mean_plastic_d,
        std_plastic_d=std_plastic_d,
        plastic_d_at_lower_bound_fraction=_fraction_near_value(
            plastic_d_values,
            target=float(context.plastic_d_lower_bound),
        ),
        plastic_d_at_zero_fraction=_fraction_near_value(plastic_d_values, target=0.0),
        node_count=len(genome.nodes),
        enabled_conn_count=len(enabled_connections),
        mean_abs_delta_w=_coerce_float(metrics.get("mean_abs_delta_w")),
        max_abs_delta_w=_coerce_float(metrics.get("max_abs_delta_w")),
        clamp_hit_rate=_coerce_float(metrics.get("clamp_hit_rate")),
        plasticity_active_fraction=_coerce_float(metrics.get("plasticity_active_fraction")),
        mean_abs_fast_state=_coerce_float(metrics.get("mean_abs_fast_state")),
        mean_abs_slow_state=_coerce_float(metrics.get("mean_abs_slow_state")),
        slow_fast_contribution_ratio=_coerce_float(metrics.get("slow_fast_contribution_ratio")),
        mean_abs_decay_term=_coerce_float(metrics.get("mean_abs_decay_term")),
        max_abs_decay_term=_coerce_float(metrics.get("max_abs_decay_term")),
        decay_effect_ratio=_coerce_float(metrics.get("decay_effect_ratio")),
        decay_near_zero_fraction=_coerce_float(metrics.get("decay_near_zero_fraction")),
        score_delay_3=_coerce_float(metrics.get("score_delay_3")),
        score_delay_5=_coerce_float(metrics.get("score_delay_5")),
        score_delay_8=_coerce_float(metrics.get("score_delay_8")),
        success_delay_3=bool(metrics.get("success_delay_3", False)),
        success_delay_5=bool(metrics.get("success_delay_5", False)),
        success_delay_8=bool(metrics.get("success_delay_8", False)),
        mean_score_over_delays=_coerce_float(metrics.get("mean_score_over_delays", context.final_max_score)),
        delay_score_std=_coerce_float(metrics.get("delay_score_std")),
        delay_score_range=_coerce_float(metrics.get("delay_score_range")),
        curriculum_enabled=bool(metrics.get("curriculum_enabled", context.curriculum_enabled)),
        curriculum_phase_1_delays=str(metrics.get("curriculum_phase_1_delays", context.curriculum_phase_1_delays)),
        curriculum_phase_2_delays=str(metrics.get("curriculum_phase_2_delays", context.curriculum_phase_2_delays)),
        curriculum_switch_generation=int(metrics.get("curriculum_switch_generation", context.curriculum_switch_generation or 0)),
        curriculum_phase=str(metrics.get("curriculum_phase", context.curriculum_phase or "static")),
        active_evaluation_delays=str(metrics.get("active_evaluation_delays", context.active_evaluation_delays)),
        score_current_phase=_coerce_float(metrics.get("score_current_phase", context.final_max_score)),
        query_accuracy=_coerce_float(metrics.get("query_accuracy")),
        retrieval_score=_coerce_float(metrics.get("retrieval_score")),
        exact_match_success=bool(metrics.get("exact_match_success", metrics.get("success", False))),
        mean_query_distance=_coerce_float(metrics.get("mean_query_distance")),
        distractor_load=_coerce_float(metrics.get("distractor_load")),
        num_stores=_coerce_float(metrics.get("num_stores")),
        num_queries=_coerce_float(metrics.get("num_queries")),
        num_distractors=_coerce_float(metrics.get("num_distractors")),
        retrieval_margin=_coerce_float(metrics.get("retrieval_margin")),
        retrieval_confusion_rate=_coerce_float(metrics.get("retrieval_confusion_rate")),
        relevant_token_retention=_coerce_float(
            metrics.get("relevant_token_retention", metrics.get("retrieval_score"))
        ),
        query_response_margin=_coerce_float(metrics.get("query_response_margin")),
        distractor_suppression_ratio=_coerce_float(metrics.get("distractor_suppression_ratio")),
        correct_key_selected=_coerce_float(metrics.get("correct_key_selected")),
        correct_value_selected=_coerce_float(metrics.get("correct_value_selected")),
        query_key_match_score=_coerce_float(metrics.get("query_key_match_score")),
        value_margin=_coerce_float(metrics.get("value_margin")),
        distractor_competition_score=_coerce_float(metrics.get("distractor_competition_score")),
        mean_abs_fast_state_during_store=_coerce_float(metrics.get("mean_abs_fast_state_during_store")),
        mean_abs_slow_state_during_store=_coerce_float(metrics.get("mean_abs_slow_state_during_store")),
        mean_abs_fast_state_during_query=_coerce_float(metrics.get("mean_abs_fast_state_during_query")),
        mean_abs_slow_state_during_query=_coerce_float(metrics.get("mean_abs_slow_state_during_query")),
        mean_abs_fast_state_during_distractor=_coerce_float(metrics.get("mean_abs_fast_state_during_distractor")),
        mean_abs_slow_state_during_distractor=_coerce_float(metrics.get("mean_abs_slow_state_during_distractor")),
        slow_query_coupling=_safe_ratio(
            _coerce_float(metrics.get("mean_abs_slow_state_during_query")),
            _coerce_float(metrics.get("mean_abs_slow_state_during_store")),
        ),
        store_query_state_gap=abs(
            _coerce_float(metrics.get("mean_abs_slow_state_during_store"))
            - _coerce_float(metrics.get("mean_abs_slow_state_during_query"))
        ),
        slow_fast_retrieval_ratio=_safe_ratio(
            _coerce_float(metrics.get("mean_abs_slow_state_during_query")),
            _coerce_float(metrics.get("mean_abs_fast_state_during_query")),
        ),
        retrieval_state_alignment=_state_alignment(
            _coerce_float(metrics.get("mean_abs_slow_state_during_store")),
            _coerce_float(metrics.get("mean_abs_slow_state_during_query")),
        ),
    )
    vector = _feature_vector_from_record(feature)
    return feature, vector


def _feature_vector_from_record(record: CandidateFeatureRecord) -> CandidateFeatureVectorRecord:
    vector = [
        record.mean_alpha,
        record.std_alpha,
        record.mean_eta,
        record.std_eta,
        record.mean_plastic_d,
        record.std_plastic_d,
        record.plastic_d_at_lower_bound_fraction,
        record.plastic_d_at_zero_fraction,
        float(record.node_count),
        float(record.enabled_conn_count),
        record.mean_abs_delta_w,
        record.max_abs_delta_w,
        record.clamp_hit_rate,
        record.plasticity_active_fraction,
        record.mean_abs_fast_state,
        record.mean_abs_slow_state,
        record.slow_fast_contribution_ratio,
        record.mean_abs_decay_term,
        record.max_abs_decay_term,
        record.decay_effect_ratio,
        record.decay_near_zero_fraction,
        record.score_delay_3,
        record.score_delay_5,
        record.score_delay_8,
        record.mean_score_over_delays,
        record.delay_score_std,
        record.delay_score_range,
        record.score_current_phase,
        record.query_accuracy,
        record.retrieval_score,
        record.mean_query_distance,
        record.distractor_load,
        record.num_stores,
        record.num_queries,
        record.num_distractors,
        record.retrieval_margin,
        record.retrieval_confusion_rate,
        record.relevant_token_retention,
        record.query_response_margin,
        record.distractor_suppression_ratio,
        record.correct_key_selected,
        record.correct_value_selected,
        record.query_key_match_score,
        record.value_margin,
        record.distractor_competition_score,
        record.mean_abs_fast_state_during_store,
        record.mean_abs_slow_state_during_store,
        record.mean_abs_fast_state_during_query,
        record.mean_abs_slow_state_during_query,
        record.mean_abs_fast_state_during_distractor,
        record.mean_abs_slow_state_during_distractor,
        record.slow_query_coupling,
        record.store_query_state_gap,
        record.slow_fast_retrieval_ratio,
        record.retrieval_state_alignment,
    ]
    array = np.asarray(vector, dtype=np.float64)
    norm_l2 = float(np.linalg.norm(array))
    return CandidateFeatureVectorRecord(
        candidate_id=record.candidate_id,
        feature_version=FEATURE_VERSION,
        vector_json=stable_json_dumps([round(float(value), 10) for value in array.tolist()]),
        norm_l2=norm_l2,
    )


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def _fraction_near_value(values: Sequence[float], *, target: float) -> float:
    if not values:
        return 0.0
    hits = sum(1 for value in values if math.isclose(float(value), target, rel_tol=0.0, abs_tol=BOUND_TOLERANCE))
    return hits / len(values)


def _coerce_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(coerced) or math.isinf(coerced):
        return 0.0
    return coerced


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1e-9:
        return 0.0
    return float(numerator) / float(denominator)


def _state_alignment(store_value: float, query_value: float) -> float:
    denominator = abs(float(store_value)) + abs(float(query_value))
    if denominator <= 1e-9:
        return 0.0
    aligned = 1.0 - (abs(float(store_value) - float(query_value)) / denominator)
    return max(0.0, min(1.0, float(aligned)))
