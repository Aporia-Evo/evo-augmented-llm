from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    task_name: str
    seed: int
    status: str
    config_json: str
    created_at: str
    finished_at: str | None
    mode: str = "generation"


@dataclass(frozen=True)
class GenerationRecord:
    run_id: str
    generation_id: int
    state: str
    best_candidate_id: str | None
    best_score: float | None
    avg_score: float | None
    created_at: str
    committed_at: str | None
    eval_duration_ms: int | None = None
    commit_duration_ms: int | None = None


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    run_id: str
    generation_id: int
    genome_blob: str
    status: str
    parent_ids: list[str]
    created_at: str


@dataclass(frozen=True)
class FitnessRecord:
    candidate_id: str
    run_id: str
    generation_id: int
    score: float
    raw_metrics: dict[str, Any]
    evaluated_at: str


@dataclass(frozen=True)
class EliteRecord:
    elite_id: str
    run_id: str
    source_generation: int
    candidate_id: str
    rank: int
    score: float
    frozen_genome_blob: str
    archived_at: str


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    run_id: str
    type: str
    payload_json: str
    created_at: str


@dataclass(frozen=True)
class CheckpointRecord:
    run_id: str
    generation_id: int
    state_blob: str
    parent_ids_json: str
    created_at: str


@dataclass(frozen=True)
class GenerationCommitResult:
    generation: GenerationRecord
    elites: tuple[EliteRecord, ...]


@dataclass(frozen=True)
class ActiveCandidateRecord:
    candidate_id: str
    run_id: str
    slot_index: int
    variant: str
    genome_blob: str
    status: str
    rolling_score: float
    eval_count: int
    birth_step: int
    last_eval_at: str | None
    parent_ids: list[str]
    created_at: str


@dataclass(frozen=True)
class EvaluationJobRecord:
    job_id: str
    run_id: str
    candidate_id: str
    task_payload_json: str
    status: str
    claimed_by: str | None
    created_at: str
    claimed_at: str | None
    finished_at: str | None


@dataclass(frozen=True)
class EvaluationResultRecord:
    result_id: str
    run_id: str
    candidate_id: str
    score: float
    raw_metrics: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class HallOfFameEntryRecord:
    entry_id: str
    run_id: str
    candidate_id: str
    score: float
    frozen_genome_blob: str
    inserted_at: str


@dataclass(frozen=True)
class CandidateLifecycleEventRecord:
    event_id: str
    run_id: str
    candidate_id: str
    event_type: str
    payload_json: str
    created_at: str


@dataclass(frozen=True)
class OnlineMetricRecord:
    metric_id: str
    run_id: str
    timestamp: str
    active_population_size: int
    rolling_best_score: float
    rolling_avg_score: float
    replacement_count: int
    success_rate_window: float


@dataclass(frozen=True)
class OnlineStateRecord:
    run_id: str
    step: int
    replacement_count: int
    success_window_json: str
    adapter_state_blob: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CandidateFeatureRecord:
    candidate_id: str
    run_id: str
    benchmark_label: str
    task_name: str
    delay_steps: int
    variant: str
    seed: int
    generation: int
    hof_flag: bool
    success: bool
    final_max_score: float
    first_success_generation: int | None
    mean_alpha: float
    std_alpha: float
    mean_eta: float
    std_eta: float
    mean_plastic_d: float
    std_plastic_d: float
    plastic_d_at_lower_bound_fraction: float
    plastic_d_at_zero_fraction: float
    node_count: int
    enabled_conn_count: int
    mean_abs_delta_w: float
    max_abs_delta_w: float
    clamp_hit_rate: float
    plasticity_active_fraction: float
    mean_abs_fast_state: float = 0.0
    mean_abs_slow_state: float = 0.0
    slow_fast_contribution_ratio: float = 0.0
    mean_abs_decay_term: float = 0.0
    max_abs_decay_term: float = 0.0
    decay_effect_ratio: float = 0.0
    decay_near_zero_fraction: float = 0.0
    score_delay_3: float = 0.0
    score_delay_5: float = 0.0
    score_delay_8: float = 0.0
    success_delay_3: bool = False
    success_delay_5: bool = False
    success_delay_8: bool = False
    mean_score_over_delays: float = 0.0
    delay_score_std: float = 0.0
    delay_score_range: float = 0.0
    curriculum_enabled: bool = False
    curriculum_phase_1_delays: str = ""
    curriculum_phase_2_delays: str = ""
    curriculum_switch_generation: int = 0
    curriculum_phase: str = "static"
    active_evaluation_delays: str = ""
    score_current_phase: float = 0.0
    query_accuracy: float = 0.0
    retrieval_score: float = 0.0
    exact_match_success: bool = False
    mean_query_distance: float = 0.0
    distractor_load: float = 0.0
    num_stores: float = 0.0
    num_queries: float = 0.0
    num_distractors: float = 0.0
    retrieval_margin: float = 0.0
    retrieval_confusion_rate: float = 0.0
    relevant_token_retention: float = 0.0
    query_response_margin: float = 0.0
    distractor_suppression_ratio: float = 0.0
    correct_key_selected: float = 0.0
    correct_value_selected: float = 0.0
    query_key_match_score: float = 0.0
    value_margin: float = 0.0
    distractor_competition_score: float = 0.0
    mean_abs_fast_state_during_store: float = 0.0
    mean_abs_slow_state_during_store: float = 0.0
    mean_abs_fast_state_during_query: float = 0.0
    mean_abs_slow_state_during_query: float = 0.0
    mean_abs_fast_state_during_distractor: float = 0.0
    mean_abs_slow_state_during_distractor: float = 0.0
    slow_query_coupling: float = 0.0
    store_query_state_gap: float = 0.0
    slow_fast_retrieval_ratio: float = 0.0
    retrieval_state_alignment: float = 0.0
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


@dataclass(frozen=True)
class CandidateFeatureVectorRecord:
    candidate_id: str
    feature_version: str
    vector_json: str
    norm_l2: float


@dataclass(frozen=True)
class ArchiveCellRecord:
    archive_id: str
    benchmark_label: str
    task_name: str
    delay_steps: int
    variant: str
    descriptor_key: str
    descriptor_values_json: str
    elite_candidate_id: str
    elite_score: float
    elite_run_id: str
    updated_at: str
    qd_profile: str = "mechanism_v2"
    descriptor_schema_version: str = "v7a-qdlight-v1"
    curriculum_enabled: bool = False
    curriculum_phase_1_delays: str = ""
    curriculum_phase_2_delays: str = ""
    curriculum_switch_generation: int = 0


@dataclass(frozen=True)
class ArchiveEventRecord:
    event_id: str
    archive_id: str
    benchmark_label: str
    task_name: str
    delay_steps: int
    variant: str
    descriptor_key: str
    candidate_id: str
    event_type: str
    score: float
    created_at: str
    qd_profile: str = "mechanism_v2"
    descriptor_schema_version: str = "v7a-qdlight-v1"
    curriculum_enabled: bool = False
    curriculum_phase_1_delays: str = ""
    curriculum_phase_2_delays: str = ""
    curriculum_switch_generation: int = 0
