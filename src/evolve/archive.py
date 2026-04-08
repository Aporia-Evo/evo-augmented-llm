from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from db.models import ArchiveCellRecord, CandidateFeatureRecord
from utils.serialization import stable_json_dumps, utc_now_iso


ARCHIVE_SCORE_BINS = 8
ARCHIVE_SLOW_RATIO_BINS = 8
ARCHIVE_SLOW_RATIO_MAX = 6.0
ARCHIVE_CONN_BINS = 8
ARCHIVE_CONN_BIN_EDGES = (2, 3, 4, 5, 6, 8, 10)
ARCHIVE_DELAY_STD_BINS = 8
ARCHIVE_DELAY_STD_MAX = 2.0
ARCHIVE_RETRIEVAL_SUPPRESSION_BINS = 8
ARCHIVE_RETRIEVAL_SUPPRESSION_MAX = 4.0
ARCHIVE_RETRIEVAL_COUPLING_BINS = 8
ARCHIVE_RETRIEVAL_COUPLING_MAX = 3.0
ARCHIVE_GATE_SELECTIVITY_BINS = 8
ARCHIVE_GATE_SELECTIVITY_MAX = 1.0

QD_PROFILE_MECHANISM_V2 = "mechanism_v2"
QD_PROFILE_GENERAL_COMPACTNESS = "general_compactness"
QD_PROFILE_DELAY_ROBUSTNESS = "delay_robustness"
QD_PROFILE_CURRICULUM_PROGRESS = "curriculum_progress"
QD_PROFILE_RETRIEVAL_STRATEGY = "retrieval_strategy"
QD_PROFILE_RETRIEVAL_MECHANISM = "retrieval_mechanism"
QD_PROFILE_GATING_MECHANISM = "gating_mechanism"
QD_PROFILE_CONTENT_RETRIEVAL = "content_retrieval"
QD_PROFILE_KV_RETRIEVAL_MECHANISM = "kv_retrieval_mechanism"
QD_PROFILE_SLOT_RETRIEVAL_MECHANISM = "slot_retrieval_mechanism"
DEFAULT_QD_PROFILE = QD_PROFILE_MECHANISM_V2

ARCHIVE_DESCRIPTOR_VERSION_MECHANISM_V2 = "v7a-qdlight-v1"
ARCHIVE_DESCRIPTOR_VERSION_GENERAL_COMPACTNESS = "v7b-qdlight-v1"
ARCHIVE_DESCRIPTOR_VERSION_DELAY_ROBUSTNESS = "v8a-qdlight-v1"
ARCHIVE_DESCRIPTOR_VERSION_CURRICULUM_PROGRESS = "v8b-qdlight-v1"
ARCHIVE_DESCRIPTOR_VERSION_RETRIEVAL_STRATEGY = "v9a-retrieval-v1"
ARCHIVE_DESCRIPTOR_VERSION_RETRIEVAL_MECHANISM = "v9b-retrieval-v1"
ARCHIVE_DESCRIPTOR_VERSION_GATING_MECHANISM = "v10a-gating-v1"
ARCHIVE_DESCRIPTOR_VERSION_CONTENT_RETRIEVAL = "v10c-content-v1"
ARCHIVE_DESCRIPTOR_VERSION_KV_RETRIEVAL_MECHANISM = "v11b-kv-v1"
ARCHIVE_DESCRIPTOR_VERSION_SLOT_RETRIEVAL_MECHANISM = "v12a-slots-v1"


@dataclass(frozen=True)
class ArchiveDescriptor:
    qd_profile: str
    descriptor_schema_version: str
    descriptor_key: str
    descriptor_values_json: str
    score_bin: int
    secondary_bin: int


@dataclass(frozen=True)
class ArchiveProfileDefinition:
    name: str
    descriptor_schema_version: str
    secondary_axis_label: str
    secondary_axis_value_key: str
    secondary_axis_bin_key: str
    secondary_axis_bin_count: int
    total_cells_per_delay: int
    builder: Callable[..., ArchiveDescriptor]


def build_archive_cell(
    feature_record: CandidateFeatureRecord,
    *,
    score_ceiling: float,
    qd_profile: str = DEFAULT_QD_PROFILE,
) -> ArchiveCellRecord:
    descriptor = build_archive_descriptor(
        final_max_score=feature_record.final_max_score,
        score_ceiling=score_ceiling,
        mean_score_over_delays=feature_record.mean_score_over_delays,
        delay_score_std=feature_record.delay_score_std,
        slow_fast_contribution_ratio=feature_record.slow_fast_contribution_ratio,
        enabled_conn_count=feature_record.enabled_conn_count,
        retrieval_score=feature_record.retrieval_score,
        query_accuracy=feature_record.query_accuracy,
        relevant_token_retention=feature_record.relevant_token_retention,
        distractor_suppression_ratio=feature_record.distractor_suppression_ratio,
        slow_query_coupling=feature_record.slow_query_coupling,
        gate_selectivity=feature_record.gate_selectivity,
        match_selectivity=feature_record.match_selectivity,
        query_key_alignment=feature_record.query_key_alignment,
        store_vs_distractor_write_gap=feature_record.store_vs_distractor_write_gap,
        readout_selectivity=feature_record.readout_selectivity,
        slot_write_focus=feature_record.slot_write_focus,
        slot_query_focus=feature_record.slot_query_focus,
        slot_utilization=feature_record.slot_utilization,
        curriculum_enabled=feature_record.curriculum_enabled,
        curriculum_phase_1_delays=feature_record.curriculum_phase_1_delays,
        curriculum_phase_2_delays=feature_record.curriculum_phase_2_delays,
        curriculum_switch_generation=feature_record.curriculum_switch_generation,
        curriculum_phase=feature_record.curriculum_phase,
        active_evaluation_delays=feature_record.active_evaluation_delays,
        qd_profile=qd_profile,
    )
    archive_id = (
        f"{feature_record.benchmark_label}|{feature_record.task_name}|"
        f"d{feature_record.delay_steps}|{feature_record.variant}|"
        f"{descriptor.qd_profile}|{descriptor.descriptor_key}"
    )
    return ArchiveCellRecord(
        archive_id=archive_id,
        benchmark_label=feature_record.benchmark_label,
        task_name=feature_record.task_name,
        delay_steps=int(feature_record.delay_steps),
        variant=feature_record.variant,
        descriptor_key=descriptor.descriptor_key,
        descriptor_values_json=descriptor.descriptor_values_json,
        elite_candidate_id=feature_record.candidate_id,
        elite_score=float(feature_record.final_max_score),
        elite_run_id=feature_record.run_id,
        updated_at=utc_now_iso(),
        qd_profile=descriptor.qd_profile,
        descriptor_schema_version=descriptor.descriptor_schema_version,
        curriculum_enabled=feature_record.curriculum_enabled,
        curriculum_phase_1_delays=feature_record.curriculum_phase_1_delays,
        curriculum_phase_2_delays=feature_record.curriculum_phase_2_delays,
        curriculum_switch_generation=feature_record.curriculum_switch_generation,
    )


def build_archive_cells(
    feature_record: CandidateFeatureRecord,
    *,
    score_ceiling: float,
) -> list[ArchiveCellRecord]:
    return [
        build_archive_cell(feature_record, score_ceiling=score_ceiling, qd_profile=profile_name)
        for profile_name in archive_profile_names(task_name=feature_record.task_name)
    ]


def build_archive_descriptor(
    *,
    final_max_score: float,
    score_ceiling: float,
    mean_score_over_delays: float | None = None,
    delay_score_std: float = 0.0,
    slow_fast_contribution_ratio: float = 0.0,
    enabled_conn_count: int = 0,
    retrieval_score: float = 0.0,
    query_accuracy: float = 0.0,
    relevant_token_retention: float = 0.0,
    distractor_suppression_ratio: float = 0.0,
    slow_query_coupling: float = 0.0,
    gate_selectivity: float = 0.0,
    match_selectivity: float = 0.0,
    query_key_alignment: float = 0.0,
    store_vs_distractor_write_gap: float = 0.0,
    readout_selectivity: float = 0.0,
    slot_write_focus: float = 0.0,
    slot_query_focus: float = 0.0,
    slot_utilization: float = 0.0,
    curriculum_enabled: bool = False,
    curriculum_phase_1_delays: str = "",
    curriculum_phase_2_delays: str = "",
    curriculum_switch_generation: int = 0,
    curriculum_phase: str = "static",
    active_evaluation_delays: str = "",
    qd_profile: str = DEFAULT_QD_PROFILE,
) -> ArchiveDescriptor:
    profile = archive_profile_definition(qd_profile)
    builder_kwargs = dict(
        final_max_score=final_max_score,
        score_ceiling=score_ceiling,
        mean_score_over_delays=mean_score_over_delays,
        delay_score_std=delay_score_std,
        slow_fast_contribution_ratio=slow_fast_contribution_ratio,
        enabled_conn_count=enabled_conn_count,
        retrieval_score=retrieval_score,
        query_accuracy=query_accuracy,
        relevant_token_retention=relevant_token_retention,
        distractor_suppression_ratio=distractor_suppression_ratio,
        slow_query_coupling=slow_query_coupling,
        gate_selectivity=gate_selectivity,
        match_selectivity=match_selectivity,
        query_key_alignment=query_key_alignment,
        curriculum_enabled=curriculum_enabled,
        curriculum_phase_1_delays=curriculum_phase_1_delays,
        curriculum_phase_2_delays=curriculum_phase_2_delays,
        curriculum_switch_generation=curriculum_switch_generation,
        curriculum_phase=curriculum_phase,
        active_evaluation_delays=active_evaluation_delays,
    )
    if qd_profile == QD_PROFILE_KV_RETRIEVAL_MECHANISM:
        builder_kwargs["store_vs_distractor_write_gap"] = store_vs_distractor_write_gap
        builder_kwargs["readout_selectivity"] = readout_selectivity
    if qd_profile == QD_PROFILE_SLOT_RETRIEVAL_MECHANISM:
        builder_kwargs["slot_write_focus"] = slot_write_focus
        builder_kwargs["slot_query_focus"] = slot_query_focus
        builder_kwargs["slot_utilization"] = slot_utilization
    return profile.builder(**builder_kwargs)


def archive_profile_names(*, task_name: str | None = None) -> tuple[str, ...]:
    profile_names = list(ARCHIVE_PROFILES.keys())
    if task_name != "key_value_memory":
        profile_names = [
            profile_name
            for profile_name in profile_names
            if profile_name not in {QD_PROFILE_RETRIEVAL_STRATEGY, QD_PROFILE_RETRIEVAL_MECHANISM, QD_PROFILE_GATING_MECHANISM, QD_PROFILE_CONTENT_RETRIEVAL, QD_PROFILE_KV_RETRIEVAL_MECHANISM, QD_PROFILE_SLOT_RETRIEVAL_MECHANISM}
        ]
    return tuple(profile_names)


def archive_profile_definition(qd_profile: str) -> ArchiveProfileDefinition:
    try:
        return ARCHIVE_PROFILES[qd_profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported QD profile: {qd_profile}") from exc


def _build_mechanism_v2_descriptor(
    *,
    final_max_score: float,
    score_ceiling: float,
    mean_score_over_delays: float | None,
    delay_score_std: float,
    slow_fast_contribution_ratio: float,
    enabled_conn_count: int,
    retrieval_score: float,
    query_accuracy: float,
    relevant_token_retention: float,
    distractor_suppression_ratio: float,
    slow_query_coupling: float,
    gate_selectivity: float,
    match_selectivity: float,
    query_key_alignment: float,
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del mean_score_over_delays, delay_score_std, curriculum_enabled
    del curriculum_phase_1_delays, curriculum_phase_2_delays, curriculum_switch_generation
    del curriculum_phase, active_evaluation_delays
    del enabled_conn_count, retrieval_score, query_accuracy, relevant_token_retention
    del distractor_suppression_ratio, slow_query_coupling
    del gate_selectivity, match_selectivity, query_key_alignment
    normalized_score, score_bin, safe_score_ceiling = _normalized_score(final_max_score, score_ceiling)
    safe_slow_ratio = max(0.0, float(slow_fast_contribution_ratio))
    capped_slow_ratio = min(ARCHIVE_SLOW_RATIO_MAX, safe_slow_ratio)
    slow_ratio_bin = _bin_value(
        capped_slow_ratio,
        upper_bound=ARCHIVE_SLOW_RATIO_MAX,
        bin_count=ARCHIVE_SLOW_RATIO_BINS,
    )
    descriptor_key = f"scorebin_{score_bin}|slowratio_{slow_ratio_bin}"
    descriptor_values_json = stable_json_dumps(
        {
            "qd_profile": QD_PROFILE_MECHANISM_V2,
            "descriptor_schema_version": ARCHIVE_DESCRIPTOR_VERSION_MECHANISM_V2,
            "score_value": round(float(final_max_score), 10),
            "score_ceiling": round(safe_score_ceiling, 10),
            "normalized_score": round(normalized_score, 10),
            "score_bin": score_bin,
            "score_bin_count": ARCHIVE_SCORE_BINS,
            "slow_fast_contribution_ratio": round(safe_slow_ratio, 10),
            "slow_ratio_capped": round(capped_slow_ratio, 10),
            "slow_ratio_bin": slow_ratio_bin,
            "slow_ratio_bin_count": ARCHIVE_SLOW_RATIO_BINS,
            "slow_ratio_max": ARCHIVE_SLOW_RATIO_MAX,
        }
    )
    return ArchiveDescriptor(
        qd_profile=QD_PROFILE_MECHANISM_V2,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_MECHANISM_V2,
        descriptor_key=descriptor_key,
        descriptor_values_json=descriptor_values_json,
        score_bin=score_bin,
        secondary_bin=slow_ratio_bin,
    )


def _build_general_compactness_descriptor(
    *,
    final_max_score: float,
    score_ceiling: float,
    mean_score_over_delays: float | None,
    delay_score_std: float,
    slow_fast_contribution_ratio: float,
    enabled_conn_count: int,
    retrieval_score: float,
    query_accuracy: float,
    relevant_token_retention: float,
    distractor_suppression_ratio: float,
    slow_query_coupling: float,
    gate_selectivity: float,
    match_selectivity: float,
    query_key_alignment: float,
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del mean_score_over_delays, delay_score_std, curriculum_enabled
    del curriculum_phase_1_delays, curriculum_phase_2_delays, curriculum_switch_generation
    del curriculum_phase, active_evaluation_delays
    del slow_fast_contribution_ratio, retrieval_score, query_accuracy, relevant_token_retention
    del distractor_suppression_ratio, slow_query_coupling
    del gate_selectivity, match_selectivity, query_key_alignment
    normalized_score, score_bin, safe_score_ceiling = _normalized_score(final_max_score, score_ceiling)
    safe_enabled_conn_count = max(0, int(enabled_conn_count))
    conn_bin = _bin_conn_count(safe_enabled_conn_count)
    descriptor_key = f"scorebin_{score_bin}|connbin_{conn_bin}"
    descriptor_values_json = stable_json_dumps(
        {
            "qd_profile": QD_PROFILE_GENERAL_COMPACTNESS,
            "descriptor_schema_version": ARCHIVE_DESCRIPTOR_VERSION_GENERAL_COMPACTNESS,
            "score_value": round(float(final_max_score), 10),
            "score_ceiling": round(safe_score_ceiling, 10),
            "normalized_score": round(normalized_score, 10),
            "score_bin": score_bin,
            "score_bin_count": ARCHIVE_SCORE_BINS,
            "enabled_conn_count": safe_enabled_conn_count,
            "conn_bin": conn_bin,
            "conn_bin_count": ARCHIVE_CONN_BINS,
            "conn_bin_edges": list(ARCHIVE_CONN_BIN_EDGES),
        }
    )
    return ArchiveDescriptor(
        qd_profile=QD_PROFILE_GENERAL_COMPACTNESS,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_GENERAL_COMPACTNESS,
        descriptor_key=descriptor_key,
        descriptor_values_json=descriptor_values_json,
        score_bin=score_bin,
        secondary_bin=conn_bin,
    )


def _build_delay_robustness_descriptor(
    *,
    final_max_score: float,
    score_ceiling: float,
    mean_score_over_delays: float | None,
    delay_score_std: float,
    slow_fast_contribution_ratio: float,
    enabled_conn_count: int,
    retrieval_score: float,
    query_accuracy: float,
    relevant_token_retention: float,
    distractor_suppression_ratio: float,
    slow_query_coupling: float,
    gate_selectivity: float,
    match_selectivity: float,
    query_key_alignment: float,
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del slow_fast_contribution_ratio, enabled_conn_count
    del retrieval_score, query_accuracy, relevant_token_retention, distractor_suppression_ratio, slow_query_coupling
    del gate_selectivity, match_selectivity, query_key_alignment
    effective_score = float(mean_score_over_delays) if mean_score_over_delays is not None else float(final_max_score)
    normalized_score, score_bin, safe_score_ceiling = _normalized_score(effective_score, score_ceiling)
    safe_delay_score_std = max(0.0, float(delay_score_std))
    capped_delay_score_std = min(ARCHIVE_DELAY_STD_MAX, safe_delay_score_std)
    delay_std_bin = _bin_value(
        capped_delay_score_std,
        upper_bound=ARCHIVE_DELAY_STD_MAX,
        bin_count=ARCHIVE_DELAY_STD_BINS,
    )
    descriptor_key = f"scorebin_{score_bin}|stdbin_{delay_std_bin}"
    descriptor_values_json = stable_json_dumps(
        {
            "qd_profile": QD_PROFILE_DELAY_ROBUSTNESS,
            "descriptor_schema_version": ARCHIVE_DESCRIPTOR_VERSION_DELAY_ROBUSTNESS,
            "score_value": round(effective_score, 10),
            "score_ceiling": round(safe_score_ceiling, 10),
            "normalized_score": round(normalized_score, 10),
            "score_bin": score_bin,
            "score_bin_count": ARCHIVE_SCORE_BINS,
            "mean_score_over_delays": round(effective_score, 10),
            "delay_score_std": round(safe_delay_score_std, 10),
            "delay_score_std_capped": round(capped_delay_score_std, 10),
            "delay_std_bin": delay_std_bin,
            "delay_std_bin_count": ARCHIVE_DELAY_STD_BINS,
            "delay_std_max": ARCHIVE_DELAY_STD_MAX,
            "curriculum_enabled": bool(curriculum_enabled),
            "curriculum_phase_1_delays": curriculum_phase_1_delays,
            "curriculum_phase_2_delays": curriculum_phase_2_delays,
            "curriculum_switch_generation": int(curriculum_switch_generation),
            "curriculum_phase": curriculum_phase or "static",
            "active_evaluation_delays": active_evaluation_delays,
        }
    )
    return ArchiveDescriptor(
        qd_profile=QD_PROFILE_DELAY_ROBUSTNESS,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_DELAY_ROBUSTNESS,
        descriptor_key=descriptor_key,
        descriptor_values_json=descriptor_values_json,
        score_bin=score_bin,
        secondary_bin=delay_std_bin,
    )


def _build_curriculum_progress_descriptor(
    *,
    final_max_score: float,
    score_ceiling: float,
    mean_score_over_delays: float | None,
    delay_score_std: float,
    slow_fast_contribution_ratio: float,
    enabled_conn_count: int,
    retrieval_score: float,
    query_accuracy: float,
    relevant_token_retention: float,
    distractor_suppression_ratio: float,
    slow_query_coupling: float,
    gate_selectivity: float,
    match_selectivity: float,
    query_key_alignment: float,
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del slow_fast_contribution_ratio, enabled_conn_count
    del retrieval_score, query_accuracy, relevant_token_retention, distractor_suppression_ratio, slow_query_coupling
    del gate_selectivity, match_selectivity, query_key_alignment
    effective_score = float(mean_score_over_delays) if mean_score_over_delays is not None else float(final_max_score)
    normalized_score, score_bin, safe_score_ceiling = _normalized_score(effective_score, score_ceiling)
    safe_delay_score_std = max(0.0, float(delay_score_std))
    capped_delay_score_std = min(ARCHIVE_DELAY_STD_MAX, safe_delay_score_std)
    delay_std_bin = _bin_value(
        capped_delay_score_std,
        upper_bound=ARCHIVE_DELAY_STD_MAX,
        bin_count=ARCHIVE_DELAY_STD_BINS,
    )
    safe_curriculum_phase = curriculum_phase or "static"
    descriptor_key = f"{safe_curriculum_phase}|scorebin_{score_bin}|stdbin_{delay_std_bin}"
    descriptor_values_json = stable_json_dumps(
        {
            "qd_profile": QD_PROFILE_CURRICULUM_PROGRESS,
            "descriptor_schema_version": ARCHIVE_DESCRIPTOR_VERSION_CURRICULUM_PROGRESS,
            "score_value": round(effective_score, 10),
            "score_ceiling": round(safe_score_ceiling, 10),
            "normalized_score": round(normalized_score, 10),
            "score_bin": score_bin,
            "score_bin_count": ARCHIVE_SCORE_BINS,
            "mean_score_over_delays": round(effective_score, 10),
            "delay_score_std": round(safe_delay_score_std, 10),
            "delay_score_std_capped": round(capped_delay_score_std, 10),
            "delay_std_bin": delay_std_bin,
            "delay_std_bin_count": ARCHIVE_DELAY_STD_BINS,
            "delay_std_max": ARCHIVE_DELAY_STD_MAX,
            "curriculum_phase": safe_curriculum_phase,
            "active_evaluation_delays": active_evaluation_delays,
            "curriculum_enabled": bool(curriculum_enabled),
            "curriculum_phase_1_delays": curriculum_phase_1_delays,
            "curriculum_phase_2_delays": curriculum_phase_2_delays,
            "curriculum_switch_generation": int(curriculum_switch_generation),
        }
    )
    return ArchiveDescriptor(
        qd_profile=QD_PROFILE_CURRICULUM_PROGRESS,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_CURRICULUM_PROGRESS,
        descriptor_key=descriptor_key,
        descriptor_values_json=descriptor_values_json,
        score_bin=score_bin,
        secondary_bin=delay_std_bin,
    )


def _build_retrieval_strategy_descriptor(
    *,
    final_max_score: float,
    score_ceiling: float,
    mean_score_over_delays: float | None,
    delay_score_std: float,
    slow_fast_contribution_ratio: float,
    enabled_conn_count: int,
    retrieval_score: float,
    query_accuracy: float,
    relevant_token_retention: float,
    distractor_suppression_ratio: float,
    slow_query_coupling: float,
    gate_selectivity: float,
    match_selectivity: float,
    query_key_alignment: float,
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del mean_score_over_delays, delay_score_std, slow_fast_contribution_ratio, enabled_conn_count, gate_selectivity, match_selectivity, query_key_alignment
    effective_score = max(0.0, min(1.0, float(retrieval_score or query_accuracy or relevant_token_retention)))
    normalized_score, score_bin, _safe_score_ceiling = _normalized_score(effective_score, 1.0)
    safe_suppression = max(0.0, float(distractor_suppression_ratio))
    capped_suppression = min(ARCHIVE_RETRIEVAL_SUPPRESSION_MAX, safe_suppression)
    suppression_bin = _bin_value(
        capped_suppression,
        upper_bound=ARCHIVE_RETRIEVAL_SUPPRESSION_MAX,
        bin_count=ARCHIVE_RETRIEVAL_SUPPRESSION_BINS,
    )
    descriptor_key = f"scorebin_{score_bin}|suppbin_{suppression_bin}"
    descriptor_values_json = stable_json_dumps(
        {
            "qd_profile": QD_PROFILE_RETRIEVAL_STRATEGY,
            "descriptor_schema_version": ARCHIVE_DESCRIPTOR_VERSION_RETRIEVAL_STRATEGY,
            "retrieval_score": round(effective_score, 10),
            "normalized_score": round(normalized_score, 10),
            "score_bin": score_bin,
            "score_bin_count": ARCHIVE_SCORE_BINS,
            "query_accuracy": round(float(query_accuracy), 10),
            "relevant_token_retention": round(float(relevant_token_retention), 10),
            "distractor_suppression_ratio": round(safe_suppression, 10),
            "distractor_suppression_capped": round(capped_suppression, 10),
            "suppression_bin": suppression_bin,
            "suppression_bin_count": ARCHIVE_RETRIEVAL_SUPPRESSION_BINS,
            "suppression_max": ARCHIVE_RETRIEVAL_SUPPRESSION_MAX,
            "slow_query_coupling": round(float(slow_query_coupling), 10),
            "curriculum_enabled": bool(curriculum_enabled),
            "curriculum_phase_1_delays": curriculum_phase_1_delays,
            "curriculum_phase_2_delays": curriculum_phase_2_delays,
            "curriculum_switch_generation": int(curriculum_switch_generation),
            "curriculum_phase": curriculum_phase or "static",
            "active_evaluation_delays": active_evaluation_delays,
            "score_value": round(float(final_max_score), 10),
            "score_ceiling": round(float(score_ceiling), 10),
        }
    )
    return ArchiveDescriptor(
        qd_profile=QD_PROFILE_RETRIEVAL_STRATEGY,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_RETRIEVAL_STRATEGY,
        descriptor_key=descriptor_key,
        descriptor_values_json=descriptor_values_json,
        score_bin=score_bin,
        secondary_bin=suppression_bin,
    )


def _build_retrieval_mechanism_descriptor(
    *,
    final_max_score: float,
    score_ceiling: float,
    mean_score_over_delays: float | None,
    delay_score_std: float,
    slow_fast_contribution_ratio: float,
    enabled_conn_count: int,
    retrieval_score: float,
    query_accuracy: float,
    relevant_token_retention: float,
    distractor_suppression_ratio: float,
    slow_query_coupling: float,
    gate_selectivity: float,
    match_selectivity: float,
    query_key_alignment: float,
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del mean_score_over_delays, delay_score_std, slow_fast_contribution_ratio, enabled_conn_count, gate_selectivity, match_selectivity, query_key_alignment
    effective_score = max(0.0, min(1.0, float(retrieval_score or query_accuracy or relevant_token_retention)))
    normalized_score, score_bin, _safe_score_ceiling = _normalized_score(effective_score, 1.0)
    safe_coupling = max(0.0, float(slow_query_coupling))
    capped_coupling = min(ARCHIVE_RETRIEVAL_COUPLING_MAX, safe_coupling)
    coupling_bin = _bin_value(
        capped_coupling,
        upper_bound=ARCHIVE_RETRIEVAL_COUPLING_MAX,
        bin_count=ARCHIVE_RETRIEVAL_COUPLING_BINS,
    )
    descriptor_key = f"scorebin_{score_bin}|couplingbin_{coupling_bin}"
    descriptor_values_json = stable_json_dumps(
        {
            "qd_profile": QD_PROFILE_RETRIEVAL_MECHANISM,
            "descriptor_schema_version": ARCHIVE_DESCRIPTOR_VERSION_RETRIEVAL_MECHANISM,
            "retrieval_score": round(effective_score, 10),
            "normalized_score": round(normalized_score, 10),
            "score_bin": score_bin,
            "score_bin_count": ARCHIVE_SCORE_BINS,
            "query_accuracy": round(float(query_accuracy), 10),
            "distractor_suppression_ratio": round(float(distractor_suppression_ratio), 10),
            "slow_query_coupling": round(safe_coupling, 10),
            "slow_query_coupling_capped": round(capped_coupling, 10),
            "coupling_bin": coupling_bin,
            "coupling_bin_count": ARCHIVE_RETRIEVAL_COUPLING_BINS,
            "coupling_max": ARCHIVE_RETRIEVAL_COUPLING_MAX,
            "curriculum_enabled": bool(curriculum_enabled),
            "curriculum_phase_1_delays": curriculum_phase_1_delays,
            "curriculum_phase_2_delays": curriculum_phase_2_delays,
            "curriculum_switch_generation": int(curriculum_switch_generation),
            "curriculum_phase": curriculum_phase or "static",
            "active_evaluation_delays": active_evaluation_delays,
            "score_value": round(float(final_max_score), 10),
            "score_ceiling": round(float(score_ceiling), 10),
        }
    )
    return ArchiveDescriptor(
        qd_profile=QD_PROFILE_RETRIEVAL_MECHANISM,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_RETRIEVAL_MECHANISM,
        descriptor_key=descriptor_key,
        descriptor_values_json=descriptor_values_json,
        score_bin=score_bin,
        secondary_bin=coupling_bin,
    )


def _build_gating_mechanism_descriptor(
    *,
    final_max_score: float,
    score_ceiling: float,
    mean_score_over_delays: float | None,
    delay_score_std: float,
    slow_fast_contribution_ratio: float,
    enabled_conn_count: int,
    retrieval_score: float,
    query_accuracy: float,
    relevant_token_retention: float,
    distractor_suppression_ratio: float,
    slow_query_coupling: float,
    gate_selectivity: float,
    match_selectivity: float,
    query_key_alignment: float,
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del mean_score_over_delays, delay_score_std, slow_fast_contribution_ratio, enabled_conn_count
    del retrieval_score, query_accuracy, relevant_token_retention, distractor_suppression_ratio, slow_query_coupling
    del match_selectivity, query_key_alignment
    del curriculum_enabled, curriculum_phase_1_delays, curriculum_phase_2_delays, curriculum_switch_generation, curriculum_phase, active_evaluation_delays
    normalized_score, score_bin, safe_score_ceiling = _normalized_score(final_max_score, score_ceiling)
    safe_selectivity = max(0.0, float(gate_selectivity))
    capped_selectivity = min(ARCHIVE_GATE_SELECTIVITY_MAX, safe_selectivity)
    selectivity_bin = _bin_value(capped_selectivity, upper_bound=ARCHIVE_GATE_SELECTIVITY_MAX, bin_count=ARCHIVE_GATE_SELECTIVITY_BINS)
    descriptor_key = f"scorebin_{score_bin}|gatebin_{selectivity_bin}"
    descriptor_values_json = stable_json_dumps(
        {
            "qd_profile": QD_PROFILE_GATING_MECHANISM,
            "descriptor_schema_version": ARCHIVE_DESCRIPTOR_VERSION_GATING_MECHANISM,
            "score_value": round(float(final_max_score), 10),
            "score_ceiling": round(safe_score_ceiling, 10),
            "normalized_score": round(normalized_score, 10),
            "score_bin": score_bin,
            "score_bin_count": ARCHIVE_SCORE_BINS,
            "gate_selectivity": round(safe_selectivity, 10),
            "gate_selectivity_capped": round(capped_selectivity, 10),
            "gate_selectivity_bin": selectivity_bin,
            "gate_selectivity_bin_count": ARCHIVE_GATE_SELECTIVITY_BINS,
        }
    )
    return ArchiveDescriptor(
        qd_profile=QD_PROFILE_GATING_MECHANISM,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_GATING_MECHANISM,
        descriptor_key=descriptor_key,
        descriptor_values_json=descriptor_values_json,
        score_bin=score_bin,
        secondary_bin=selectivity_bin,
    )


def _build_content_retrieval_descriptor(
    *,
    final_max_score: float,
    score_ceiling: float,
    mean_score_over_delays: float | None,
    delay_score_std: float,
    slow_fast_contribution_ratio: float,
    enabled_conn_count: int,
    retrieval_score: float,
    query_accuracy: float,
    relevant_token_retention: float,
    distractor_suppression_ratio: float,
    slow_query_coupling: float,
    gate_selectivity: float,
    match_selectivity: float,
    query_key_alignment: float,
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del mean_score_over_delays, delay_score_std, slow_fast_contribution_ratio, enabled_conn_count
    del retrieval_score, query_accuracy, relevant_token_retention, distractor_suppression_ratio, slow_query_coupling
    del gate_selectivity, query_key_alignment, curriculum_enabled, curriculum_phase_1_delays, curriculum_phase_2_delays, curriculum_switch_generation, curriculum_phase, active_evaluation_delays
    normalized_score, score_bin, safe_score_ceiling = _normalized_score(final_max_score, score_ceiling)
    safe_selectivity = max(0.0, float(match_selectivity))
    capped_selectivity = min(1.0, safe_selectivity)
    selectivity_bin = _bin_value(capped_selectivity, upper_bound=1.0, bin_count=8)
    descriptor_key = f"scorebin_{score_bin}|matchbin_{selectivity_bin}"
    descriptor_values_json = stable_json_dumps(
        {
            "qd_profile": QD_PROFILE_CONTENT_RETRIEVAL,
            "descriptor_schema_version": ARCHIVE_DESCRIPTOR_VERSION_CONTENT_RETRIEVAL,
            "score_value": round(float(final_max_score), 10),
            "score_ceiling": round(safe_score_ceiling, 10),
            "normalized_score": round(normalized_score, 10),
            "score_bin": score_bin,
            "score_bin_count": ARCHIVE_SCORE_BINS,
            "match_selectivity": round(safe_selectivity, 10),
            "match_selectivity_bin": selectivity_bin,
            "match_selectivity_bin_count": 8,
        }
    )
    return ArchiveDescriptor(
        qd_profile=QD_PROFILE_CONTENT_RETRIEVAL,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_CONTENT_RETRIEVAL,
        descriptor_key=descriptor_key,
        descriptor_values_json=descriptor_values_json,
        score_bin=score_bin,
        secondary_bin=selectivity_bin,
    )


def _build_kv_retrieval_mechanism_descriptor(
    *,
    final_max_score: float,
    score_ceiling: float,
    mean_score_over_delays: float | None,
    delay_score_std: float,
    slow_fast_contribution_ratio: float,
    enabled_conn_count: int,
    retrieval_score: float,
    query_accuracy: float,
    relevant_token_retention: float,
    distractor_suppression_ratio: float,
    slow_query_coupling: float,
    gate_selectivity: float,
    match_selectivity: float,
    query_key_alignment: float,
    store_vs_distractor_write_gap: float,
    readout_selectivity: float,
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del mean_score_over_delays, delay_score_std, slow_fast_contribution_ratio, enabled_conn_count
    del retrieval_score, query_accuracy, relevant_token_retention, distractor_suppression_ratio, slow_query_coupling
    del gate_selectivity, match_selectivity, query_key_alignment, curriculum_enabled, curriculum_phase_1_delays, curriculum_phase_2_delays, curriculum_switch_generation, curriculum_phase, active_evaluation_delays
    normalized_score, score_bin, safe_score_ceiling = _normalized_score(final_max_score, score_ceiling)
    safe_write_gap = max(-1.0, min(1.0, float(store_vs_distractor_write_gap)))
    write_gap_bin = _bin_value(safe_write_gap + 1.0, upper_bound=2.0, bin_count=8)
    descriptor_values_json = stable_json_dumps(
        {
            "qd_profile": QD_PROFILE_KV_RETRIEVAL_MECHANISM,
            "descriptor_schema_version": ARCHIVE_DESCRIPTOR_VERSION_KV_RETRIEVAL_MECHANISM,
            "score_value": round(float(final_max_score), 10),
            "score_ceiling": round(safe_score_ceiling, 10),
            "normalized_score": round(normalized_score, 10),
            "score_bin": score_bin,
            "score_bin_count": ARCHIVE_SCORE_BINS,
            "store_vs_distractor_write_gap": round(safe_write_gap, 10),
            "store_vs_distractor_write_gap_bin": write_gap_bin,
            "store_vs_distractor_write_gap_bin_count": 8,
            "readout_selectivity": round(max(0.0, float(readout_selectivity)), 10),
        }
    )
    descriptor_key = f"scorebin_{score_bin}|writegapbin_{write_gap_bin}"
    return ArchiveDescriptor(
        qd_profile=QD_PROFILE_KV_RETRIEVAL_MECHANISM,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_KV_RETRIEVAL_MECHANISM,
        descriptor_key=descriptor_key,
        descriptor_values_json=descriptor_values_json,
        score_bin=score_bin,
        secondary_bin=write_gap_bin,
    )


def _build_slot_retrieval_mechanism_descriptor(
    *,
    final_max_score: float,
    score_ceiling: float,
    mean_score_over_delays: float | None,
    delay_score_std: float,
    slow_fast_contribution_ratio: float,
    enabled_conn_count: int,
    retrieval_score: float,
    query_accuracy: float,
    relevant_token_retention: float,
    distractor_suppression_ratio: float,
    slow_query_coupling: float,
    gate_selectivity: float,
    match_selectivity: float,
    query_key_alignment: float,
    slot_write_focus: float,
    slot_query_focus: float,
    slot_utilization: float,
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del mean_score_over_delays, delay_score_std, slow_fast_contribution_ratio, enabled_conn_count
    del retrieval_score, query_accuracy, relevant_token_retention, distractor_suppression_ratio, slow_query_coupling
    del gate_selectivity, match_selectivity, query_key_alignment, curriculum_enabled, curriculum_phase_1_delays, curriculum_phase_2_delays, curriculum_switch_generation, curriculum_phase, active_evaluation_delays
    normalized_score, score_bin, safe_score_ceiling = _normalized_score(final_max_score, score_ceiling)
    combined_focus = max(0.0, min(1.0, 0.5 * (float(slot_write_focus) + float(slot_query_focus))))
    focus_bin = _bin_value(combined_focus, upper_bound=1.0, bin_count=8)
    descriptor_values_json = stable_json_dumps(
        {
            "qd_profile": QD_PROFILE_SLOT_RETRIEVAL_MECHANISM,
            "descriptor_schema_version": ARCHIVE_DESCRIPTOR_VERSION_SLOT_RETRIEVAL_MECHANISM,
            "score_value": round(float(final_max_score), 10),
            "score_ceiling": round(safe_score_ceiling, 10),
            "normalized_score": round(normalized_score, 10),
            "score_bin": score_bin,
            "score_bin_count": ARCHIVE_SCORE_BINS,
            "slot_focus": round(combined_focus, 10),
            "slot_focus_bin": focus_bin,
            "slot_focus_bin_count": 8,
            "slot_utilization": round(max(0.0, min(1.0, float(slot_utilization))), 10),
        }
    )
    descriptor_key = f"scorebin_{score_bin}|slotfocusbin_{focus_bin}"
    return ArchiveDescriptor(
        qd_profile=QD_PROFILE_SLOT_RETRIEVAL_MECHANISM,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_SLOT_RETRIEVAL_MECHANISM,
        descriptor_key=descriptor_key,
        descriptor_values_json=descriptor_values_json,
        score_bin=score_bin,
        secondary_bin=focus_bin,
    )


def _normalized_score(final_max_score: float, score_ceiling: float) -> tuple[float, int, float]:
    safe_score_ceiling = max(float(score_ceiling), 1e-6)
    normalized_score = max(0.0, min(1.0, float(final_max_score) / safe_score_ceiling))
    score_bin = _bin_value(normalized_score, upper_bound=1.0, bin_count=ARCHIVE_SCORE_BINS)
    return normalized_score, score_bin, safe_score_ceiling


def _bin_conn_count(enabled_conn_count: int) -> int:
    for index, upper_bound in enumerate(ARCHIVE_CONN_BIN_EDGES):
        if enabled_conn_count <= upper_bound:
            return index
    return ARCHIVE_CONN_BINS - 1


def _bin_value(value: float, *, upper_bound: float, bin_count: int) -> int:
    if bin_count <= 1:
        return 0
    clipped = max(0.0, min(float(upper_bound), float(value)))
    if upper_bound <= 0.0:
        return 0
    scaled = clipped / float(upper_bound)
    if scaled >= 1.0:
        return bin_count - 1
    return max(0, min(bin_count - 1, int(scaled * bin_count)))


ARCHIVE_PROFILES: dict[str, ArchiveProfileDefinition] = {
    QD_PROFILE_MECHANISM_V2: ArchiveProfileDefinition(
        name=QD_PROFILE_MECHANISM_V2,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_MECHANISM_V2,
        secondary_axis_label="slow_fast_contribution_ratio",
        secondary_axis_value_key="slow_fast_contribution_ratio",
        secondary_axis_bin_key="slow_ratio_bin",
        secondary_axis_bin_count=ARCHIVE_SLOW_RATIO_BINS,
        total_cells_per_delay=ARCHIVE_SCORE_BINS * ARCHIVE_SLOW_RATIO_BINS,
        builder=_build_mechanism_v2_descriptor,
    ),
    QD_PROFILE_GENERAL_COMPACTNESS: ArchiveProfileDefinition(
        name=QD_PROFILE_GENERAL_COMPACTNESS,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_GENERAL_COMPACTNESS,
        secondary_axis_label="enabled_conn_count",
        secondary_axis_value_key="enabled_conn_count",
        secondary_axis_bin_key="conn_bin",
        secondary_axis_bin_count=ARCHIVE_CONN_BINS,
        total_cells_per_delay=ARCHIVE_SCORE_BINS * ARCHIVE_CONN_BINS,
        builder=_build_general_compactness_descriptor,
    ),
    QD_PROFILE_DELAY_ROBUSTNESS: ArchiveProfileDefinition(
        name=QD_PROFILE_DELAY_ROBUSTNESS,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_DELAY_ROBUSTNESS,
        secondary_axis_label="delay_score_std",
        secondary_axis_value_key="delay_score_std",
        secondary_axis_bin_key="delay_std_bin",
        secondary_axis_bin_count=ARCHIVE_DELAY_STD_BINS,
        total_cells_per_delay=ARCHIVE_SCORE_BINS * ARCHIVE_DELAY_STD_BINS,
        builder=_build_delay_robustness_descriptor,
    ),
    QD_PROFILE_CURRICULUM_PROGRESS: ArchiveProfileDefinition(
        name=QD_PROFILE_CURRICULUM_PROGRESS,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_CURRICULUM_PROGRESS,
        secondary_axis_label="delay_score_std",
        secondary_axis_value_key="delay_score_std",
        secondary_axis_bin_key="delay_std_bin",
        secondary_axis_bin_count=ARCHIVE_DELAY_STD_BINS,
        total_cells_per_delay=ARCHIVE_SCORE_BINS * ARCHIVE_DELAY_STD_BINS * 2,
        builder=_build_curriculum_progress_descriptor,
    ),
    QD_PROFILE_RETRIEVAL_STRATEGY: ArchiveProfileDefinition(
        name=QD_PROFILE_RETRIEVAL_STRATEGY,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_RETRIEVAL_STRATEGY,
        secondary_axis_label="distractor_suppression_ratio",
        secondary_axis_value_key="distractor_suppression_ratio",
        secondary_axis_bin_key="suppression_bin",
        secondary_axis_bin_count=ARCHIVE_RETRIEVAL_SUPPRESSION_BINS,
        total_cells_per_delay=ARCHIVE_SCORE_BINS * ARCHIVE_RETRIEVAL_SUPPRESSION_BINS,
        builder=_build_retrieval_strategy_descriptor,
    ),
    QD_PROFILE_RETRIEVAL_MECHANISM: ArchiveProfileDefinition(
        name=QD_PROFILE_RETRIEVAL_MECHANISM,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_RETRIEVAL_MECHANISM,
        secondary_axis_label="slow_query_coupling",
        secondary_axis_value_key="slow_query_coupling",
        secondary_axis_bin_key="coupling_bin",
        secondary_axis_bin_count=ARCHIVE_RETRIEVAL_COUPLING_BINS,
        total_cells_per_delay=ARCHIVE_SCORE_BINS * ARCHIVE_RETRIEVAL_COUPLING_BINS,
        builder=_build_retrieval_mechanism_descriptor,
    ),
    QD_PROFILE_GATING_MECHANISM: ArchiveProfileDefinition(
        name=QD_PROFILE_GATING_MECHANISM,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_GATING_MECHANISM,
        secondary_axis_label="gate_selectivity",
        secondary_axis_value_key="gate_selectivity",
        secondary_axis_bin_key="gate_selectivity_bin",
        secondary_axis_bin_count=ARCHIVE_GATE_SELECTIVITY_BINS,
        total_cells_per_delay=ARCHIVE_SCORE_BINS * ARCHIVE_GATE_SELECTIVITY_BINS,
        builder=_build_gating_mechanism_descriptor,
    ),
    QD_PROFILE_CONTENT_RETRIEVAL: ArchiveProfileDefinition(
        name=QD_PROFILE_CONTENT_RETRIEVAL,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_CONTENT_RETRIEVAL,
        secondary_axis_label="match_selectivity",
        secondary_axis_value_key="match_selectivity",
        secondary_axis_bin_key="match_selectivity_bin",
        secondary_axis_bin_count=8,
        total_cells_per_delay=ARCHIVE_SCORE_BINS * 8,
        builder=_build_content_retrieval_descriptor,
    ),
    QD_PROFILE_KV_RETRIEVAL_MECHANISM: ArchiveProfileDefinition(
        name=QD_PROFILE_KV_RETRIEVAL_MECHANISM,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_KV_RETRIEVAL_MECHANISM,
        secondary_axis_label="store_vs_distractor_write_gap",
        secondary_axis_value_key="store_vs_distractor_write_gap",
        secondary_axis_bin_key="store_vs_distractor_write_gap_bin",
        secondary_axis_bin_count=8,
        total_cells_per_delay=ARCHIVE_SCORE_BINS * 8,
        builder=_build_kv_retrieval_mechanism_descriptor,
    ),
    QD_PROFILE_SLOT_RETRIEVAL_MECHANISM: ArchiveProfileDefinition(
        name=QD_PROFILE_SLOT_RETRIEVAL_MECHANISM,
        descriptor_schema_version=ARCHIVE_DESCRIPTOR_VERSION_SLOT_RETRIEVAL_MECHANISM,
        secondary_axis_label="slot_focus",
        secondary_axis_value_key="slot_focus",
        secondary_axis_bin_key="slot_focus_bin",
        secondary_axis_bin_count=8,
        total_cells_per_delay=ARCHIVE_SCORE_BINS * 8,
        builder=_build_slot_retrieval_mechanism_descriptor,
    ),
}
