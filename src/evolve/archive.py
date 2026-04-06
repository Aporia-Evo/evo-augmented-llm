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

QD_PROFILE_MECHANISM_V2 = "mechanism_v2"
QD_PROFILE_GENERAL_COMPACTNESS = "general_compactness"
QD_PROFILE_DELAY_ROBUSTNESS = "delay_robustness"
QD_PROFILE_CURRICULUM_PROGRESS = "curriculum_progress"
QD_PROFILE_RETRIEVAL_STRATEGY = "retrieval_strategy"
QD_PROFILE_RETRIEVAL_MECHANISM = "retrieval_mechanism"
DEFAULT_QD_PROFILE = QD_PROFILE_MECHANISM_V2

ARCHIVE_DESCRIPTOR_VERSION_MECHANISM_V2 = "v7a-qdlight-v1"
ARCHIVE_DESCRIPTOR_VERSION_GENERAL_COMPACTNESS = "v7b-qdlight-v1"
ARCHIVE_DESCRIPTOR_VERSION_DELAY_ROBUSTNESS = "v8a-qdlight-v1"
ARCHIVE_DESCRIPTOR_VERSION_CURRICULUM_PROGRESS = "v8b-qdlight-v1"
ARCHIVE_DESCRIPTOR_VERSION_RETRIEVAL_STRATEGY = "v9a-retrieval-v1"
ARCHIVE_DESCRIPTOR_VERSION_RETRIEVAL_MECHANISM = "v9b-retrieval-v1"


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
    curriculum_enabled: bool = False,
    curriculum_phase_1_delays: str = "",
    curriculum_phase_2_delays: str = "",
    curriculum_switch_generation: int = 0,
    curriculum_phase: str = "static",
    active_evaluation_delays: str = "",
    qd_profile: str = DEFAULT_QD_PROFILE,
) -> ArchiveDescriptor:
    profile = archive_profile_definition(qd_profile)
    return profile.builder(
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
        curriculum_enabled=curriculum_enabled,
        curriculum_phase_1_delays=curriculum_phase_1_delays,
        curriculum_phase_2_delays=curriculum_phase_2_delays,
        curriculum_switch_generation=curriculum_switch_generation,
        curriculum_phase=curriculum_phase,
        active_evaluation_delays=active_evaluation_delays,
    )


def archive_profile_names(*, task_name: str | None = None) -> tuple[str, ...]:
    profile_names = list(ARCHIVE_PROFILES.keys())
    if task_name != "key_value_memory":
        profile_names = [
            profile_name
            for profile_name in profile_names
            if profile_name not in {QD_PROFILE_RETRIEVAL_STRATEGY, QD_PROFILE_RETRIEVAL_MECHANISM}
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
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del slow_fast_contribution_ratio, enabled_conn_count
    del retrieval_score, query_accuracy, relevant_token_retention, distractor_suppression_ratio, slow_query_coupling
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
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del slow_fast_contribution_ratio, enabled_conn_count
    del retrieval_score, query_accuracy, relevant_token_retention, distractor_suppression_ratio, slow_query_coupling
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
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del mean_score_over_delays, delay_score_std, slow_fast_contribution_ratio, enabled_conn_count
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
    curriculum_enabled: bool,
    curriculum_phase_1_delays: str,
    curriculum_phase_2_delays: str,
    curriculum_switch_generation: int,
    curriculum_phase: str,
    active_evaluation_delays: str,
) -> ArchiveDescriptor:
    del mean_score_over_delays, delay_score_std, slow_fast_contribution_ratio, enabled_conn_count
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
}
