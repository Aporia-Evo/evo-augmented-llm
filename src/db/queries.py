from __future__ import annotations

import json
import math
from typing import Any

from db.client import SpacetimeHttpClient
from db.models import (
    ActiveCandidateRecord,
    ArchiveCellRecord,
    ArchiveEventRecord,
    CandidateRecord,
    CandidateFeatureRecord,
    CandidateFeatureVectorRecord,
    CandidateLifecycleEventRecord,
    CheckpointRecord,
    EliteRecord,
    EvaluationJobRecord,
    EvaluationResultRecord,
    EventRecord,
    FitnessRecord,
    GenerationRecord,
    HallOfFameEntryRecord,
    OnlineMetricRecord,
    OnlineStateRecord,
    RunRecord,
)


def sql_quote(value: str) -> str:
    return value.replace("'", "''")


def get_run(client: SpacetimeHttpClient, run_id: str) -> RunRecord | None:
    rows = client.sql(
        "SELECT run_id, task_name, seed, status, config_json, created_at, finished_at, mode "
        f"FROM runs WHERE run_id = '{sql_quote(run_id)}';"
    )
    if not rows:
        return None
    return _run_from_row(rows[0])


def list_runs(client: SpacetimeHttpClient, limit: int = 10) -> list[RunRecord]:
    rows = client.sql(
        "SELECT run_id, task_name, seed, status, config_json, created_at, finished_at, mode FROM runs;"
    )
    records = [_run_from_row(row) for row in rows]
    records.sort(key=lambda run: run.created_at, reverse=True)
    return records[:limit]


def list_generations(client: SpacetimeHttpClient, run_id: str) -> list[GenerationRecord]:
    rows = client.sql(
        "SELECT run_id, generation_id, state, best_candidate_id, best_score, avg_score, created_at, committed_at, "
        "eval_duration_ms, commit_duration_ms "
        f"FROM generations WHERE run_id = '{sql_quote(run_id)}';"
    )
    records = [_generation_from_row(row) for row in rows]
    records.sort(key=lambda generation: generation.generation_id)
    return records


def get_generation(client: SpacetimeHttpClient, run_id: str, generation_id: int) -> GenerationRecord | None:
    rows = client.sql(
        "SELECT run_id, generation_id, state, best_candidate_id, best_score, avg_score, created_at, committed_at, "
        "eval_duration_ms, commit_duration_ms "
        f"FROM generations WHERE run_id = '{sql_quote(run_id)}' AND generation_id = {generation_id};"
    )
    if not rows:
        return None
    return _generation_from_row(rows[0])


def list_candidates(client: SpacetimeHttpClient, run_id: str, generation_id: int) -> list[CandidateRecord]:
    rows = client.sql(
        "SELECT candidate_id, run_id, generation_id, genome_blob, status, parent_ids_json, created_at "
        f"FROM candidates WHERE run_id = '{sql_quote(run_id)}' AND generation_id = {generation_id};"
    )
    records = [
        CandidateRecord(
            candidate_id=row["candidate_id"],
            run_id=row["run_id"],
            generation_id=int(row["generation_id"]),
            genome_blob=row["genome_blob"],
            status=row["status"],
            parent_ids=json.loads(row["parent_ids_json"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]
    records.sort(key=lambda candidate: candidate.candidate_id)
    return records


def list_fitness(client: SpacetimeHttpClient, run_id: str, generation_id: int) -> list[FitnessRecord]:
    rows = client.sql(
        "SELECT candidate_id, run_id, generation_id, score, raw_metrics_json, evaluated_at "
        f"FROM fitness WHERE run_id = '{sql_quote(run_id)}' AND generation_id = {generation_id};"
    )
    records = [
        FitnessRecord(
            candidate_id=row["candidate_id"],
            run_id=row["run_id"],
            generation_id=int(row["generation_id"]),
            score=float(row["score"]),
            raw_metrics=json.loads(row["raw_metrics_json"]),
            evaluated_at=row["evaluated_at"],
        )
        for row in rows
    ]
    records.sort(key=lambda record: record.candidate_id)
    return records


def get_checkpoint(client: SpacetimeHttpClient, run_id: str, generation_id: int) -> CheckpointRecord | None:
    rows = client.sql(
        "SELECT run_id, generation_id, state_blob, parent_ids_json, created_at "
        f"FROM checkpoints WHERE run_id = '{sql_quote(run_id)}' AND generation_id = {generation_id};"
    )
    if not rows:
        return None
    row = rows[0]
    return CheckpointRecord(
        run_id=row["run_id"],
        generation_id=int(row["generation_id"]),
        state_blob=row["state_blob"],
        parent_ids_json=row["parent_ids_json"],
        created_at=row["created_at"],
    )


def list_elites(
    client: SpacetimeHttpClient,
    run_id: str,
    generation_id: int | None = None,
    limit: int = 10,
) -> list[EliteRecord]:
    where = f"run_id = '{sql_quote(run_id)}'"
    if generation_id is not None:
        where += f" AND source_generation = {generation_id}"
    rows = client.sql(
        "SELECT elite_id, run_id, source_generation, candidate_id, rank, score, frozen_genome_blob, archived_at "
        f"FROM elite_archive WHERE {where};"
    )
    records = [
        EliteRecord(
            elite_id=row["elite_id"],
            run_id=row["run_id"],
            source_generation=int(row["source_generation"]),
            candidate_id=row["candidate_id"],
            rank=int(row["rank"]),
            score=float(row["score"]),
            frozen_genome_blob=row["frozen_genome_blob"],
            archived_at=row["archived_at"],
        )
        for row in rows
    ]
    records.sort(key=lambda elite: (-elite.source_generation, elite.rank))
    return records[:limit]


def list_events(client: SpacetimeHttpClient, run_id: str, limit: int = 20) -> list[EventRecord]:
    rows = client.sql(
        "SELECT event_id, run_id, type, payload_json, created_at "
        f"FROM events WHERE run_id = '{sql_quote(run_id)}';"
    )
    records = [
        EventRecord(
            event_id=row["event_id"],
            run_id=row["run_id"],
            type=row["type"],
            payload_json=row["payload_json"],
            created_at=row["created_at"],
        )
        for row in rows
    ]
    records.sort(key=lambda event: event.created_at, reverse=True)
    return records[:limit]


def list_active_candidates(
    client: SpacetimeHttpClient,
    run_id: str,
    statuses: list[str] | None = None,
) -> list[ActiveCandidateRecord]:
    where = f"run_id = '{sql_quote(run_id)}'"
    if statuses:
        clauses = " OR ".join(f"status = '{sql_quote(status)}'" for status in statuses)
        where += f" AND ({clauses})"
    rows = client.sql(
        "SELECT candidate_id, run_id, slot_index, variant, genome_blob, status, rolling_score, eval_count, birth_step, "
        "last_eval_at, parent_ids_json, created_at "
        f"FROM active_candidates WHERE {where};"
    )
    records = [_active_candidate_from_row(row) for row in rows]
    records.sort(key=lambda record: record.slot_index)
    return records


def get_active_candidate(client: SpacetimeHttpClient, candidate_id: str) -> ActiveCandidateRecord | None:
    rows = client.sql(
        "SELECT candidate_id, run_id, slot_index, variant, genome_blob, status, rolling_score, eval_count, birth_step, "
        "last_eval_at, parent_ids_json, created_at "
        f"FROM active_candidates WHERE candidate_id = '{sql_quote(candidate_id)}';"
    )
    if not rows:
        return None
    return _active_candidate_from_row(rows[0])


def list_evaluation_jobs(
    client: SpacetimeHttpClient,
    run_id: str,
    status: str | None = None,
    limit: int = 200,
) -> list[EvaluationJobRecord]:
    where = f"run_id = '{sql_quote(run_id)}'"
    if status is not None:
        where += f" AND status = '{sql_quote(status)}'"
    rows = client.sql(
        "SELECT job_id, run_id, candidate_id, task_payload_json, status, claimed_by, created_at, claimed_at, finished_at "
        f"FROM evaluation_jobs WHERE {where};"
    )
    records = [_job_from_row(row) for row in rows]
    records.sort(key=lambda record: (record.created_at, record.job_id))
    return records[:limit]


def get_evaluation_job(client: SpacetimeHttpClient, job_id: str) -> EvaluationJobRecord | None:
    rows = client.sql(
        "SELECT job_id, run_id, candidate_id, task_payload_json, status, claimed_by, created_at, claimed_at, finished_at "
        f"FROM evaluation_jobs WHERE job_id = '{sql_quote(job_id)}';"
    )
    if not rows:
        return None
    return _job_from_row(rows[0])


def list_evaluation_results(
    client: SpacetimeHttpClient,
    run_id: str,
    limit: int = 200,
) -> list[EvaluationResultRecord]:
    rows = client.sql(
        "SELECT result_id, run_id, candidate_id, score, raw_metrics_json, created_at "
        f"FROM evaluation_results WHERE run_id = '{sql_quote(run_id)}';"
    )
    records = [
        EvaluationResultRecord(
            result_id=row["result_id"],
            run_id=row["run_id"],
            candidate_id=row["candidate_id"],
            score=float(row["score"]),
            raw_metrics=json.loads(row["raw_metrics_json"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]
    records.sort(key=lambda record: record.created_at, reverse=True)
    return records[:limit]


def list_hall_of_fame(
    client: SpacetimeHttpClient,
    run_id: str,
    limit: int = 50,
) -> list[HallOfFameEntryRecord]:
    rows = client.sql(
        "SELECT entry_id, run_id, candidate_id, score, frozen_genome_blob, inserted_at "
        f"FROM hall_of_fame WHERE run_id = '{sql_quote(run_id)}';"
    )
    records = [
        HallOfFameEntryRecord(
            entry_id=row["entry_id"],
            run_id=row["run_id"],
            candidate_id=row["candidate_id"],
            score=float(row["score"]),
            frozen_genome_blob=row["frozen_genome_blob"],
            inserted_at=row["inserted_at"],
        )
        for row in rows
    ]
    records.sort(key=lambda entry: (-entry.score, entry.inserted_at, entry.entry_id))
    return records[:limit]


def list_candidate_lifecycle_events(
    client: SpacetimeHttpClient,
    run_id: str,
    candidate_id: str | None = None,
    limit: int = 100,
) -> list[CandidateLifecycleEventRecord]:
    where = f"run_id = '{sql_quote(run_id)}'"
    if candidate_id is not None:
        where += f" AND candidate_id = '{sql_quote(candidate_id)}'"
    rows = client.sql(
        "SELECT event_id, run_id, candidate_id, event_type, payload_json, created_at "
        f"FROM candidate_lifecycle_events WHERE {where};"
    )
    records = [
        CandidateLifecycleEventRecord(
            event_id=row["event_id"],
            run_id=row["run_id"],
            candidate_id=row["candidate_id"],
            event_type=row["event_type"],
            payload_json=row["payload_json"],
            created_at=row["created_at"],
        )
        for row in rows
    ]
    records.sort(key=lambda record: record.created_at, reverse=True)
    return records[:limit]


def list_online_metrics(client: SpacetimeHttpClient, run_id: str, limit: int = 100) -> list[OnlineMetricRecord]:
    rows = client.sql(
        "SELECT metric_id, run_id, timestamp, active_population_size, rolling_best_score, rolling_avg_score, "
        "replacement_count, success_rate_window "
        f"FROM online_metrics WHERE run_id = '{sql_quote(run_id)}';"
    )
    records = [
        OnlineMetricRecord(
            metric_id=row["metric_id"],
            run_id=row["run_id"],
            timestamp=row["timestamp"],
            active_population_size=int(row["active_population_size"]),
            rolling_best_score=float(row["rolling_best_score"]),
            rolling_avg_score=float(row["rolling_avg_score"]),
            replacement_count=int(row["replacement_count"]),
            success_rate_window=float(row["success_rate_window"]),
        )
        for row in rows
    ]
    records.sort(key=lambda record: record.timestamp, reverse=True)
    return records[:limit]


def get_online_state(client: SpacetimeHttpClient, run_id: str) -> OnlineStateRecord | None:
    rows = client.sql(
        "SELECT run_id, step, replacement_count, success_window_json, adapter_state_blob, created_at, updated_at "
        f"FROM online_state WHERE run_id = '{sql_quote(run_id)}';"
    )
    if not rows:
        return None
    row = rows[0]
    return OnlineStateRecord(
        run_id=row["run_id"],
        step=int(row["step"]),
        replacement_count=int(row["replacement_count"]),
        success_window_json=row["success_window_json"],
        adapter_state_blob=row["adapter_state_blob"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def list_candidate_features(
    client: SpacetimeHttpClient,
    *,
    benchmark_label: str | None = None,
    task_name: str | None = None,
    variant: str | None = None,
    delay_steps: int | None = None,
    run_id: str | None = None,
    candidate_id: str | None = None,
) -> list[CandidateFeatureRecord]:
    where = _compose_where_clause(
        run_id=run_id,
        benchmark_label=benchmark_label,
        task_name=task_name,
        variant=variant,
        delay_steps=delay_steps,
        candidate_id=candidate_id,
    )
    rows = client.sql(
        "SELECT candidate_id, run_id, benchmark_label, task_name, delay_steps, variant, seed, generation, "
        "hof_flag, success, final_max_score, first_success_generation, mean_alpha, std_alpha, mean_eta, std_eta, "
        "mean_plastic_d, std_plastic_d, plastic_d_at_lower_bound_fraction, plastic_d_at_zero_fraction, node_count, "
        "enabled_conn_count, mean_abs_delta_w, max_abs_delta_w, clamp_hit_rate, plasticity_active_fraction, "
        "mean_abs_fast_state, mean_abs_slow_state, slow_fast_contribution_ratio, "
        "mean_abs_decay_term, max_abs_decay_term, decay_effect_ratio, decay_near_zero_fraction, "
        "score_delay_3, score_delay_5, score_delay_8, success_delay_3, success_delay_5, success_delay_8, "
        "mean_score_over_delays, delay_score_std, delay_score_range, "
        "curriculum_enabled, curriculum_phase_1_delays, curriculum_phase_2_delays, curriculum_switch_generation, "
        "curriculum_phase, active_evaluation_delays, score_current_phase, "
        "query_accuracy, retrieval_score, exact_match_success, mean_query_distance, distractor_load, "
        "num_stores, num_queries, num_distractors, retrieval_margin, retrieval_confusion_rate, "
        "relevant_token_retention, query_response_margin, distractor_suppression_ratio, "
        "correct_key_selected, correct_value_selected, query_key_match_score, value_margin, distractor_competition_score, "
        "mean_abs_fast_state_during_store, mean_abs_slow_state_during_store, "
        "mean_abs_fast_state_during_query, mean_abs_slow_state_during_query, "
        "mean_abs_fast_state_during_distractor, mean_abs_slow_state_during_distractor, "
        "slow_query_coupling, store_query_state_gap, slow_fast_retrieval_ratio, retrieval_state_alignment "
        f"FROM candidate_features{where};"
    )
    records = [
        CandidateFeatureRecord(
            candidate_id=row["candidate_id"],
            run_id=row["run_id"],
            benchmark_label=row["benchmark_label"],
            task_name=row["task_name"],
            delay_steps=int(row["delay_steps"]),
            variant=row["variant"],
            seed=int(row["seed"]),
            generation=int(row["generation"]),
            hof_flag=bool(row["hof_flag"]),
            success=bool(row["success"]),
            final_max_score=float(row["final_max_score"]),
            first_success_generation=_none_if_negative_int(row.get("first_success_generation")),
            mean_alpha=float(row["mean_alpha"]),
            std_alpha=float(row["std_alpha"]),
            mean_eta=float(row["mean_eta"]),
            std_eta=float(row["std_eta"]),
            mean_plastic_d=float(row["mean_plastic_d"]),
            std_plastic_d=float(row["std_plastic_d"]),
            plastic_d_at_lower_bound_fraction=float(row["plastic_d_at_lower_bound_fraction"]),
            plastic_d_at_zero_fraction=float(row["plastic_d_at_zero_fraction"]),
            node_count=int(row["node_count"]),
            enabled_conn_count=int(row["enabled_conn_count"]),
            mean_abs_delta_w=float(row["mean_abs_delta_w"]),
            max_abs_delta_w=float(row["max_abs_delta_w"]),
            clamp_hit_rate=float(row["clamp_hit_rate"]),
            plasticity_active_fraction=float(row["plasticity_active_fraction"]),
            mean_abs_fast_state=float(row.get("mean_abs_fast_state", 0.0)),
            mean_abs_slow_state=float(row.get("mean_abs_slow_state", 0.0)),
            slow_fast_contribution_ratio=float(row.get("slow_fast_contribution_ratio", 0.0)),
            mean_abs_decay_term=float(row.get("mean_abs_decay_term", 0.0)),
            max_abs_decay_term=float(row.get("max_abs_decay_term", 0.0)),
            decay_effect_ratio=float(row.get("decay_effect_ratio", 0.0)),
            decay_near_zero_fraction=float(row.get("decay_near_zero_fraction", 0.0)),
            score_delay_3=float(row.get("score_delay_3", 0.0)),
            score_delay_5=float(row.get("score_delay_5", 0.0)),
            score_delay_8=float(row.get("score_delay_8", 0.0)),
            success_delay_3=bool(row.get("success_delay_3", False)),
            success_delay_5=bool(row.get("success_delay_5", False)),
            success_delay_8=bool(row.get("success_delay_8", False)),
            mean_score_over_delays=float(row.get("mean_score_over_delays", row["final_max_score"])),
            delay_score_std=float(row.get("delay_score_std", 0.0)),
            delay_score_range=float(row.get("delay_score_range", 0.0)),
            curriculum_enabled=bool(row.get("curriculum_enabled", False)),
            curriculum_phase_1_delays=row.get("curriculum_phase_1_delays", ""),
            curriculum_phase_2_delays=row.get("curriculum_phase_2_delays", ""),
            curriculum_switch_generation=int(row.get("curriculum_switch_generation", 0)),
            curriculum_phase=row.get("curriculum_phase", "static"),
            active_evaluation_delays=row.get("active_evaluation_delays", ""),
            score_current_phase=float(row.get("score_current_phase", row["final_max_score"])),
            query_accuracy=float(row.get("query_accuracy", 0.0)),
            retrieval_score=float(row.get("retrieval_score", 0.0)),
            exact_match_success=bool(row.get("exact_match_success", False)),
            mean_query_distance=float(row.get("mean_query_distance", 0.0)),
            distractor_load=float(row.get("distractor_load", 0.0)),
            num_stores=float(row.get("num_stores", 0.0)),
            num_queries=float(row.get("num_queries", 0.0)),
            num_distractors=float(row.get("num_distractors", 0.0)),
            retrieval_margin=float(row.get("retrieval_margin", 0.0)),
            retrieval_confusion_rate=float(row.get("retrieval_confusion_rate", 0.0)),
            relevant_token_retention=float(row.get("relevant_token_retention", 0.0)),
            query_response_margin=float(row.get("query_response_margin", 0.0)),
            distractor_suppression_ratio=float(row.get("distractor_suppression_ratio", 0.0)),
            correct_key_selected=float(row.get("correct_key_selected", 0.0)),
            correct_value_selected=float(row.get("correct_value_selected", 0.0)),
            query_key_match_score=float(row.get("query_key_match_score", 0.0)),
            value_margin=float(row.get("value_margin", 0.0)),
            distractor_competition_score=float(row.get("distractor_competition_score", 0.0)),
            mean_abs_fast_state_during_store=float(row.get("mean_abs_fast_state_during_store", 0.0)),
            mean_abs_slow_state_during_store=float(row.get("mean_abs_slow_state_during_store", 0.0)),
            mean_abs_fast_state_during_query=float(row.get("mean_abs_fast_state_during_query", 0.0)),
            mean_abs_slow_state_during_query=float(row.get("mean_abs_slow_state_during_query", 0.0)),
            mean_abs_fast_state_during_distractor=float(row.get("mean_abs_fast_state_during_distractor", 0.0)),
            mean_abs_slow_state_during_distractor=float(row.get("mean_abs_slow_state_during_distractor", 0.0)),
            slow_query_coupling=float(row.get("slow_query_coupling", 0.0)),
            store_query_state_gap=float(row.get("store_query_state_gap", 0.0)),
            slow_fast_retrieval_ratio=float(row.get("slow_fast_retrieval_ratio", 0.0)),
            retrieval_state_alignment=float(row.get("retrieval_state_alignment", 0.0)),
        )
        for row in rows
    ]
    records.sort(key=lambda record: (record.task_name, record.delay_steps, record.variant, record.seed, record.generation, record.candidate_id))
    return records


def list_candidate_feature_vectors(
    client: SpacetimeHttpClient,
    *,
    candidate_ids: list[str] | None = None,
    feature_version: str | None = None,
) -> list[CandidateFeatureVectorRecord]:
    where_clauses: list[str] = []
    if candidate_ids:
        quoted = ", ".join(f"'{sql_quote(candidate_id)}'" for candidate_id in candidate_ids)
        where_clauses.append(f"candidate_id IN ({quoted})")
    if feature_version is not None:
        where_clauses.append(f"feature_version = '{sql_quote(feature_version)}'")
    where = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    rows = client.sql(
        "SELECT candidate_id, feature_version, vector_json, norm_l2 "
        f"FROM candidate_feature_vectors{where};"
    )
    records = [
        CandidateFeatureVectorRecord(
            candidate_id=row["candidate_id"],
            feature_version=row["feature_version"],
            vector_json=row["vector_json"],
            norm_l2=float(row["norm_l2"]),
        )
        for row in rows
    ]
    records.sort(key=lambda record: (record.feature_version, record.candidate_id))
    return records


def list_archive_cells(
    client: SpacetimeHttpClient,
    *,
    benchmark_label: str | None = None,
    task_name: str | None = None,
    variant: str | None = None,
    delay_steps: int | None = None,
    qd_profile: str | None = None,
) -> list[ArchiveCellRecord]:
    where = _compose_where_clause(
        benchmark_label=benchmark_label,
        task_name=task_name,
        variant=variant,
        delay_steps=delay_steps,
        qd_profile=qd_profile,
    )
    rows = client.sql(
        "SELECT archive_id, benchmark_label, task_name, delay_steps, variant, qd_profile, "
        "descriptor_schema_version, descriptor_key, descriptor_values_json, elite_candidate_id, "
        "curriculum_enabled, curriculum_phase_1_delays, curriculum_phase_2_delays, curriculum_switch_generation, "
        "elite_score, elite_run_id, updated_at "
        f"FROM archive_cells{where};"
    )
    records = [
        ArchiveCellRecord(
            archive_id=row["archive_id"],
            benchmark_label=row["benchmark_label"],
            task_name=row["task_name"],
            delay_steps=int(row["delay_steps"]),
            variant=row["variant"],
            descriptor_key=row["descriptor_key"],
            descriptor_values_json=row["descriptor_values_json"],
            elite_candidate_id=row["elite_candidate_id"],
            elite_score=float(row["elite_score"]),
            elite_run_id=row["elite_run_id"],
            updated_at=row["updated_at"],
            qd_profile=_none_if_blank(row.get("qd_profile")) or "mechanism_v2",
            descriptor_schema_version=_none_if_blank(row.get("descriptor_schema_version")) or "v7a-qdlight-v1",
            curriculum_enabled=bool(row.get("curriculum_enabled", False)),
            curriculum_phase_1_delays=row.get("curriculum_phase_1_delays", ""),
            curriculum_phase_2_delays=row.get("curriculum_phase_2_delays", ""),
            curriculum_switch_generation=int(row.get("curriculum_switch_generation", 0)),
        )
        for row in rows
    ]
    records.sort(key=lambda record: (record.task_name, record.delay_steps, record.variant, record.qd_profile, -record.elite_score, record.archive_id))
    return records


def list_archive_events(
    client: SpacetimeHttpClient,
    *,
    benchmark_label: str | None = None,
    task_name: str | None = None,
    variant: str | None = None,
    delay_steps: int | None = None,
    qd_profile: str | None = None,
) -> list[ArchiveEventRecord]:
    where = _compose_where_clause(
        benchmark_label=benchmark_label,
        task_name=task_name,
        variant=variant,
        delay_steps=delay_steps,
        qd_profile=qd_profile,
    )
    rows = client.sql(
        "SELECT event_id, archive_id, benchmark_label, task_name, delay_steps, variant, qd_profile, "
        "descriptor_schema_version, descriptor_key, candidate_id, event_type, score, created_at, "
        "curriculum_enabled, curriculum_phase_1_delays, curriculum_phase_2_delays, curriculum_switch_generation "
        f"FROM archive_events{where};"
    )
    records = [
        ArchiveEventRecord(
            event_id=row["event_id"],
            archive_id=row["archive_id"],
            benchmark_label=row["benchmark_label"],
            task_name=row["task_name"],
            delay_steps=int(row["delay_steps"]),
            variant=row["variant"],
            descriptor_key=row["descriptor_key"],
            candidate_id=row["candidate_id"],
            event_type=row["event_type"],
            score=float(row["score"]),
            created_at=row["created_at"],
            qd_profile=_none_if_blank(row.get("qd_profile")) or "mechanism_v2",
            descriptor_schema_version=_none_if_blank(row.get("descriptor_schema_version")) or "v7a-qdlight-v1",
            curriculum_enabled=bool(row.get("curriculum_enabled", False)),
            curriculum_phase_1_delays=row.get("curriculum_phase_1_delays", ""),
            curriculum_phase_2_delays=row.get("curriculum_phase_2_delays", ""),
            curriculum_switch_generation=int(row.get("curriculum_switch_generation", 0)),
        )
        for row in rows
    ]
    records.sort(key=lambda record: (record.task_name, record.delay_steps, record.variant, record.qd_profile, record.created_at, record.event_id))
    return records


def _run_from_row(row: dict[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        task_name=row["task_name"],
        seed=int(row["seed"]),
        status=row["status"],
        config_json=row["config_json"],
        created_at=row["created_at"],
        finished_at=_none_if_blank(row.get("finished_at")),
        mode=_none_if_blank(row.get("mode")) or "generation",
    )


def _generation_from_row(row: dict[str, Any]) -> GenerationRecord:
    return GenerationRecord(
        run_id=row["run_id"],
        generation_id=int(row["generation_id"]),
        state=row["state"],
        best_candidate_id=_none_if_blank(row.get("best_candidate_id")),
        best_score=_none_if_nan(row.get("best_score")),
        avg_score=_none_if_nan(row.get("avg_score")),
        created_at=row["created_at"],
        committed_at=_none_if_blank(row.get("committed_at")),
        eval_duration_ms=_none_if_negative_int(row.get("eval_duration_ms")),
        commit_duration_ms=_none_if_negative_int(row.get("commit_duration_ms")),
    )


def _active_candidate_from_row(row: dict[str, Any]) -> ActiveCandidateRecord:
    return ActiveCandidateRecord(
        candidate_id=row["candidate_id"],
        run_id=row["run_id"],
        slot_index=int(row["slot_index"]),
        variant=row["variant"],
        genome_blob=row["genome_blob"],
        status=row["status"],
        rolling_score=float(row["rolling_score"]),
        eval_count=int(row["eval_count"]),
        birth_step=int(row["birth_step"]),
        last_eval_at=_none_if_blank(row.get("last_eval_at")),
        parent_ids=json.loads(row["parent_ids_json"]),
        created_at=row["created_at"],
    )


def _job_from_row(row: dict[str, Any]) -> EvaluationJobRecord:
    return EvaluationJobRecord(
        job_id=row["job_id"],
        run_id=row["run_id"],
        candidate_id=row["candidate_id"],
        task_payload_json=row["task_payload_json"],
        status=row["status"],
        claimed_by=_none_if_blank(row.get("claimed_by")),
        created_at=row["created_at"],
        claimed_at=_none_if_blank(row.get("claimed_at")),
        finished_at=_none_if_blank(row.get("finished_at")),
    )


def _compose_where_clause(
    *,
    run_id: str | None = None,
    benchmark_label: str | None = None,
    task_name: str | None = None,
    variant: str | None = None,
    delay_steps: int | None = None,
    candidate_id: str | None = None,
    qd_profile: str | None = None,
) -> str:
    clauses: list[str] = []
    if run_id is not None:
        clauses.append(f"run_id = '{sql_quote(run_id)}'")
    if benchmark_label is not None:
        clauses.append(f"benchmark_label = '{sql_quote(benchmark_label)}'")
    if task_name is not None:
        clauses.append(f"task_name = '{sql_quote(task_name)}'")
    if variant is not None:
        clauses.append(f"variant = '{sql_quote(variant)}'")
    if delay_steps is not None:
        clauses.append(f"delay_steps = {int(delay_steps)}")
    if candidate_id is not None:
        clauses.append(f"candidate_id = '{sql_quote(candidate_id)}'")
    if qd_profile is not None:
        clauses.append(f"qd_profile = '{sql_quote(qd_profile)}'")
    if not clauses:
        return ""
    return f" WHERE {' AND '.join(clauses)}"


def _none_if_blank(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _none_if_nan(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return number


def _none_if_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number < 0:
        return None
    return number
