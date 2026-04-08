from __future__ import annotations

from pathlib import Path

from analysis.archive import (
    load_archive_cells_from_jsonl,
    load_archive_events_from_jsonl,
    write_archive_cells_jsonl,
    write_archive_events_jsonl,
)
from db.models import ArchiveCellRecord, ArchiveEventRecord, CandidateFeatureRecord
from db.reducers import InMemoryRepository
from evolve.archive import (
    DEFAULT_QD_PROFILE,
    QD_PROFILE_CURRICULUM_PROGRESS,
    QD_PROFILE_DELAY_ROBUSTNESS,
    QD_PROFILE_GENERAL_COMPACTNESS,
    QD_PROFILE_CONTENT_RETRIEVAL,
    QD_PROFILE_DELTA_MEMORY_MECHANISM,
    QD_PROFILE_GATING_MECHANISM,
    QD_PROFILE_KV_RETRIEVAL_MECHANISM,
    QD_PROFILE_SLOT_RETRIEVAL_MECHANISM,
    QD_PROFILE_RETRIEVAL_MECHANISM,
    QD_PROFILE_RETRIEVAL_STRATEGY,
    build_archive_cell,
    build_archive_descriptor,
)
from ui.cli import main


def _feature_record(*, candidate_id: str, variant: str = "stateful_v2") -> CandidateFeatureRecord:
    return CandidateFeatureRecord(
        candidate_id=candidate_id,
        run_id="run-1",
        benchmark_label="bench-v7",
        task_name="bit_memory",
        delay_steps=8,
        variant=variant,
        seed=7,
        generation=5,
        hof_flag=True,
        success=True,
        final_max_score=4.0,
        first_success_generation=5,
        mean_alpha=0.4,
        std_alpha=0.1,
        mean_eta=0.0,
        std_eta=0.0,
        mean_plastic_d=0.0,
        std_plastic_d=0.0,
        plastic_d_at_lower_bound_fraction=0.0,
        plastic_d_at_zero_fraction=1.0,
        node_count=5,
        enabled_conn_count=4,
        mean_abs_delta_w=0.0,
        max_abs_delta_w=0.0,
        clamp_hit_rate=0.0,
        plasticity_active_fraction=0.0,
        mean_abs_fast_state=3.6,
        mean_abs_slow_state=4.2,
        slow_fast_contribution_ratio=3.7,
        mean_abs_decay_term=0.0,
        max_abs_decay_term=0.0,
        decay_effect_ratio=0.0,
        decay_near_zero_fraction=0.0,
        score_delay_3=0.0,
        score_delay_5=3.8,
        score_delay_8=4.0,
        success_delay_3=False,
        success_delay_5=True,
        success_delay_8=True,
        mean_score_over_delays=3.9,
        delay_score_std=0.1,
        delay_score_range=0.2,
        curriculum_enabled=True,
        curriculum_phase_1_delays="5",
        curriculum_phase_2_delays="5,8",
        curriculum_switch_generation=6,
        curriculum_phase="phase_2",
        active_evaluation_delays="5,8",
        score_current_phase=3.9,
    )


def test_archive_descriptor_binning_is_deterministic_for_mechanism_profile() -> None:
    first = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        slow_fast_contribution_ratio=2.25,
    )
    second = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        slow_fast_contribution_ratio=2.25,
    )

    assert first == second
    assert first.qd_profile == DEFAULT_QD_PROFILE
    assert first.descriptor_key == second.descriptor_key


def test_archive_descriptor_binning_is_deterministic_for_general_compactness() -> None:
    first = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        enabled_conn_count=5,
        qd_profile=QD_PROFILE_GENERAL_COMPACTNESS,
    )
    second = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        enabled_conn_count=5,
        qd_profile=QD_PROFILE_GENERAL_COMPACTNESS,
    )

    assert first == second
    assert first.qd_profile == QD_PROFILE_GENERAL_COMPACTNESS
    assert "connbin_" in first.descriptor_key


def test_archive_descriptor_binning_is_deterministic_for_gating_mechanism() -> None:
    first = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        gate_selectivity=0.6,
        qd_profile=QD_PROFILE_GATING_MECHANISM,
    )
    second = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        gate_selectivity=0.6,
        qd_profile=QD_PROFILE_GATING_MECHANISM,
    )
    assert first == second
    assert first.qd_profile == QD_PROFILE_GATING_MECHANISM
    assert "gatebin_" in first.descriptor_key


def test_archive_descriptor_binning_is_deterministic_for_content_retrieval() -> None:
    first = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        match_selectivity=0.6,
        qd_profile=QD_PROFILE_CONTENT_RETRIEVAL,
    )
    second = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        match_selectivity=0.6,
        qd_profile=QD_PROFILE_CONTENT_RETRIEVAL,
    )
    assert first == second
    assert first.qd_profile == QD_PROFILE_CONTENT_RETRIEVAL
    assert "matchbin_" in first.descriptor_key


def test_archive_descriptor_binning_is_deterministic_for_kv_retrieval_mechanism() -> None:
    first = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        store_vs_distractor_write_gap=0.4,
        qd_profile=QD_PROFILE_KV_RETRIEVAL_MECHANISM,
    )
    second = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        store_vs_distractor_write_gap=0.4,
        qd_profile=QD_PROFILE_KV_RETRIEVAL_MECHANISM,
    )
    assert first == second
    assert first.qd_profile == QD_PROFILE_KV_RETRIEVAL_MECHANISM
    assert "writegapbin_" in first.descriptor_key


def test_archive_descriptor_binning_is_deterministic_for_slot_retrieval_mechanism() -> None:
    first = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        slot_write_focus=0.4,
        slot_query_focus=0.6,
        slot_utilization=1.0,
        qd_profile=QD_PROFILE_SLOT_RETRIEVAL_MECHANISM,
    )
    second = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        slot_write_focus=0.4,
        slot_query_focus=0.6,
        slot_utilization=1.0,
        qd_profile=QD_PROFILE_SLOT_RETRIEVAL_MECHANISM,
    )
    assert first == second
    assert first.qd_profile == QD_PROFILE_SLOT_RETRIEVAL_MECHANISM
    assert "slotfocusbin_" in first.descriptor_key


def test_archive_descriptor_binning_is_deterministic_for_delta_memory_mechanism() -> None:
    first = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        query_memory_alignment=0.4,
        qd_profile=QD_PROFILE_DELTA_MEMORY_MECHANISM,
    )
    second = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        query_memory_alignment=0.4,
        qd_profile=QD_PROFILE_DELTA_MEMORY_MECHANISM,
    )
    assert first == second
    assert first.qd_profile == QD_PROFILE_DELTA_MEMORY_MECHANISM
    assert "qmalignbin_" in first.descriptor_key


def test_general_compactness_builds_for_all_variants() -> None:
    keys = {
        build_archive_cell(
            _feature_record(candidate_id=f"c-{variant}", variant=variant),
            score_ceiling=4.0,
            qd_profile=QD_PROFILE_GENERAL_COMPACTNESS,
        ).descriptor_key
        for variant in ("stateful", "stateful_v2", "stateful_plastic_hebb")
    }

    assert len(keys) == 1
    assert all("connbin_" in key for key in keys)


def test_delay_robustness_descriptor_binning_is_deterministic() -> None:
    first = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        mean_score_over_delays=3.9,
        delay_score_std=0.1,
        qd_profile=QD_PROFILE_DELAY_ROBUSTNESS,
    )
    second = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        mean_score_over_delays=3.9,
        delay_score_std=0.1,
        qd_profile=QD_PROFILE_DELAY_ROBUSTNESS,
    )

    assert first == second
    assert first.qd_profile == QD_PROFILE_DELAY_ROBUSTNESS
    assert "stdbin_" in first.descriptor_key


def test_curriculum_progress_descriptor_includes_phase() -> None:
    descriptor = build_archive_descriptor(
        final_max_score=3.9,
        score_ceiling=4.0,
        mean_score_over_delays=3.9,
        delay_score_std=0.1,
        curriculum_phase="phase_2",
        active_evaluation_delays="5,8",
        qd_profile=QD_PROFILE_CURRICULUM_PROGRESS,
    )

    assert descriptor.qd_profile == QD_PROFILE_CURRICULUM_PROGRESS
    assert descriptor.descriptor_key.startswith("phase_2|scorebin_")


def test_retrieval_strategy_descriptor_is_deterministic() -> None:
    first = build_archive_descriptor(
        final_max_score=10.0,
        score_ceiling=12.0,
        retrieval_score=0.82,
        query_accuracy=0.75,
        relevant_token_retention=0.82,
        distractor_suppression_ratio=2.5,
        slow_query_coupling=1.4,
        qd_profile=QD_PROFILE_RETRIEVAL_STRATEGY,
    )
    second = build_archive_descriptor(
        final_max_score=10.0,
        score_ceiling=12.0,
        retrieval_score=0.82,
        query_accuracy=0.75,
        relevant_token_retention=0.82,
        distractor_suppression_ratio=2.5,
        slow_query_coupling=1.4,
        qd_profile=QD_PROFILE_RETRIEVAL_STRATEGY,
    )

    assert first == second
    assert first.qd_profile == QD_PROFILE_RETRIEVAL_STRATEGY
    assert "suppbin_" in first.descriptor_key


def test_retrieval_mechanism_descriptor_is_deterministic() -> None:
    first = build_archive_descriptor(
        final_max_score=10.0,
        score_ceiling=12.0,
        retrieval_score=0.82,
        query_accuracy=0.75,
        relevant_token_retention=0.82,
        distractor_suppression_ratio=2.5,
        slow_query_coupling=1.4,
        qd_profile=QD_PROFILE_RETRIEVAL_MECHANISM,
    )
    second = build_archive_descriptor(
        final_max_score=10.0,
        score_ceiling=12.0,
        retrieval_score=0.82,
        query_accuracy=0.75,
        relevant_token_retention=0.82,
        distractor_suppression_ratio=2.5,
        slow_query_coupling=1.4,
        qd_profile=QD_PROFILE_RETRIEVAL_MECHANISM,
    )

    assert first == second
    assert first.qd_profile == QD_PROFILE_RETRIEVAL_MECHANISM
    assert "couplingbin_" in first.descriptor_key


def test_in_memory_repository_keeps_profiles_separate() -> None:
    repository = InMemoryRepository(run_id_prefix="archive")
    mechanism = ArchiveCellRecord(
        archive_id="bench|bit_memory|d8|stateful_v2|mechanism_v2|scorebin_7|slowratio_4",
        benchmark_label="bench",
        task_name="bit_memory",
        delay_steps=8,
        variant="stateful_v2",
        descriptor_key="scorebin_7|slowratio_4",
        descriptor_values_json='{"normalized_score":0.95,"score_bin":7,"slow_ratio_bin":4}',
        elite_candidate_id="c-1",
        elite_score=3.8,
        elite_run_id="run-1",
            updated_at="2026-04-03T10:00:00+00:00",
            qd_profile="mechanism_v2",
            descriptor_schema_version="v7a-qdlight-v1",
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
        )
    compactness = ArchiveCellRecord(
        archive_id="bench|bit_memory|d8|stateful_v2|general_compactness|scorebin_7|connbin_2",
        benchmark_label="bench",
        task_name="bit_memory",
        delay_steps=8,
        variant="stateful_v2",
        descriptor_key="scorebin_7|connbin_2",
        descriptor_values_json='{"normalized_score":0.95,"score_bin":7,"conn_bin":2,"enabled_conn_count":4}',
        elite_candidate_id="c-2",
        elite_score=3.8,
        elite_run_id="run-2",
            updated_at="2026-04-03T10:01:00+00:00",
            qd_profile=QD_PROFILE_GENERAL_COMPACTNESS,
            descriptor_schema_version="v7b-qdlight-v1",
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
        )

    repository.consider_archive_candidate(mechanism)
    repository.consider_archive_candidate(compactness)

    stored_mechanism = repository.list_archive_cells(benchmark_label="bench", qd_profile="mechanism_v2")
    stored_compactness = repository.list_archive_cells(
        benchmark_label="bench",
        qd_profile=QD_PROFILE_GENERAL_COMPACTNESS,
    )

    assert len(stored_mechanism) == 1
    assert len(stored_compactness) == 1
    assert stored_mechanism[0].qd_profile == "mechanism_v2"
    assert stored_compactness[0].qd_profile == QD_PROFILE_GENERAL_COMPACTNESS


def test_archive_jsonl_roundtrip_and_cli_report_for_both_profiles(tmp_path: Path, capsys) -> None:
    archive_cells = [
        ArchiveCellRecord(
            archive_id="bench|bit_memory|d5|stateful_v2|mechanism_v2|scorebin_7|slowratio_6",
            benchmark_label="bench-v7",
            task_name="bit_memory",
            delay_steps=5,
            variant="stateful_v2",
            descriptor_key="scorebin_7|slowratio_6",
            descriptor_values_json='{"normalized_score":1.0,"score_bin":7,"slow_fast_contribution_ratio":4.9,"slow_ratio_bin":6}',
            elite_candidate_id="c-1",
            elite_score=4.0,
            elite_run_id="run-1",
            updated_at="2026-04-03T10:00:00+00:00",
            qd_profile="mechanism_v2",
            descriptor_schema_version="v7a-qdlight-v1",
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
        ),
        ArchiveCellRecord(
            archive_id="bench|bit_memory|d5|stateful_v2|general_compactness|scorebin_7|connbin_2",
            benchmark_label="bench-v7",
            task_name="bit_memory",
            delay_steps=5,
            variant="stateful_v2",
            descriptor_key="scorebin_7|connbin_2",
            descriptor_values_json='{"normalized_score":1.0,"score_bin":7,"enabled_conn_count":4,"conn_bin":2}',
            elite_candidate_id="c-1",
            elite_score=4.0,
            elite_run_id="run-1",
            updated_at="2026-04-03T10:00:00+00:00",
            qd_profile=QD_PROFILE_GENERAL_COMPACTNESS,
            descriptor_schema_version="v7b-qdlight-v1",
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
        ),
        ArchiveCellRecord(
            archive_id="bench|bit_memory|d5|stateful_v2|delay_robustness|scorebin_7|stdbin_0",
            benchmark_label="bench-v7",
            task_name="bit_memory",
            delay_steps=5,
            variant="stateful_v2",
            descriptor_key="scorebin_7|stdbin_0",
            descriptor_values_json='{"normalized_score":0.975,"score_bin":7,"mean_score_over_delays":3.9,"delay_score_std":0.1,"delay_std_bin":0}',
            elite_candidate_id="c-1",
            elite_score=3.9,
            elite_run_id="run-1",
            updated_at="2026-04-03T10:00:00+00:00",
            qd_profile=QD_PROFILE_DELAY_ROBUSTNESS,
            descriptor_schema_version="v8a-qdlight-v1",
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
        ),
        ArchiveCellRecord(
            archive_id="bench|bit_memory|d5|stateful_v2|curriculum_progress|phase_2|scorebin_7|stdbin_0",
            benchmark_label="bench-v7",
            task_name="bit_memory",
            delay_steps=5,
            variant="stateful_v2",
            descriptor_key="phase_2|scorebin_7|stdbin_0",
            descriptor_values_json='{"normalized_score":0.975,"score_bin":7,"mean_score_over_delays":3.9,"delay_score_std":0.1,"delay_std_bin":0,"curriculum_phase":"phase_2","active_evaluation_delays":"5,8"}',
            elite_candidate_id="c-1",
            elite_score=3.9,
            elite_run_id="run-1",
            updated_at="2026-04-03T10:00:00+00:00",
            qd_profile=QD_PROFILE_CURRICULUM_PROGRESS,
            descriptor_schema_version="v8b-qdlight-v1",
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
        ),
    ]
    archive_events = [
        ArchiveEventRecord(
            event_id="e-1",
            archive_id=archive_cells[0].archive_id,
            benchmark_label="bench-v7",
            task_name="bit_memory",
            delay_steps=5,
            variant="stateful_v2",
            descriptor_key=archive_cells[0].descriptor_key,
            candidate_id="c-1",
            event_type="insert",
            score=4.0,
            created_at="2026-04-03T10:00:00+00:00",
            qd_profile="mechanism_v2",
            descriptor_schema_version="v7a-qdlight-v1",
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
        ),
        ArchiveEventRecord(
            event_id="e-2",
            archive_id=archive_cells[1].archive_id,
            benchmark_label="bench-v7",
            task_name="bit_memory",
            delay_steps=5,
            variant="stateful_v2",
            descriptor_key=archive_cells[1].descriptor_key,
            candidate_id="c-1",
            event_type="insert",
            score=4.0,
            created_at="2026-04-03T10:00:00+00:00",
            qd_profile=QD_PROFILE_GENERAL_COMPACTNESS,
            descriptor_schema_version="v7b-qdlight-v1",
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
        ),
        ArchiveEventRecord(
            event_id="e-3",
            archive_id=archive_cells[2].archive_id,
            benchmark_label="bench-v7",
            task_name="bit_memory",
            delay_steps=5,
            variant="stateful_v2",
            descriptor_key=archive_cells[2].descriptor_key,
            candidate_id="c-1",
            event_type="insert",
            score=3.9,
            created_at="2026-04-03T10:00:00+00:00",
            qd_profile=QD_PROFILE_DELAY_ROBUSTNESS,
            descriptor_schema_version="v8a-qdlight-v1",
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
        ),
        ArchiveEventRecord(
            event_id="e-4",
            archive_id=archive_cells[3].archive_id,
            benchmark_label="bench-v7",
            task_name="bit_memory",
            delay_steps=5,
            variant="stateful_v2",
            descriptor_key=archive_cells[3].descriptor_key,
            candidate_id="c-1",
            event_type="insert",
            score=3.9,
            created_at="2026-04-03T10:00:00+00:00",
            qd_profile=QD_PROFILE_CURRICULUM_PROGRESS,
            descriptor_schema_version="v8b-qdlight-v1",
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
        ),
    ]
    feature_records = [_feature_record(candidate_id="c-1")]

    archive_cell_path = tmp_path / "bench-v7.archive-cells.jsonl"
    archive_event_path = tmp_path / "bench-v7.archive-events.jsonl"
    feature_path = tmp_path / "bench-v7.candidate-features.jsonl"
    write_archive_cells_jsonl(archive_cell_path, archive_cells)
    write_archive_events_jsonl(archive_event_path, archive_events)
    from analysis.search_space import write_feature_records_jsonl

    write_feature_records_jsonl(feature_path, feature_records)

    assert load_archive_cells_from_jsonl(archive_cell_path) == archive_cells
    assert load_archive_events_from_jsonl(archive_event_path) == archive_events

    exit_code = main(
        [
            "analyze-archive",
            "--store",
            "memory",
            "--benchmark-label",
            "bench-v7",
            "--task",
            "bit_memory",
            "--variant",
            "stateful_v2",
            "--qd-profile",
            "mechanism_v2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "## Archive Summary (mechanism_v2)" in output
    assert "## Curriculum Metadata" in output
    assert "slow_fast_ratio" in output

    exit_code = main(
        [
            "analyze-archive",
            "--store",
            "memory",
            "--benchmark-label",
            "bench-v7",
            "--task",
            "bit_memory",
            "--variant",
            "stateful_v2",
            "--qd-profile",
            "general_compactness",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "## Archive Summary (general_compactness)" in output
    assert "enabled_conn_count" in output

    exit_code = main(
        [
            "analyze-archive",
            "--store",
            "memory",
            "--benchmark-label",
            "bench-v7",
            "--task",
            "bit_memory",
            "--variant",
            "stateful_v2",
            "--qd-profile",
            "delay_robustness",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "## Archive Summary (delay_robustness)" in output
    assert "delay_score_std" in output

    exit_code = main(
        [
            "analyze-archive",
            "--store",
            "memory",
            "--benchmark-label",
            "bench-v7",
            "--task",
            "bit_memory",
            "--variant",
            "stateful_v2",
            "--qd-profile",
            "curriculum_progress",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "## Archive Summary (curriculum_progress)" in output
    assert "switch_generation" in output
    assert "active_evaluation_delays" in output
    assert "## By Phase" in output


def test_analyze_archive_supports_curriculum_phase_filter(tmp_path: Path, capsys) -> None:
    phase_1_cell = ArchiveCellRecord(
        archive_id="bench|bit_memory|d5|stateful_v2|delay_robustness|scorebin_6|stdbin_0",
        benchmark_label="bench-v8d",
        task_name="bit_memory",
        delay_steps=5,
        variant="stateful_v2",
        descriptor_key="scorebin_6|stdbin_0",
        descriptor_values_json='{"normalized_score":0.80,"score_bin":6,"mean_score_over_delays":3.2,"delay_score_std":0.0,"delay_std_bin":0,"curriculum_phase":"phase_1","active_evaluation_delays":"5"}',
        elite_candidate_id="c-phase-1",
        elite_score=3.2,
        elite_run_id="run-1",
        updated_at="2026-04-04T10:00:00+00:00",
        qd_profile=QD_PROFILE_DELAY_ROBUSTNESS,
        descriptor_schema_version="v8a-qdlight-v1",
        curriculum_enabled=True,
        curriculum_phase_1_delays="5",
        curriculum_phase_2_delays="5,8",
        curriculum_switch_generation=6,
    )
    phase_2_cell = ArchiveCellRecord(
        archive_id="bench|bit_memory|d5|stateful_v2|delay_robustness|scorebin_7|stdbin_0",
        benchmark_label="bench-v8d",
        task_name="bit_memory",
        delay_steps=5,
        variant="stateful_v2",
        descriptor_key="scorebin_7|stdbin_0",
        descriptor_values_json='{"normalized_score":1.0,"score_bin":7,"mean_score_over_delays":4.0,"delay_score_std":0.0,"delay_std_bin":0,"curriculum_phase":"phase_2","active_evaluation_delays":"5,8"}',
        elite_candidate_id="c-phase-2",
        elite_score=4.0,
        elite_run_id="run-1",
        updated_at="2026-04-04T10:01:00+00:00",
        qd_profile=QD_PROFILE_DELAY_ROBUSTNESS,
        descriptor_schema_version="v8a-qdlight-v1",
        curriculum_enabled=True,
        curriculum_phase_1_delays="5",
        curriculum_phase_2_delays="5,8",
        curriculum_switch_generation=6,
    )
    archive_cell_path = tmp_path / "bench-v8d.archive-cells.jsonl"
    archive_event_path = tmp_path / "bench-v8d.archive-events.jsonl"
    feature_path = tmp_path / "bench-v8d.candidate-features.jsonl"
    write_archive_cells_jsonl(archive_cell_path, [phase_1_cell, phase_2_cell])
    write_archive_events_jsonl(archive_event_path, [])
    from analysis.search_space import write_feature_records_jsonl

    write_feature_records_jsonl(
        feature_path,
        [
            _feature_record(candidate_id="c-phase-1"),
            _feature_record(candidate_id="c-phase-2"),
        ],
    )

    exit_code = main(
        [
            "analyze-archive",
            "--store",
            "memory",
            "--benchmark-label",
            "bench-v8d",
            "--task",
            "bit_memory",
            "--variant",
            "stateful_v2",
            "--qd-profile",
            "delay_robustness",
            "--curriculum-phase",
            "phase_2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "## Archive Summary (delay_robustness, phase_2)" in output
    assert "c-phase-2" in output
    assert "c-phase-1" not in output


def test_analyze_archive_supports_retrieval_strategy_profile(tmp_path: Path, capsys) -> None:
    archive_cell = ArchiveCellRecord(
        archive_id="bench|key_value_memory|d8|stateful_v2|retrieval_strategy|scorebin_6|suppbin_4",
        benchmark_label="bench-v9a",
        task_name="key_value_memory",
        delay_steps=8,
        variant="stateful_v2",
        descriptor_key="scorebin_6|suppbin_4",
        descriptor_values_json='{"normalized_score":0.82,"score_bin":6,"retrieval_score":0.82,"query_accuracy":0.75,"distractor_suppression_ratio":2.5,"suppression_bin":4,"slow_query_coupling":1.4}',
        elite_candidate_id="c-retrieval",
        elite_score=10.0,
        elite_run_id="run-1",
        updated_at="2026-04-05T10:00:00+00:00",
        qd_profile=QD_PROFILE_RETRIEVAL_STRATEGY,
        descriptor_schema_version="v9a-retrieval-v1",
    )
    feature_record = CandidateFeatureRecord(
        **{
            **_feature_record(candidate_id="c-retrieval").__dict__,
            "task_name": "key_value_memory",
            "query_accuracy": 0.75,
            "retrieval_score": 0.82,
            "distractor_suppression_ratio": 2.5,
            "slow_query_coupling": 1.4,
        }
    )
    archive_cell_path = tmp_path / "bench-v9a.archive-cells.jsonl"
    archive_event_path = tmp_path / "bench-v9a.archive-events.jsonl"
    feature_path = tmp_path / "bench-v9a.candidate-features.jsonl"
    write_archive_cells_jsonl(archive_cell_path, [archive_cell])
    write_archive_events_jsonl(archive_event_path, [])
    from analysis.search_space import write_feature_records_jsonl

    write_feature_records_jsonl(feature_path, [feature_record])

    exit_code = main(
        [
            "analyze-archive",
            "--store",
            "memory",
            "--benchmark-label",
            "bench-v9a",
            "--task",
            "key_value_memory",
            "--variant",
            "stateful_v2",
            "--qd-profile",
            "retrieval_strategy",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "## Archive Summary (retrieval_strategy)" in output
    assert "distractor_suppression_ratio" in output
    assert "slow_query_coupling" in output


def test_analyze_archive_supports_retrieval_mechanism_profile(tmp_path: Path, capsys) -> None:
    archive_cell = ArchiveCellRecord(
        archive_id="bench|key_value_memory|d8|stateful_v2|retrieval_mechanism|scorebin_6|couplingbin_3",
        benchmark_label="bench-v9b",
        task_name="key_value_memory",
        delay_steps=8,
        variant="stateful_v2",
        descriptor_key="scorebin_6|couplingbin_3",
        descriptor_values_json='{"normalized_score":0.82,"score_bin":6,"retrieval_score":0.82,"query_accuracy":0.75,"distractor_suppression_ratio":2.5,"slow_query_coupling":1.4,"coupling_bin":3}',
        elite_candidate_id="c-mechanism",
        elite_score=10.0,
        elite_run_id="run-1",
        updated_at="2026-04-05T10:00:00+00:00",
        qd_profile=QD_PROFILE_RETRIEVAL_MECHANISM,
        descriptor_schema_version="v9b-retrieval-v1",
    )
    feature_record = CandidateFeatureRecord(
        **{
            **_feature_record(candidate_id="c-mechanism").__dict__,
            "task_name": "key_value_memory",
            "query_accuracy": 0.75,
            "retrieval_score": 0.82,
            "distractor_suppression_ratio": 2.5,
            "slow_query_coupling": 1.4,
            "query_response_margin": 0.33,
        }
    )
    archive_cell_path = tmp_path / "bench-v9b.archive-cells.jsonl"
    archive_event_path = tmp_path / "bench-v9b.archive-events.jsonl"
    feature_path = tmp_path / "bench-v9b.candidate-features.jsonl"
    write_archive_cells_jsonl(archive_cell_path, [archive_cell])
    write_archive_events_jsonl(archive_event_path, [])
    from analysis.search_space import write_feature_records_jsonl

    write_feature_records_jsonl(feature_path, [feature_record])

    exit_code = main(
        [
            "analyze-archive",
            "--store",
            "memory",
            "--benchmark-label",
            "bench-v9b",
            "--task",
            "key_value_memory",
            "--variant",
            "stateful_v2",
            "--qd-profile",
            "retrieval_mechanism",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "## Archive Summary (retrieval_mechanism)" in output
    assert "slow_query_coupling" in output
    assert "query_response_margin" in output


def test_build_archive_cell_uses_feature_context_for_requested_profile() -> None:
    feature = _feature_record(candidate_id="c-1")

    mechanism_cell = build_archive_cell(feature, score_ceiling=4.0)
    compactness_cell = build_archive_cell(
        feature,
        score_ceiling=4.0,
        qd_profile=QD_PROFILE_GENERAL_COMPACTNESS,
    )
    retrieval_mechanism_cell = build_archive_cell(
        CandidateFeatureRecord(
            **{
                **feature.__dict__,
                "task_name": "key_value_memory",
                "retrieval_score": 0.82,
                "query_accuracy": 0.75,
                "distractor_suppression_ratio": 2.5,
                "slow_query_coupling": 1.4,
            }
        ),
        score_ceiling=12.0,
        qd_profile=QD_PROFILE_RETRIEVAL_MECHANISM,
    )

    assert mechanism_cell.qd_profile == "mechanism_v2"
    assert compactness_cell.qd_profile == QD_PROFILE_GENERAL_COMPACTNESS
    assert retrieval_mechanism_cell.qd_profile == QD_PROFILE_RETRIEVAL_MECHANISM
    assert "slowratio_" in mechanism_cell.descriptor_key
    assert "connbin_" in compactness_cell.descriptor_key
    assert "couplingbin_" in retrieval_mechanism_cell.descriptor_key
    assert compactness_cell.elite_candidate_id == "c-1"
