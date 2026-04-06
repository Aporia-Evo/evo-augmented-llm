from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from analysis.search_space import load_feature_records_from_jsonl, write_feature_records_jsonl
from db.models import CandidateFeatureRecord
from db.reducers import InMemoryRepository
from evolve.candidate_features import CandidateFeatureContext, extract_candidate_features
from evolve.genome_codec import ConnectionGeneModel, GenomeModel, NodeGeneModel
from ui.cli import main


def test_extract_candidate_features_is_deterministic() -> None:
    genome = _plastic_genome()
    context = _context()
    raw_metrics = {
        "success": True,
        "mean_abs_delta_w": 0.12,
        "max_abs_delta_w": 0.4,
        "clamp_hit_rate": 0.05,
        "plasticity_active_fraction": 0.75,
    }

    first_feature, first_vector = extract_candidate_features(genome, raw_metrics, context)
    second_feature, second_vector = extract_candidate_features(genome, raw_metrics, context)

    assert first_feature == second_feature
    assert first_vector == second_vector


def test_extract_candidate_features_handles_missing_metrics_for_old_variants() -> None:
    genome = _stateful_genome()
    feature, vector = extract_candidate_features(genome, {}, _context(variant="stateful"))

    assert feature.mean_eta == 0.0
    assert feature.mean_plastic_d == 0.0
    assert feature.mean_abs_delta_w == 0.0
    assert feature.max_abs_delta_w == 0.0
    assert feature.clamp_hit_rate == 0.0
    assert feature.plasticity_active_fraction == 0.0
    assert feature.mean_abs_fast_state == 0.0
    assert feature.mean_abs_slow_state == 0.0
    assert feature.slow_fast_contribution_ratio == 0.0
    assert feature.mean_abs_decay_term == 0.0
    assert feature.max_abs_decay_term == 0.0
    assert feature.decay_effect_ratio == 0.0
    assert feature.decay_near_zero_fraction == 0.0
    assert feature.score_delay_3 == 0.0
    assert feature.score_delay_5 == 0.0
    assert feature.score_delay_8 == 0.0
    assert feature.mean_score_over_delays == 3.9
    assert feature.delay_score_std == 0.0
    assert feature.delay_score_range == 0.0
    assert feature.curriculum_enabled is False
    assert feature.curriculum_phase_1_delays == ""
    assert feature.curriculum_phase_2_delays == ""
    assert feature.curriculum_switch_generation == 0
    assert feature.curriculum_phase == "static"
    assert feature.active_evaluation_delays == ""
    assert feature.score_current_phase == 3.9
    assert feature.query_accuracy == 0.0
    assert feature.retrieval_score == 0.0
    assert feature.mean_query_distance == 0.0
    assert feature.distractor_load == 0.0
    assert feature.correct_key_selected == 0.0
    assert feature.correct_value_selected == 0.0
    assert feature.query_key_match_score == 0.0
    assert feature.value_margin == 0.0
    assert feature.distractor_competition_score == 0.0
    assert feature.distractor_suppression_ratio == 0.0
    assert feature.mean_abs_fast_state_during_store == 0.0
    assert feature.mean_abs_slow_state_during_query == 0.0
    assert feature.slow_query_coupling == 0.0
    assert feature.store_query_state_gap == 0.0
    assert feature.slow_fast_retrieval_ratio == 0.0
    assert feature.retrieval_state_alignment == 0.0
    assert vector.norm_l2 >= 0.0


def test_extract_candidate_features_captures_multi_delay_metrics() -> None:
    genome = _plastic_genome()
    context = _context(variant="stateful_v2")
    raw_metrics = {
        "success": True,
        "score_delay_5": 3.8,
        "score_delay_8": 4.0,
        "success_delay_5": False,
        "success_delay_8": True,
        "mean_score_over_delays": 3.9,
        "delay_score_std": 0.1,
        "delay_score_range": 0.2,
        "curriculum_enabled": True,
        "curriculum_phase_1_delays": "5",
        "curriculum_phase_2_delays": "5,8",
        "curriculum_switch_generation": 6,
        "curriculum_phase": "phase_2",
        "active_evaluation_delays": "5,8",
        "score_current_phase": 3.9,
    }

    feature, _vector = extract_candidate_features(genome, raw_metrics, context)

    assert feature.score_delay_5 == 3.8
    assert feature.score_delay_8 == 4.0
    assert feature.success_delay_5 is False
    assert feature.success_delay_8 is True
    assert feature.mean_score_over_delays == 3.9
    assert feature.delay_score_std == 0.1
    assert feature.delay_score_range == 0.2
    assert feature.curriculum_enabled is True
    assert feature.curriculum_phase_1_delays == "5"
    assert feature.curriculum_phase_2_delays == "5,8"
    assert feature.curriculum_switch_generation == 6
    assert feature.curriculum_phase == "phase_2"
    assert feature.active_evaluation_delays == "5,8"
    assert feature.score_current_phase == 3.9


def test_extract_candidate_features_captures_retrieval_metrics() -> None:
    genome = _plastic_genome()
    context = _context(variant="stateful_v2")
    raw_metrics = {
        "success": True,
        "query_accuracy": 0.75,
        "retrieval_score": 0.82,
        "exact_match_success": True,
        "mean_query_distance": 6.0,
        "distractor_load": 4.0,
        "num_stores": 2,
        "num_queries": 1,
        "num_distractors": 4,
        "correct_key_selected": 0.9,
        "correct_value_selected": 0.7,
        "query_key_match_score": 0.6,
        "value_margin": 0.35,
        "distractor_competition_score": 0.18,
        "retrieval_margin": 0.31,
        "retrieval_confusion_rate": 0.25,
        "relevant_token_retention": 0.82,
        "query_response_margin": 0.44,
        "distractor_suppression_ratio": 2.5,
        "mean_abs_fast_state_during_store": 0.7,
        "mean_abs_slow_state_during_store": 1.4,
        "mean_abs_fast_state_during_query": 0.5,
        "mean_abs_slow_state_during_query": 1.5,
        "mean_abs_fast_state_during_distractor": 0.2,
        "mean_abs_slow_state_during_distractor": 0.3,
    }

    feature, _vector = extract_candidate_features(genome, raw_metrics, context)

    assert feature.query_accuracy == 0.75
    assert feature.retrieval_score == 0.82
    assert feature.exact_match_success is True
    assert feature.mean_query_distance == 6.0
    assert feature.distractor_load == 4.0
    assert feature.correct_key_selected == 0.9
    assert feature.correct_value_selected == 0.7
    assert feature.query_key_match_score == 0.6
    assert feature.value_margin == 0.35
    assert feature.distractor_competition_score == 0.18
    assert feature.retrieval_margin == 0.31
    assert feature.query_response_margin == 0.44
    assert feature.distractor_suppression_ratio == 2.5
    assert feature.mean_abs_fast_state_during_store == 0.7
    assert feature.mean_abs_slow_state_during_query == 1.5
    assert feature.slow_query_coupling > 1.0
    assert abs(feature.store_query_state_gap - 0.1) <= 1e-9
    assert feature.slow_fast_retrieval_ratio == 3.0
    assert feature.retrieval_state_alignment > 0.9


def test_in_memory_repository_stores_feature_values_and_hof_flag() -> None:
    repository = InMemoryRepository(run_id_prefix="features")
    feature, vector = extract_candidate_features(
        _plastic_genome(),
        {"success": True, "mean_abs_delta_w": 0.25, "max_abs_delta_w": 0.5},
        _context(),
    )

    repository.upsert_candidate_features(feature)
    repository.upsert_candidate_feature_vector(vector)
    updated = repository.mark_hof_candidate(feature.candidate_id, hof_flag=True)

    assert updated is not None
    assert updated.hof_flag is True
    assert repository.list_candidate_features(benchmark_label="bench-v5b15")[0].mean_eta == feature.mean_eta
    assert repository.list_candidate_feature_vectors(candidate_ids=[feature.candidate_id])[0].feature_version == vector.feature_version


def test_feature_jsonl_roundtrip_and_analysis_cli_output(tmp_path: Path, capsys) -> None:
    records = [
        CandidateFeatureRecord(
            candidate_id="c-1",
            run_id="run-1",
            benchmark_label="bench-v5b15",
            task_name="bit_memory",
            delay_steps=5,
            variant="stateful_plastic_ad",
            seed=7,
            generation=4,
            hof_flag=True,
            success=True,
            final_max_score=3.9,
            first_success_generation=4,
            mean_alpha=0.4,
            std_alpha=0.1,
            mean_eta=0.02,
            std_eta=0.01,
            mean_plastic_d=-0.05,
            std_plastic_d=0.02,
            plastic_d_at_lower_bound_fraction=0.0,
            plastic_d_at_zero_fraction=0.25,
            node_count=5,
            enabled_conn_count=4,
            mean_abs_delta_w=0.11,
            max_abs_delta_w=0.32,
            clamp_hit_rate=0.02,
            plasticity_active_fraction=0.5,
            mean_abs_fast_state=0.21,
            mean_abs_slow_state=0.08,
            slow_fast_contribution_ratio=0.18,
            mean_abs_decay_term=0.01,
            max_abs_decay_term=0.03,
            decay_effect_ratio=0.09,
            decay_near_zero_fraction=0.7,
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
            curriculum_phase="phase_2",
            active_evaluation_delays="5,8",
            score_current_phase=3.9,
        ),
        CandidateFeatureRecord(
            candidate_id="c-2",
            run_id="run-2",
            benchmark_label="bench-v5b15",
            task_name="bit_memory",
            delay_steps=5,
            variant="stateful_plastic_ad_d01",
            seed=11,
            generation=6,
            hof_flag=False,
            success=False,
            final_max_score=2.8,
            first_success_generation=None,
            mean_alpha=0.3,
            std_alpha=0.15,
            mean_eta=0.08,
            std_eta=0.03,
            mean_plastic_d=-0.2,
            std_plastic_d=0.04,
            plastic_d_at_lower_bound_fraction=0.5,
            plastic_d_at_zero_fraction=0.0,
            node_count=6,
            enabled_conn_count=5,
            mean_abs_delta_w=0.35,
            max_abs_delta_w=0.5,
            clamp_hit_rate=0.4,
            plasticity_active_fraction=0.9,
            mean_abs_fast_state=0.31,
            mean_abs_slow_state=0.12,
            slow_fast_contribution_ratio=0.28,
            mean_abs_decay_term=0.08,
            max_abs_decay_term=0.16,
            decay_effect_ratio=0.42,
            decay_near_zero_fraction=0.1,
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
            curriculum_phase="phase_2",
            active_evaluation_delays="5,8",
            score_current_phase=2.8,
        ),
        CandidateFeatureRecord(
            candidate_id="c-3",
            run_id="run-3",
            benchmark_label="bench-v5b15",
            task_name="bit_memory",
            delay_steps=8,
            variant="stateful_plastic_ad_d005",
            seed=13,
            generation=5,
            hof_flag=True,
            success=True,
            final_max_score=3.95,
            first_success_generation=5,
            mean_alpha=0.45,
            std_alpha=0.08,
            mean_eta=0.03,
            std_eta=0.01,
            mean_plastic_d=-0.05,
            std_plastic_d=0.0,
            plastic_d_at_lower_bound_fraction=0.0,
            plastic_d_at_zero_fraction=0.0,
            node_count=5,
            enabled_conn_count=4,
            mean_abs_delta_w=0.2,
            max_abs_delta_w=0.4,
            clamp_hit_rate=0.01,
            plasticity_active_fraction=0.8,
            mean_abs_fast_state=0.26,
            mean_abs_slow_state=0.11,
            slow_fast_contribution_ratio=0.24,
            mean_abs_decay_term=0.04,
            max_abs_decay_term=0.08,
            decay_effect_ratio=0.2,
            decay_near_zero_fraction=0.2,
            curriculum_enabled=True,
            curriculum_phase_1_delays="5",
            curriculum_phase_2_delays="5,8",
            curriculum_switch_generation=6,
            curriculum_phase="phase_2",
            active_evaluation_delays="5,8",
            score_current_phase=3.95,
        ),
    ]
    feature_path = tmp_path / "bench-v5b15.candidate-features.jsonl"
    write_feature_records_jsonl(feature_path, records)

    loaded = load_feature_records_from_jsonl(feature_path)
    assert loaded == records

    exit_code = main(
        [
            "analyze-search-space",
            "--store",
            "memory",
            "--benchmark-label",
            "bench-v5b15",
            "--task",
            "bit_memory",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "## Search Space Summary" in output
    assert "## By Variant" in output
    assert "## By Delay" in output
    assert "## HOF vs Non-HOF" in output
    assert "## Success vs Failure" in output
    assert "clamp_hit_rate" in output
    assert "mean_abs_fast_state" in output
    assert "slow_fast_contribution_ratio" in output
    assert "mean_abs_decay_term" in output
    assert "decay_effect_ratio" in output
    assert "mean_score_over_delays" in output
    assert "delay_score_std" in output
    assert "## Curriculum Metadata" in output


def test_analyze_search_space_supports_curriculum_phase_filter(tmp_path: Path, capsys) -> None:
    phase_1 = CandidateFeatureRecord(
        candidate_id="c-phase-1",
        run_id="run-1",
        benchmark_label="bench-v8d",
        task_name="bit_memory",
        delay_steps=8,
        variant="stateful_v2",
        seed=7,
        generation=3,
        hof_flag=False,
        success=False,
        final_max_score=2.2,
        first_success_generation=None,
        mean_alpha=0.4,
        std_alpha=0.1,
        mean_eta=0.0,
        std_eta=0.0,
        mean_plastic_d=0.0,
        std_plastic_d=0.0,
        plastic_d_at_lower_bound_fraction=0.0,
        plastic_d_at_zero_fraction=1.0,
        node_count=4,
        enabled_conn_count=3,
        mean_abs_delta_w=0.0,
        max_abs_delta_w=0.0,
        clamp_hit_rate=0.0,
        plasticity_active_fraction=0.0,
        mean_abs_fast_state=0.8,
        mean_abs_slow_state=0.5,
        slow_fast_contribution_ratio=0.6,
        mean_abs_decay_term=0.0,
        max_abs_decay_term=0.0,
        decay_effect_ratio=0.0,
        decay_near_zero_fraction=0.0,
        score_delay_3=0.0,
        score_delay_5=2.2,
        score_delay_8=0.0,
        success_delay_3=False,
        success_delay_5=False,
        success_delay_8=False,
        mean_score_over_delays=2.2,
        delay_score_std=0.0,
        delay_score_range=0.0,
        curriculum_enabled=True,
        curriculum_phase_1_delays="5",
        curriculum_phase_2_delays="5,8",
        curriculum_switch_generation=6,
        curriculum_phase="phase_1",
        active_evaluation_delays="5",
        score_current_phase=2.2,
    )
    phase_2 = replace(
        phase_1,
        candidate_id="c-phase-2",
        generation=8,
        hof_flag=True,
        success=True,
        final_max_score=3.95,
        first_success_generation=8,
        enabled_conn_count=4,
        mean_abs_fast_state=1.2,
        mean_abs_slow_state=2.4,
        slow_fast_contribution_ratio=2.0,
        score_delay_5=3.9,
        score_delay_8=4.0,
        success_delay_5=True,
        success_delay_8=True,
        mean_score_over_delays=3.95,
        delay_score_std=0.05,
        delay_score_range=0.1,
        curriculum_phase="phase_2",
        active_evaluation_delays="5,8",
        score_current_phase=3.95,
    )

    feature_path = tmp_path / "bench-v8d.candidate-features.jsonl"
    write_feature_records_jsonl(feature_path, [phase_1, phase_2])

    exit_code = main(
        [
            "analyze-search-space",
            "--store",
            "memory",
            "--benchmark-label",
            "bench-v8d",
            "--task",
            "bit_memory",
            "--curriculum-phase",
            "phase_2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "## Search Space Summary (phase_2)" in output
    assert "## By Curriculum Phase" not in output
    assert "3.950000" in output


def _context(*, variant: str = "stateful_plastic_ad") -> CandidateFeatureContext:
    return CandidateFeatureContext(
        candidate_id="run-1-g0004-c0001",
        run_id="run-1",
        benchmark_label="bench-v5b15",
        task_name="bit_memory",
        delay_steps=5,
        variant=variant,
        seed=7,
        generation=4,
        final_max_score=3.9,
        first_success_generation=4,
        eta_lower_bound=-0.1,
        eta_upper_bound=0.1,
        plastic_d_lower_bound=-1.0,
    )


def _stateful_genome() -> GenomeModel:
    return GenomeModel(
        input_ids=(0,),
        output_ids=(2,),
        nodes=(
            NodeGeneModel(node_id=0, bias=0.0, alpha=0.0, is_input=True, is_output=False),
            NodeGeneModel(node_id=1, bias=0.1, alpha=0.4, is_input=False, is_output=False),
            NodeGeneModel(node_id=2, bias=-0.2, alpha=0.3, is_input=False, is_output=True),
        ),
        connections=(
            ConnectionGeneModel(in_id=0, out_id=1, historical_marker=1, weight=1.0, enabled=True),
            ConnectionGeneModel(in_id=1, out_id=2, historical_marker=2, weight=0.5, enabled=True),
        ),
    )


def _plastic_genome() -> GenomeModel:
    return GenomeModel(
        input_ids=(0,),
        output_ids=(2,),
        nodes=_stateful_genome().nodes,
        connections=(
            ConnectionGeneModel(
                in_id=0,
                out_id=1,
                historical_marker=1,
                weight=1.0,
                enabled=True,
                eta=0.03,
                plastic_d=-0.1,
            ),
            ConnectionGeneModel(
                in_id=1,
                out_id=2,
                historical_marker=2,
                weight=0.5,
                enabled=True,
                eta=0.01,
                plastic_d=0.0,
            ),
        ),
    )
