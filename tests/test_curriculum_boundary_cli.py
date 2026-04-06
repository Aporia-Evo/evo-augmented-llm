from __future__ import annotations

import json
from pathlib import Path

from analysis.search_space import write_feature_records_jsonl
from db.models import CandidateFeatureRecord
from ui.cli import main


def test_analyze_curriculum_boundaries_reports_post_switch_winner_and_default_policy(
    tmp_path: Path,
    capsys,
) -> None:
    _write_suite_rows(
        tmp_path / "boundary-4.jsonl",
        switch_generation=4,
        success_values={"stateful_v2": [False, True], "stateful": [False, False], "stateful_plastic_hebb": [True, False]},
        final_scores={"stateful_v2": [3.4, 3.8], "stateful": [2.8, 2.9], "stateful_plastic_hebb": [3.2, 3.3]},
    )
    _write_suite_rows(
        tmp_path / "boundary-6.jsonl",
        switch_generation=6,
        success_values={"stateful_v2": [True, True], "stateful": [False, False], "stateful_plastic_hebb": [True, False]},
        final_scores={"stateful_v2": [3.9, 4.0], "stateful": [3.0, 3.1], "stateful_plastic_hebb": [3.4, 3.5]},
    )
    _write_suite_rows(
        tmp_path / "boundary-8.jsonl",
        switch_generation=8,
        success_values={"stateful_v2": [True, True], "stateful": [False, False], "stateful_plastic_hebb": [True, False]},
        final_scores={"stateful_v2": [3.95, 4.0], "stateful": [2.7, 2.8], "stateful_plastic_hebb": [3.3, 3.2]},
    )
    _write_feature_export(
        tmp_path / "boundary-4.candidate-features.jsonl",
        benchmark_label="boundary-4",
        switch_generation=4,
        v2_mean_score=3.5,
        v2_delay_8=2.9,
        v2_delay_std=0.18,
    )
    _write_feature_export(
        tmp_path / "boundary-6.candidate-features.jsonl",
        benchmark_label="boundary-6",
        switch_generation=6,
        v2_mean_score=3.95,
        v2_delay_8=3.9,
        v2_delay_std=0.05,
    )
    _write_feature_export(
        tmp_path / "boundary-8.candidate-features.jsonl",
        benchmark_label="boundary-8",
        switch_generation=8,
        v2_mean_score=4.0,
        v2_delay_8=4.0,
        v2_delay_std=0.02,
    )

    exit_code = main(
        [
            "analyze-curriculum-boundaries",
            "--benchmark-labels",
            "boundary-4,boundary-6,boundary-8",
            "--task",
            "bit_memory",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "## Overall Summary" in output
    assert "## Post-Switch Summary" in output
    assert "## Focus Variant: stateful_v2 (Post-Switch)" in output
    assert "## Focus Duel: stateful_v2 (g6 vs g8)" in output
    assert "## Phase Dynamics: stateful_v2" in output
    assert "`stateful_v2` gewinnt in der harten Phase aktuell bei `switch_generation=8`." in output
    assert "`default_boundary=6`" in output
    assert "`performance_boundary=8`" in output


def _write_suite_rows(
    path: Path,
    *,
    switch_generation: int,
    success_values: dict[str, list[bool]],
    final_scores: dict[str, list[float]],
) -> None:
    rows: list[dict[str, object]] = []
    for variant, successes in success_values.items():
        for seed_index, success in enumerate(successes, start=1):
            rows.append(
                {
                    "task_name": "bit_memory",
                    "delay_steps": 8,
                    "variant": variant,
                    "seed": seed_index,
                    "run_id": f"{variant}-run-{seed_index}",
                    "status": "finished",
                    "generation_budget": 12,
                    "population_size": 40,
                    "score_ceiling": 4.0,
                    "success": success,
                    "first_success_generation": 5 if success else None,
                    "best_generation_id": 5,
                    "best_candidate_id": f"{variant}-c-{seed_index}",
                    "final_max_score": final_scores[variant][seed_index - 1],
                    "final_avg_score": final_scores[variant][seed_index - 1] - 0.1,
                    "completed_generations": 12,
                    "best_node_count": 4,
                    "best_enabled_connection_count": 4,
                    "success_node_count": 4 if success else None,
                    "success_enabled_connection_count": 4 if success else None,
                    "avg_score_path": "1.0->2.0->3.0",
                    "resume_hint": "not_resumed",
                    "evaluation_delay_steps_label": f"5->5,8@g{switch_generation}",
                    "curriculum_enabled": True,
                    "curriculum_phase_1_delays": "5",
                    "curriculum_phase_2_delays": "5,8",
                    "curriculum_switch_generation": switch_generation,
                }
            )
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_feature_export(
    path: Path,
    *,
    benchmark_label: str,
    switch_generation: int,
    v2_mean_score: float,
    v2_delay_8: float,
    v2_delay_std: float,
) -> None:
    records = [
        _feature_record(
            benchmark_label=benchmark_label,
            candidate_id="stateful-v2-phase-1",
            variant="stateful_v2",
            switch_generation=switch_generation,
            curriculum_phase="phase_1",
            mean_score_over_delays=v2_mean_score - 0.3,
            score_delay_8=0.0,
            delay_score_std=0.0,
        ),
        _feature_record(
            benchmark_label=benchmark_label,
            candidate_id="stateful-v2-phase-2",
            variant="stateful_v2",
            switch_generation=switch_generation,
            curriculum_phase="phase_2",
            mean_score_over_delays=v2_mean_score,
            score_delay_8=v2_delay_8,
            delay_score_std=v2_delay_std,
            success=True,
        ),
        _feature_record(
            benchmark_label=benchmark_label,
            candidate_id="stateful-phase-2",
            variant="stateful",
            switch_generation=switch_generation,
            curriculum_phase="phase_2",
            mean_score_over_delays=2.8,
            score_delay_8=1.8,
            delay_score_std=0.1,
        ),
        _feature_record(
            benchmark_label=benchmark_label,
            candidate_id="hebb-phase-2",
            variant="stateful_plastic_hebb",
            switch_generation=switch_generation,
            curriculum_phase="phase_2",
            mean_score_over_delays=3.1,
            score_delay_8=2.6,
            delay_score_std=0.08,
        ),
    ]
    write_feature_records_jsonl(path, records)


def _feature_record(
    *,
    benchmark_label: str,
    candidate_id: str,
    variant: str,
    switch_generation: int,
    curriculum_phase: str,
    mean_score_over_delays: float,
    score_delay_8: float,
    delay_score_std: float,
    success: bool = False,
) -> CandidateFeatureRecord:
    return CandidateFeatureRecord(
        candidate_id=candidate_id,
        run_id="run-1",
        benchmark_label=benchmark_label,
        task_name="bit_memory",
        delay_steps=8,
        variant=variant,
        seed=7,
        generation=5,
        hof_flag=False,
        success=success,
        final_max_score=mean_score_over_delays,
        first_success_generation=5 if success else None,
        mean_alpha=0.4,
        std_alpha=0.1,
        mean_eta=0.0,
        std_eta=0.0,
        mean_plastic_d=0.0,
        std_plastic_d=0.0,
        plastic_d_at_lower_bound_fraction=0.0,
        plastic_d_at_zero_fraction=1.0,
        node_count=4,
        enabled_conn_count=4,
        mean_abs_delta_w=0.0,
        max_abs_delta_w=0.0,
        clamp_hit_rate=0.0,
        plasticity_active_fraction=0.0,
        mean_abs_fast_state=1.0,
        mean_abs_slow_state=1.5,
        slow_fast_contribution_ratio=1.5,
        mean_abs_decay_term=0.0,
        max_abs_decay_term=0.0,
        decay_effect_ratio=0.0,
        decay_near_zero_fraction=0.0,
        score_delay_3=0.0,
        score_delay_5=mean_score_over_delays,
        score_delay_8=score_delay_8,
        success_delay_3=False,
        success_delay_5=success,
        success_delay_8=success,
        mean_score_over_delays=mean_score_over_delays,
        delay_score_std=delay_score_std,
        delay_score_range=delay_score_std * 2.0,
        curriculum_enabled=True,
        curriculum_phase_1_delays="5",
        curriculum_phase_2_delays="5,8",
        curriculum_switch_generation=switch_generation,
        curriculum_phase=curriculum_phase,
        active_evaluation_delays="5" if curriculum_phase == "phase_1" else "5,8",
        score_current_phase=mean_score_over_delays,
    )
