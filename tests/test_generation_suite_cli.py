from __future__ import annotations

import json
from pathlib import Path

from ui.cli import (
    GenerationBenchmarkRow,
    _build_generation_suite_aggregates,
    _parse_task_delay_sweeps,
    _parse_task_list,
    _render_generation_suite_markdown,
    main,
)


def test_parse_task_list_accepts_generation_tasks() -> None:
    assert _parse_task_list("delayed_xor, bit_memory") == ["delayed_xor", "bit_memory"]
    assert _parse_task_list("key_value_memory, bit_memory") == ["key_value_memory", "bit_memory"]


def test_parse_task_delay_sweeps_accepts_task_specific_mappings() -> None:
    assert _parse_task_delay_sweeps(["bit_memory=1,3,5,8", "delayed_xor=1"]) == {
        "bit_memory": [1, 3, 5, 8],
        "delayed_xor": [1],
    }


def test_build_generation_suite_aggregates_summarizes_per_task_variant() -> None:
    rows = [
        GenerationBenchmarkRow(
            task_name="bit_memory",
            delay_steps=1,
            variant="stateful",
            seed=7,
            run_id="r1",
            status="finished",
            generation_budget=12,
            population_size=40,
            score_ceiling=4.0,
            success=True,
            first_success_generation=4,
            best_generation_id=4,
            best_candidate_id="c1",
            final_max_score=4.0,
            final_avg_score=3.5,
            completed_generations=12,
            best_node_count=6,
            best_enabled_connection_count=8,
            success_node_count=6,
            success_enabled_connection_count=8,
            avg_score_path="1.0->2.0->3.0",
            resume_hint="not_resumed",
        ),
        GenerationBenchmarkRow(
            task_name="bit_memory",
            delay_steps=1,
            variant="stateful",
            seed=11,
            run_id="r2",
            status="finished",
            generation_budget=12,
            population_size=40,
            score_ceiling=4.0,
            success=False,
            first_success_generation=None,
            best_generation_id=11,
            best_candidate_id="c2",
            final_max_score=3.5,
            final_avg_score=3.0,
            completed_generations=12,
            best_node_count=7,
            best_enabled_connection_count=10,
            success_node_count=None,
            success_enabled_connection_count=None,
            avg_score_path="1.0->2.0->3.0",
            resume_hint="not_resumed",
        ),
        GenerationBenchmarkRow(
            task_name="bit_memory",
            delay_steps=1,
            variant="stateless",
            seed=7,
            run_id="r3",
            status="finished",
            generation_budget=12,
            population_size=40,
            score_ceiling=4.0,
            success=False,
            first_success_generation=None,
            best_generation_id=11,
            best_candidate_id="c3",
            final_max_score=3.0,
            final_avg_score=2.5,
            completed_generations=12,
            best_node_count=5,
            best_enabled_connection_count=6,
            success_node_count=None,
            success_enabled_connection_count=None,
            avg_score_path="1.0->2.0->3.0",
            resume_hint="not_resumed",
        ),
        GenerationBenchmarkRow(
            task_name="bit_memory",
            delay_steps=1,
            variant="stateful_plastic",
            seed=13,
            run_id="r4",
            status="finished",
            generation_budget=12,
            population_size=40,
            score_ceiling=4.0,
            success=True,
            first_success_generation=3,
            best_generation_id=3,
            best_candidate_id="c4",
            final_max_score=4.0,
            final_avg_score=3.6,
            completed_generations=12,
            best_node_count=6,
            best_enabled_connection_count=7,
            success_node_count=6,
            success_enabled_connection_count=7,
            avg_score_path="1.0->2.0->3.0",
            resume_hint="not_resumed",
        ),
        GenerationBenchmarkRow(
            task_name="bit_memory",
            delay_steps=1,
            variant="stateful_plastic_ad",
            seed=17,
            run_id="r5",
            status="finished",
            generation_budget=12,
            population_size=40,
            score_ceiling=4.0,
            success=True,
            first_success_generation=5,
            best_generation_id=5,
            best_candidate_id="c5",
            final_max_score=3.8,
            final_avg_score=3.1,
            completed_generations=12,
            best_node_count=7,
            best_enabled_connection_count=9,
            success_node_count=7,
            success_enabled_connection_count=9,
            avg_score_path="1.0->2.0->3.0",
            resume_hint="not_resumed",
        ),
    ]

    aggregates = _build_generation_suite_aggregates(rows)
    by_key = {
        (aggregate.task_name, aggregate.variant): aggregate
        for aggregate in aggregates
    }

    assert by_key[("bit_memory", "stateful")].success_rate == 0.5
    assert by_key[("bit_memory", "stateful")].mean_final_max_score == 3.75
    assert by_key[("bit_memory", "stateful")].mean_first_success_generation == 4.0
    assert by_key[("bit_memory", "stateful")].median_first_success_generation == 4.0
    assert by_key[("bit_memory", "stateful")].mean_best_node_count == 6.5
    assert by_key[("bit_memory", "stateless")].mean_best_enabled_connection_count == 6.0
    assert by_key[("bit_memory", "stateful_plastic")].success_rate == 1.0
    assert by_key[("bit_memory", "stateful_plastic")].mean_first_success_generation == 3.0
    assert by_key[("bit_memory", "stateful_plastic_ad")].success_rate == 1.0
    assert by_key[("bit_memory", "stateful_plastic_ad")].mean_final_max_score == 3.8


def test_render_generation_suite_markdown_contains_summary_table() -> None:
    aggregates = _build_generation_suite_aggregates(
        [
            GenerationBenchmarkRow(
                task_name="delayed_xor",
                delay_steps=1,
                variant="stateful",
                seed=7,
                run_id="r1",
                status="finished",
                generation_budget=12,
                population_size=40,
                score_ceiling=4.0,
                success=True,
                first_success_generation=2,
                best_generation_id=2,
                best_candidate_id="c1",
                final_max_score=4.0,
                final_avg_score=3.5,
                completed_generations=12,
                best_node_count=5,
                best_enabled_connection_count=7,
                success_node_count=5,
                success_enabled_connection_count=7,
                avg_score_path="1.0->2.0->3.0",
                resume_hint="not_resumed",
            )
        ]
    )
    markdown = _render_generation_suite_markdown(
        label="demo-suite",
        rows=[],
        aggregates=aggregates,
        tasks=["delayed_xor"],
        seeds=[7],
    )

    assert "# Generation Benchmark Suite: demo-suite" in markdown
    assert "| task | delay | variant | runs | success_rate |" in markdown
    assert "| delayed_xor | 1 | stateful | 1 | 1.000 |" in markdown


def test_benchmark_suite_writes_exports(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark-suite",
            "--store",
            "memory",
            "--tasks",
            "delayed_xor,bit_memory",
            "--seeds",
            "7",
            "--variants",
            "stateful,stateful_plastic_ad_d005",
            "--task-delay-sweep",
            "bit_memory=1,3",
            "--generations",
            "2",
            "--population-size",
            "4",
            "--output-dir",
            str(tmp_path),
            "--label",
            "smoke-suite",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "smoke-suite.jsonl").exists()
    assert (tmp_path / "smoke-suite.csv").exists()
    assert (tmp_path / "smoke-suite.md").exists()
    assert (tmp_path / "smoke-suite.candidate-features.jsonl").exists()
    assert (tmp_path / "smoke-suite.archive-cells.jsonl").exists()
    assert (tmp_path / "smoke-suite.archive-events.jsonl").exists()
    archive_cells = [
        json.loads(line)
        for line in (tmp_path / "smoke-suite.archive-cells.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["qd_profile"] for row in archive_cells} == {
        "mechanism_v2",
        "general_compactness",
        "delay_robustness",
        "curriculum_progress",
    }
    markdown = (tmp_path / "smoke-suite.md").read_text(encoding="utf-8")
    assert "| bit_memory | 1 |" in markdown
    assert "| bit_memory | 3 |" in markdown
    assert "| bit_memory | 1 | stateful_plastic_ad_d005 |" in markdown


def test_benchmark_suite_curriculum_exports_include_switch_generation(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark-suite",
            "--store",
            "memory",
            "--tasks",
            "bit_memory",
            "--seeds",
            "7",
            "--variants",
            "stateful_v2",
            "--generations",
            "2",
            "--population-size",
            "4",
            "--temporal-delay-steps",
            "8",
            "--curriculum-enabled",
            "--curriculum-phase-1-delays",
            "5",
            "--curriculum-phase-2-delays",
            "5,8",
            "--curriculum-phase-switch-generation",
            "4",
            "--output-dir",
            str(tmp_path),
            "--label",
            "curriculum-suite",
        ]
    )

    assert exit_code == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "curriculum-suite.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert rows[0]["curriculum_enabled"] is True
    assert rows[0]["curriculum_phase_1_delays"] == "5"
    assert rows[0]["curriculum_phase_2_delays"] == "5,8"
    assert rows[0]["curriculum_switch_generation"] == 4
    feature_rows = [
        json.loads(line)
        for line in (tmp_path / "curriculum-suite.candidate-features.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert feature_rows
    assert feature_rows[0]["curriculum_switch_generation"] == 4


def test_benchmark_suite_key_value_memory_supports_profile_overrides(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark-suite",
            "--store",
            "memory",
            "--tasks",
            "key_value_memory",
            "--seeds",
            "7",
            "--variants",
            "stateful_v2",
            "--generations",
            "1",
            "--population-size",
            "4",
            "--curriculum-enabled",
            "--curriculum-phase-1-delays",
            "3",
            "--curriculum-phase-2-delays",
            "8",
            "--curriculum-phase-switch-generation",
            "1",
            "--key-value-profile",
            "kv_full",
            "--curriculum-phase-1-key-value-profile",
            "kv_easy",
            "--curriculum-phase-2-key-value-profile",
            "kv_mid",
            "--output-dir",
            str(tmp_path),
            "--label",
            "kv-suite",
        ]
    )

    assert exit_code == 0
    archive_cells = [
        json.loads(line)
        for line in (tmp_path / "kv-suite.archive-cells.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["qd_profile"] for row in archive_cells} == {
        "mechanism_v2",
        "general_compactness",
        "delay_robustness",
        "curriculum_progress",
        "retrieval_strategy",
        "retrieval_mechanism",
        "gating_mechanism",
        "content_retrieval",
        "kv_retrieval_mechanism",
        "slot_retrieval_mechanism",
        "addressed_slot_mechanism",
    }
    markdown = (tmp_path / "kv-suite.md").read_text(encoding="utf-8")
    assert "## Retrieval Diagnostics" in markdown
    assert "mean_correct_key_selected" in markdown
    assert "## KV Selectivity Diagnostics" in markdown
    assert "mean_store_vs_distractor_write_gap" in markdown
    assert "## Slot Retrieval Diagnostics" in markdown
    assert "mean_query_slot_match_max" in markdown
    feature_rows = [
        json.loads(line)
        for line in (tmp_path / "kv-suite.candidate-features.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert feature_rows
    assert feature_rows[0]["task_name"] == "key_value_memory"


def test_benchmark_suite_supports_v11c_kv_preset_overlay(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark-suite",
            "--store",
            "memory",
            "--tasks",
            "key_value_memory",
            "--seeds",
            "7",
            "--variants",
            "stateful_v3_kv",
            "--generations",
            "1",
            "--population-size",
            "4",
            "--config",
            "configs/v11c_kv_selective.yaml",
            "--output-dir",
            str(tmp_path),
            "--label",
            "kv-v11c-selective-smoke",
        ]
    )
    assert exit_code == 0
    markdown = (tmp_path / "kv-v11c-selective-smoke.md").read_text(encoding="utf-8")
    assert "stateful_v3_kv" in markdown


def test_benchmark_suite_supports_v11d_kv_preset_overlay(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark-suite",
            "--store",
            "memory",
            "--tasks",
            "key_value_memory",
            "--seeds",
            "7",
            "--variants",
            "stateful_v3_kv",
            "--generations",
            "1",
            "--population-size",
            "4",
            "--config",
            "configs/v11d_kv_conservative_plus.yaml",
            "--output-dir",
            str(tmp_path),
            "--label",
            "kv-v11d-conservative-plus-smoke",
        ]
    )
    assert exit_code == 0
    markdown = (tmp_path / "kv-v11d-conservative-plus-smoke.md").read_text(encoding="utf-8")
    assert "stateful_v3_kv" in markdown


def test_benchmark_suite_supports_v12b_slots_querysharp_overlay(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark-suite",
            "--store",
            "memory",
            "--tasks",
            "key_value_memory",
            "--seeds",
            "7",
            "--variants",
            "stateful_v4_slots",
            "--generations",
            "1",
            "--population-size",
            "4",
            "--config",
            "configs/v12b_slots_querysharp.yaml",
            "--output-dir",
            str(tmp_path),
            "--label",
            "kv-v12b-querysharp-smoke",
        ]
    )
    assert exit_code == 0
    markdown = (tmp_path / "kv-v12b-querysharp-smoke.md").read_text(encoding="utf-8")
    assert "stateful_v4_slots" in markdown


def test_benchmark_suite_supports_v12b_slots_readoutsharp_overlay(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark-suite",
            "--store",
            "memory",
            "--tasks",
            "key_value_memory",
            "--seeds",
            "7",
            "--variants",
            "stateful_v4_slots",
            "--generations",
            "1",
            "--population-size",
            "4",
            "--config",
            "configs/v12b_slots_readoutsharp.yaml",
            "--output-dir",
            str(tmp_path),
            "--label",
            "kv-v12b-readoutsharp-smoke",
        ]
    )
    assert exit_code == 0
    markdown = (tmp_path / "kv-v12b-readoutsharp-smoke.md").read_text(encoding="utf-8")
    assert "stateful_v4_slots" in markdown


def test_benchmark_suite_supports_v12c_slots_readoutplus_overlay(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark-suite",
            "--store",
            "memory",
            "--tasks",
            "key_value_memory",
            "--seeds",
            "7",
            "--variants",
            "stateful_v4_slots",
            "--generations",
            "1",
            "--population-size",
            "4",
            "--config",
            "configs/v12c_slots_readoutplus.yaml",
            "--output-dir",
            str(tmp_path),
            "--label",
            "kv-v12c-readoutplus-smoke",
        ]
    )
    assert exit_code == 0
    markdown = (tmp_path / "kv-v12c-readoutplus-smoke.md").read_text(encoding="utf-8")
    assert "stateful_v4_slots" in markdown


def test_benchmark_suite_supports_v12c_slots_focusplus_overlay(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark-suite",
            "--store",
            "memory",
            "--tasks",
            "key_value_memory",
            "--seeds",
            "7",
            "--variants",
            "stateful_v4_slots",
            "--generations",
            "1",
            "--population-size",
            "4",
            "--config",
            "configs/v12c_slots_focusplus.yaml",
            "--output-dir",
            str(tmp_path),
            "--label",
            "kv-v12c-focusplus-smoke",
        ]
    )
    assert exit_code == 0
    markdown = (tmp_path / "kv-v12c-focusplus-smoke.md").read_text(encoding="utf-8")
    assert "stateful_v4_slots" in markdown


def test_benchmark_suite_supports_v12d_slots_conservative_plus_overlay(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark-suite",
            "--store",
            "memory",
            "--tasks",
            "key_value_memory",
            "--seeds",
            "7",
            "--variants",
            "stateful_v4_slots",
            "--generations",
            "1",
            "--population-size",
            "4",
            "--config",
            "configs/v12d_slots_conservative_plus.yaml",
            "--output-dir",
            str(tmp_path),
            "--label",
            "kv-v12d-conservative-plus-smoke",
        ]
    )
    assert exit_code == 0
    markdown = (tmp_path / "kv-v12d-conservative-plus-smoke.md").read_text(encoding="utf-8")
    assert "stateful_v4_slots" in markdown


def test_benchmark_suite_supports_v12d_slots_margin_plus_overlay(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark-suite",
            "--store",
            "memory",
            "--tasks",
            "key_value_memory",
            "--seeds",
            "7",
            "--variants",
            "stateful_v4_slots",
            "--generations",
            "1",
            "--population-size",
            "4",
            "--config",
            "configs/v12d_slots_margin_plus.yaml",
            "--output-dir",
            str(tmp_path),
            "--label",
            "kv-v12d-margin-plus-smoke",
        ]
    )
    assert exit_code == 0
    markdown = (tmp_path / "kv-v12d-margin-plus-smoke.md").read_text(encoding="utf-8")
    assert "stateful_v4_slots" in markdown
