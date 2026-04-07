from __future__ import annotations

from ui.cli import (
    CompareSummary,
    _parse_optional_delay_list,
    _build_benchmark_aggregates,
    _parse_variants,
    _pair_delay_summaries,
    _parse_delay_sweep,
    _parse_seed_list,
)


def test_parse_seed_list_accepts_comma_separated_values() -> None:
    assert _parse_seed_list("7, 11,13") == [7, 11, 13]


def test_parse_delay_sweep_uses_default_when_missing() -> None:
    assert _parse_delay_sweep(None, default_delay=3) == [3]
    assert _parse_delay_sweep("1,3,5", default_delay=3) == [1, 3, 5]


def test_parse_optional_delay_list_accepts_comma_separated_values() -> None:
    assert _parse_optional_delay_list(None) == ()
    assert _parse_optional_delay_list("5, 8") == (5, 8)


def test_parse_variants_accepts_stateful_stateless_and_plastic() -> None:
    assert _parse_variants(
        "stateful, stateful_v2, stateful_v2_gated, stateful_v3_kv, content_gated, stateless, stateful_plastic, stateful_plastic_hebb, stateful_plastic_ad, "
        "stateful_plastic_ad_narrow, stateful_plastic_ad_d0, stateful_plastic_ad_d005, "
        "stateful_plastic_ad_d01, stateful_plastic_ad_d02, stateful"
    ) == [
        "stateful",
        "stateful_v2",
        "stateful_v2_gated",
        "stateful_v3_kv",
        "content_gated",
        "stateless",
        "stateful_plastic",
        "stateful_plastic_hebb",
        "stateful_plastic_ad",
        "stateful_plastic_ad_narrow",
        "stateful_plastic_ad_d0",
        "stateful_plastic_ad_d005",
        "stateful_plastic_ad_d01",
        "stateful_plastic_ad_d02",
    ]


def test_build_benchmark_aggregates_summarizes_variants_per_delay() -> None:
    summaries = [
        CompareSummary(
            run_id="r1",
            variant="stateful",
            seed=7,
            task_name="bit_memory",
            delay_steps=1,
            status="finished",
            first_max_generation=2,
            first_success_generation=2,
            final_max_score=4.0,
            score_ceiling=4.0,
            success=True,
            avg_score_path="1.0->2.0->3.0",
            resume_hint="not_resumed",
        ),
        CompareSummary(
            run_id="r2",
            variant="stateful",
            seed=11,
            task_name="bit_memory",
            delay_steps=1,
            status="finished",
            first_max_generation=4,
            first_success_generation=None,
            final_max_score=3.5,
            score_ceiling=4.0,
            success=False,
            avg_score_path="1.0->2.0->3.0",
            resume_hint="not_resumed",
        ),
        CompareSummary(
            run_id="r3",
            variant="stateless",
            seed=7,
            task_name="bit_memory",
            delay_steps=1,
            status="finished",
            first_max_generation=7,
            first_success_generation=None,
            final_max_score=3.0,
            score_ceiling=4.0,
            success=False,
            avg_score_path="1.0->2.0->3.0",
            resume_hint="not_resumed",
        ),
    ]

    aggregates = _build_benchmark_aggregates(summaries)
    by_variant = {
        (aggregate.delay_steps, aggregate.variant): aggregate
        for aggregate in aggregates
    }

    assert by_variant[(1, "stateful")].run_count == 2
    assert by_variant[(1, "stateful")].completed_count == 2
    assert by_variant[(1, "stateful")].success_rate == 0.5
    assert by_variant[(1, "stateful")].mean_final_max_score == 3.75
    assert by_variant[(1, "stateful")].mean_first_max_generation == 3.0
    assert by_variant[(1, "stateful")].mean_first_success_generation == 2.0
    assert by_variant[(1, "stateless")].run_count == 1
    assert by_variant[(1, "stateless")].best_final_max_score == 3.0
    assert by_variant[(1, "stateless")].mean_first_success_generation is None


def test_pair_delay_summaries_matches_stateful_and_stateless_rows() -> None:
    summaries = [
        CompareSummary(
            run_id="r1",
            variant="stateful",
            seed=7,
            task_name="bit_memory",
            delay_steps=3,
            status="finished",
            first_max_generation=2,
            first_success_generation=2,
            final_max_score=4.0,
            score_ceiling=4.0,
            success=True,
            avg_score_path="1.0->2.0->3.0",
            resume_hint="not_resumed",
        ),
        CompareSummary(
            run_id="r2",
            variant="stateless",
            seed=7,
            task_name="bit_memory",
            delay_steps=3,
            status="finished",
            first_max_generation=8,
            first_success_generation=None,
            final_max_score=3.0,
            score_ceiling=4.0,
            success=False,
            avg_score_path="1.0->2.0->3.0",
            resume_hint="not_resumed",
        ),
    ]

    pairs = _pair_delay_summaries(summaries)

    assert pairs == [
        {
            "delay_steps": 3,
            "left_variant": "stateful",
            "right_variant": "stateless",
            "left_success_rate": 1.0,
            "right_success_rate": 0.0,
            "delta_success_rate": 1.0,
            "left_mean_first_success_generation": 2.0,
            "right_mean_first_success_generation": None,
            "delta_mean_final_max_score": 1.0,
        }
    ]
