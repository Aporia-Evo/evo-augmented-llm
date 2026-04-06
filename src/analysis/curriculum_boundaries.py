from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

from analysis.search_space import filter_feature_records, load_feature_records_from_jsonl


@dataclass(frozen=True)
class CurriculumBoundarySummary:
    benchmark_label: str
    task_name: str
    variant: str
    curriculum_enabled: bool
    curriculum_phase_1_delays: str
    curriculum_phase_2_delays: str
    curriculum_switch_generation: int
    success_rate: float
    mean_final_max_score: float | None
    mean_first_success_generation: float | None
    mean_score_over_delays: float | None
    mean_score_delay_8: float | None
    mean_delay_score_std: float | None
    post_switch_candidate_count: int
    post_switch_success_rate: float
    post_switch_mean_score: float | None
    post_switch_mean_score_over_delays: float | None
    post_switch_score_delay_8: float | None
    post_switch_delay_score_std: float | None
    post_switch_delay_score_range: float | None
    post_switch_best_score: float | None
    post_switch_first_success_generation: float | None


@dataclass(frozen=True)
class CurriculumPhaseSummary:
    benchmark_label: str
    task_name: str
    variant: str
    curriculum_switch_generation: int
    curriculum_phase: str
    candidates: int
    success_rate: float
    mean_score_current_phase: float | None
    mean_score_over_delays: float | None
    mean_score_delay_8: float | None
    mean_delay_score_std: float | None
    mean_delay_score_range: float | None


def render_curriculum_boundary_report(
    *,
    output_dir: Path,
    benchmark_labels: Sequence[str],
    task_name: str,
    variants: Sequence[str],
    focus_variant: str,
) -> str:
    summaries = _load_boundary_summaries(
        output_dir=output_dir,
        benchmark_labels=benchmark_labels,
        task_name=task_name,
        variants=variants,
    )
    if not summaries:
        return "No curriculum boundary summaries found for the requested labels.\n"

    phase_summaries = _load_phase_summaries(
        output_dir=output_dir,
        benchmark_labels=benchmark_labels,
        task_name=task_name,
        variants=variants,
    )
    focus_rows = [summary for summary in summaries if summary.variant == focus_variant]

    lines = [
        "## Curriculum Boundary Summary",
        "",
        "## Overall Summary",
        "",
        _render_table(
            [
                "benchmark_label",
                "variant",
                "switch_generation",
                "phase_1_delays",
                "phase_2_delays",
                "success_rate",
                "mean_final_max_score",
                "mean_first_success_generation",
                "mean_score_over_delays",
                "score_delay_8",
                "delay_score_std",
            ],
            [
                [
                    summary.benchmark_label,
                    summary.variant,
                    str(summary.curriculum_switch_generation),
                    summary.curriculum_phase_1_delays,
                    summary.curriculum_phase_2_delays,
                    _format_float(summary.success_rate, precision=3),
                    _format_float(summary.mean_final_max_score, precision=6),
                    _format_float(summary.mean_first_success_generation, precision=2),
                    _format_float(summary.mean_score_over_delays, precision=6),
                    _format_float(summary.mean_score_delay_8, precision=6),
                    _format_float(summary.mean_delay_score_std, precision=6),
                ]
                for summary in summaries
            ],
        ),
        "",
        "## Post-Switch Summary",
        "",
        _render_table(
            [
                "benchmark_label",
                "variant",
                "switch_generation",
                "post_switch_candidates",
                "post_switch_success_rate",
                "post_switch_mean_score",
                "post_switch_mean_score_over_delays",
                "post_switch_score_delay_8",
                "post_switch_delay_score_std",
                "post_switch_delay_score_range",
            ],
            [
                [
                    summary.benchmark_label,
                    summary.variant,
                    str(summary.curriculum_switch_generation),
                    str(summary.post_switch_candidate_count),
                    _format_float(summary.post_switch_success_rate, precision=3),
                    _format_float(summary.post_switch_mean_score, precision=6),
                    _format_float(summary.post_switch_mean_score_over_delays, precision=6),
                    _format_float(summary.post_switch_score_delay_8, precision=6),
                    _format_float(summary.post_switch_delay_score_std, precision=6),
                    _format_float(summary.post_switch_delay_score_range, precision=6),
                ]
                for summary in summaries
            ],
        ),
        "",
    ]

    if focus_rows:
        lines.extend(
            [
                f"## Focus Variant: {focus_variant} (Overall)",
                "",
                _render_table(
                    [
                        "switch_generation",
                        "success_rate",
                        "mean_final_max_score",
                        "mean_first_success_generation",
                        "mean_score_over_delays",
                        "score_delay_8",
                        "delay_score_std",
                    ],
                    [
                        [
                            str(summary.curriculum_switch_generation),
                            _format_float(summary.success_rate, precision=3),
                            _format_float(summary.mean_final_max_score, precision=6),
                            _format_float(summary.mean_first_success_generation, precision=2),
                            _format_float(summary.mean_score_over_delays, precision=6),
                            _format_float(summary.mean_score_delay_8, precision=6),
                            _format_float(summary.mean_delay_score_std, precision=6),
                        ]
                        for summary in focus_rows
                    ],
                ),
                "",
                f"## Focus Variant: {focus_variant} (Post-Switch)",
                "",
                _render_table(
                    [
                        "switch_generation",
                        "post_switch_success_rate",
                        "post_switch_mean_score",
                        "post_switch_mean_score_over_delays",
                        "post_switch_score_delay_8",
                        "post_switch_delay_score_std",
                        "post_switch_delay_score_range",
                    ],
                    [
                        [
                            str(summary.curriculum_switch_generation),
                            _format_float(summary.post_switch_success_rate, precision=3),
                            _format_float(summary.post_switch_mean_score, precision=6),
                            _format_float(summary.post_switch_mean_score_over_delays, precision=6),
                            _format_float(summary.post_switch_score_delay_8, precision=6),
                            _format_float(summary.post_switch_delay_score_std, precision=6),
                            _format_float(summary.post_switch_delay_score_range, precision=6),
                        ]
                        for summary in focus_rows
                    ],
                ),
                "",
            ]
        )

    focus_duel_rows = [
        summary
        for summary in focus_rows
        if summary.curriculum_switch_generation in {6, 8}
    ]
    if focus_duel_rows:
        lines.extend(
            [
                f"## Focus Duel: {focus_variant} (g6 vs g8)",
                "",
                _render_table(
                    [
                        "switch_generation",
                        "overall_success_rate",
                        "overall_mean_score_over_delays",
                        "post_switch_success_rate",
                        "post_switch_mean_score_over_delays",
                        "post_switch_score_delay_8",
                        "post_switch_delay_score_std",
                    ],
                    [
                        [
                            str(summary.curriculum_switch_generation),
                            _format_float(summary.success_rate, precision=3),
                            _format_float(summary.mean_score_over_delays, precision=6),
                            _format_float(summary.post_switch_success_rate, precision=3),
                            _format_float(summary.post_switch_mean_score_over_delays, precision=6),
                            _format_float(summary.post_switch_score_delay_8, precision=6),
                            _format_float(summary.post_switch_delay_score_std, precision=6),
                        ]
                        for summary in focus_duel_rows
                    ],
                ),
                "",
            ]
        )

    focus_phase_rows = [summary for summary in phase_summaries if summary.variant == focus_variant]
    if focus_phase_rows:
        lines.extend(
            [
                f"## Phase Dynamics: {focus_variant}",
                "",
                _render_table(
                    [
                        "switch_generation",
                        "phase",
                        "candidates",
                        "success_rate",
                        "mean_score_current_phase",
                        "mean_score_over_delays",
                        "score_delay_8",
                        "delay_score_std",
                        "delay_score_range",
                    ],
                    [
                        [
                            str(summary.curriculum_switch_generation),
                            summary.curriculum_phase,
                            str(summary.candidates),
                            _format_float(summary.success_rate, precision=3),
                            _format_float(summary.mean_score_current_phase, precision=6),
                            _format_float(summary.mean_score_over_delays, precision=6),
                            _format_float(summary.mean_score_delay_8, precision=6),
                            _format_float(summary.mean_delay_score_std, precision=6),
                            _format_float(summary.mean_delay_score_range, precision=6),
                        ]
                        for summary in focus_phase_rows
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Decision Rule",
            "",
            "- Prioritaet fuer den Performance-Sieger von `stateful_v2`: `post_switch_success_rate`, dann `post_switch_mean_score_over_delays`, dann `post_switch_score_delay_8`, dann niedrige `post_switch_delay_score_std`.",
            "- Gesamtmetriken bleiben sichtbar, entscheiden aber nicht mehr primaer ueber den Curriculum-Boundary.",
            "",
        ]
    )

    performance_winner = _pick_boundary_winner(focus_rows)
    runner_up = _pick_runner_up(focus_rows, performance_winner)
    if performance_winner is not None:
        lines.extend(
            [
                "## Boundary Winner",
                "",
                f"- `stateful_v2` gewinnt in der harten Phase aktuell bei `switch_generation={performance_winner.curriculum_switch_generation}`.",
                f"- Grundlage: `post_switch_success_rate={performance_winner.post_switch_success_rate:.3f}`, `post_switch_mean_score_over_delays={_format_float(performance_winner.post_switch_mean_score_over_delays, precision=6)}`, `post_switch_score_delay_8={_format_float(performance_winner.post_switch_score_delay_8, precision=6)}`, `post_switch_delay_score_std={_format_float(performance_winner.post_switch_delay_score_std, precision=6)}`.",
                "",
            ]
        )
        if runner_up is not None:
            lines.extend(
                [
                    "- Runner-up:",
                    f"  `switch_generation={runner_up.curriculum_switch_generation}` mit `post_switch_success_rate={runner_up.post_switch_success_rate:.3f}`, `post_switch_mean_score_over_delays={_format_float(runner_up.post_switch_mean_score_over_delays, precision=6)}`.",
                    "",
                ]
            )

    default_boundary = _pick_default_boundary(focus_rows)
    if default_boundary is not None or performance_winner is not None:
        lines.extend(
            [
                "## Boundary Policy",
                "",
            ]
        )
        if default_boundary is not None:
            lines.append(
                f"- `default_boundary={default_boundary.curriculum_switch_generation}` fuer konservative Reproduzierbarkeit: kleinster Boundary mit bester Overall-`success_rate` von `stateful_v2`."
            )
        if performance_winner is not None:
            lines.append(
                f"- `performance_boundary={performance_winner.curriculum_switch_generation}` fuer harte Post-Switch-Leistung: Sieger nach Phase-2-Metriken."
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _load_boundary_summaries(
    *,
    output_dir: Path,
    benchmark_labels: Sequence[str],
    task_name: str,
    variants: Sequence[str],
) -> list[CurriculumBoundarySummary]:
    summaries: list[CurriculumBoundarySummary] = []
    for benchmark_label in benchmark_labels:
        rows = _load_generation_rows(_suite_export_path(output_dir, benchmark_label))
        feature_records = load_feature_records_from_jsonl(_feature_export_path(output_dir, benchmark_label))
        for variant in variants:
            variant_rows = [
                row
                for row in rows
                if row.get("task_name") == task_name and row.get("variant") == variant
            ]
            if not variant_rows:
                continue
            variant_features = filter_feature_records(
                feature_records,
                benchmark_label=benchmark_label,
                task_name=task_name,
                variant=variant,
            )
            phase_two_features = filter_feature_records(
                feature_records,
                benchmark_label=benchmark_label,
                task_name=task_name,
                variant=variant,
                curriculum_phase="phase_2",
            )
            metadata_row = variant_rows[0]
            summaries.append(
                CurriculumBoundarySummary(
                    benchmark_label=benchmark_label,
                    task_name=task_name,
                    variant=variant,
                    curriculum_enabled=bool(metadata_row.get("curriculum_enabled", False)),
                    curriculum_phase_1_delays=str(metadata_row.get("curriculum_phase_1_delays", "")),
                    curriculum_phase_2_delays=str(metadata_row.get("curriculum_phase_2_delays", "")),
                    curriculum_switch_generation=int(metadata_row.get("curriculum_switch_generation", 0)),
                    success_rate=_fraction(bool(row.get("success", False)) for row in variant_rows),
                    mean_final_max_score=_mean_nullable(
                        float(row["final_max_score"])
                        for row in variant_rows
                        if row.get("final_max_score") is not None
                    ),
                    mean_first_success_generation=_mean_nullable(
                        float(row["first_success_generation"])
                        for row in variant_rows
                        if row.get("first_success_generation") is not None
                    ),
                    mean_score_over_delays=_mean_nullable(
                        record.mean_score_over_delays for record in variant_features
                    ),
                    mean_score_delay_8=_mean_nullable(record.score_delay_8 for record in variant_features),
                    mean_delay_score_std=_mean_nullable(record.delay_score_std for record in variant_features),
                    post_switch_candidate_count=len(phase_two_features),
                    post_switch_success_rate=_fraction(record.success for record in phase_two_features),
                    post_switch_mean_score=_mean_nullable(record.score_current_phase for record in phase_two_features),
                    post_switch_mean_score_over_delays=_mean_nullable(
                        record.mean_score_over_delays for record in phase_two_features
                    ),
                    post_switch_score_delay_8=_mean_nullable(
                        record.score_delay_8 for record in phase_two_features
                    ),
                    post_switch_delay_score_std=_mean_nullable(
                        record.delay_score_std for record in phase_two_features
                    ),
                    post_switch_delay_score_range=_mean_nullable(
                        record.delay_score_range for record in phase_two_features
                    ),
                    post_switch_best_score=max(
                        (record.final_max_score for record in phase_two_features),
                        default=None,
                    ),
                    post_switch_first_success_generation=_mean_nullable(
                        float(record.generation)
                        for record in phase_two_features
                        if record.success
                    ),
                )
            )
    return sorted(
        summaries,
        key=lambda summary: (
            summary.variant,
            summary.curriculum_switch_generation,
            summary.benchmark_label,
        ),
    )


def _load_phase_summaries(
    *,
    output_dir: Path,
    benchmark_labels: Sequence[str],
    task_name: str,
    variants: Sequence[str],
) -> list[CurriculumPhaseSummary]:
    summaries: list[CurriculumPhaseSummary] = []
    for benchmark_label in benchmark_labels:
        feature_records = load_feature_records_from_jsonl(_feature_export_path(output_dir, benchmark_label))
        for variant in variants:
            variant_features = filter_feature_records(
                feature_records,
                benchmark_label=benchmark_label,
                task_name=task_name,
                variant=variant,
            )
            if not variant_features:
                continue
            switch_generation = variant_features[0].curriculum_switch_generation
            grouped = {
                phase: [record for record in variant_features if record.curriculum_phase == phase]
                for phase in sorted({record.curriculum_phase for record in variant_features})
            }
            for phase, phase_records in grouped.items():
                if not phase_records:
                    continue
                summaries.append(
                    CurriculumPhaseSummary(
                        benchmark_label=benchmark_label,
                        task_name=task_name,
                        variant=variant,
                        curriculum_switch_generation=switch_generation,
                        curriculum_phase=phase,
                        candidates=len(phase_records),
                        success_rate=_fraction(record.success for record in phase_records),
                        mean_score_current_phase=_mean_nullable(record.score_current_phase for record in phase_records),
                        mean_score_over_delays=_mean_nullable(record.mean_score_over_delays for record in phase_records),
                        mean_score_delay_8=_mean_nullable(record.score_delay_8 for record in phase_records),
                        mean_delay_score_std=_mean_nullable(record.delay_score_std for record in phase_records),
                        mean_delay_score_range=_mean_nullable(record.delay_score_range for record in phase_records),
                    )
                )
    return sorted(
        summaries,
        key=lambda summary: (
            summary.curriculum_switch_generation,
            summary.variant,
            summary.curriculum_phase,
            summary.benchmark_label,
        ),
    )


def _pick_boundary_winner(
    summaries: Sequence[CurriculumBoundarySummary],
) -> CurriculumBoundarySummary | None:
    if not summaries:
        return None
    ranked = sorted(summaries, key=_performance_sort_key, reverse=True)
    return ranked[0]


def _pick_runner_up(
    summaries: Sequence[CurriculumBoundarySummary],
    winner: CurriculumBoundarySummary | None,
) -> CurriculumBoundarySummary | None:
    if winner is None:
        return None
    ranked = [summary for summary in sorted(summaries, key=_performance_sort_key, reverse=True) if summary != winner]
    return ranked[0] if ranked else None


def _pick_default_boundary(
    summaries: Sequence[CurriculumBoundarySummary],
) -> CurriculumBoundarySummary | None:
    if not summaries:
        return None
    best_overall_success = max(summary.success_rate for summary in summaries)
    candidates = [
        summary
        for summary in summaries
        if math.isclose(summary.success_rate, best_overall_success, rel_tol=0.0, abs_tol=1e-12)
    ]
    return sorted(
        candidates,
        key=lambda summary: (
            summary.curriculum_switch_generation,
            -_or_neg_inf(summary.mean_score_over_delays),
            -_or_neg_inf(summary.post_switch_mean_score_over_delays),
            _or_pos_inf(summary.post_switch_delay_score_std),
        ),
    )[0]


def _performance_sort_key(summary: CurriculumBoundarySummary) -> tuple[float, float, float, float, int]:
    return (
        summary.post_switch_success_rate,
        _or_neg_inf(summary.post_switch_mean_score_over_delays),
        _or_neg_inf(summary.post_switch_score_delay_8),
        -_or_pos_inf(summary.post_switch_delay_score_std),
        -summary.curriculum_switch_generation,
    )


def _suite_export_path(output_dir: Path, benchmark_label: str) -> Path:
    return output_dir / f"{benchmark_label}.jsonl"


def _feature_export_path(output_dir: Path, benchmark_label: str) -> Path:
    return output_dir / f"{benchmark_label}.candidate-features.jsonl"


def _load_generation_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark suite export not found: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean_nullable(values: Iterable[float]) -> float | None:
    collected = [float(value) for value in values]
    if not collected:
        return None
    return float(mean(collected))


def _fraction(values: Iterable[bool]) -> float:
    collected = [bool(value) for value in values]
    if not collected:
        return 0.0
    return sum(1 for value in collected if value) / len(collected)


def _or_neg_inf(value: float | None) -> float:
    return value if value is not None else float("-inf")


def _or_pos_inf(value: float | None) -> float:
    return value if value is not None else float("inf")


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _format_float(value: float | None, *, precision: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"
