from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

from db.models import ArchiveCellRecord, ArchiveEventRecord, CandidateFeatureRecord
from evolve.archive import (
    DEFAULT_QD_PROFILE,
    QD_PROFILE_CURRICULUM_PROGRESS,
    QD_PROFILE_DELAY_ROBUSTNESS,
    QD_PROFILE_GENERAL_COMPACTNESS,
    QD_PROFILE_RETRIEVAL_MECHANISM,
    QD_PROFILE_RETRIEVAL_STRATEGY,
    archive_profile_definition,
)
from utils.serialization import stable_json_dumps


def load_archive_cells_from_jsonl(path: Path) -> list[ArchiveCellRecord]:
    records: list[ArchiveCellRecord] = []
    if not path.exists():
        raise FileNotFoundError(f"Archive export not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            records.append(ArchiveCellRecord(**payload))
    return records


def write_archive_cells_jsonl(path: Path, records: Sequence[ArchiveCellRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json_dumps(asdict(record)))
            handle.write("\n")


def load_archive_events_from_jsonl(path: Path) -> list[ArchiveEventRecord]:
    records: list[ArchiveEventRecord] = []
    if not path.exists():
        raise FileNotFoundError(f"Archive export not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            records.append(ArchiveEventRecord(**payload))
    return records


def write_archive_events_jsonl(path: Path, records: Sequence[ArchiveEventRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json_dumps(asdict(record)))
            handle.write("\n")


def filter_archive_cells(
    records: Iterable[ArchiveCellRecord],
    *,
    benchmark_label: str,
    task_name: str | None = None,
    variant: str | None = None,
    delay_steps: int | None = None,
    qd_profile: str | None = None,
    curriculum_phase: str | None = None,
) -> list[ArchiveCellRecord]:
    filtered = [record for record in records if record.benchmark_label == benchmark_label]
    if task_name is not None:
        filtered = [record for record in filtered if record.task_name == task_name]
    if variant is not None:
        filtered = [record for record in filtered if record.variant == variant]
    if delay_steps is not None:
        filtered = [record for record in filtered if record.delay_steps == delay_steps]
    if qd_profile is not None:
        filtered = [record for record in filtered if record.qd_profile == qd_profile]
    if curriculum_phase is not None:
        filtered = [record for record in filtered if _curriculum_phase_for_record(record) == curriculum_phase]
    filtered.sort(
        key=lambda record: (
            record.task_name,
            record.delay_steps,
            record.variant,
            record.qd_profile,
            -record.elite_score,
            record.archive_id,
        )
    )
    return filtered


def filter_archive_events(
    records: Iterable[ArchiveEventRecord],
    *,
    benchmark_label: str,
    task_name: str | None = None,
    variant: str | None = None,
    delay_steps: int | None = None,
    qd_profile: str | None = None,
) -> list[ArchiveEventRecord]:
    filtered = [record for record in records if record.benchmark_label == benchmark_label]
    if task_name is not None:
        filtered = [record for record in filtered if record.task_name == task_name]
    if variant is not None:
        filtered = [record for record in filtered if record.variant == variant]
    if delay_steps is not None:
        filtered = [record for record in filtered if record.delay_steps == delay_steps]
    if qd_profile is not None:
        filtered = [record for record in filtered if record.qd_profile == qd_profile]
    filtered.sort(
        key=lambda record: (
            record.task_name,
            record.delay_steps,
            record.variant,
            record.qd_profile,
            record.created_at,
            record.event_id,
        )
    )
    return filtered


def render_archive_report(
    archive_cells: Sequence[ArchiveCellRecord],
    *,
    feature_records: Sequence[CandidateFeatureRecord],
    archive_events: Sequence[ArchiveEventRecord] | None = None,
    curriculum_phase_filter: str | None = None,
) -> str:
    if not archive_cells:
        return "No archive cells found for the requested filter."

    qd_profiles = sorted({record.qd_profile for record in archive_cells})
    if len(qd_profiles) != 1:
        return "Archive report requires a single qd_profile filter."
    qd_profile = qd_profiles[0] or DEFAULT_QD_PROFILE
    profile = archive_profile_definition(qd_profile)

    event_records = archive_events or []
    features_by_candidate = {record.candidate_id: record for record in feature_records}
    delays = sorted({record.delay_steps for record in archive_cells})
    total_cells = len(delays) * profile.total_cells_per_delay
    unique_secondary_bins = sorted(
        {
            int(_descriptor_payload(record).get(profile.secondary_axis_bin_key, 0))
            for record in archive_cells
        }
    )

    summary_title = f"## Archive Summary ({qd_profile})"
    if curriculum_phase_filter is not None:
        summary_title = f"## Archive Summary ({qd_profile}, {curriculum_phase_filter})"

    lines = [
        summary_title,
        "",
        _render_table(
            [
                "occupied_cells",
                "total_cells",
                "archive_coverage",
                "archive_best_score",
                "archive_mean_elite_score",
                "strategy_diversity",
            ],
            [
                [
                    str(len(archive_cells)),
                    str(total_cells),
                    _format_stat(len(archive_cells) / total_cells if total_cells else None, precision=3),
                    _format_stat(max(record.elite_score for record in archive_cells), precision=6),
                    _format_stat(mean(record.elite_score for record in archive_cells), precision=6),
                    str(len(unique_secondary_bins)),
                ]
            ],
        ),
        "",
    ]
    if any(record.curriculum_enabled for record in archive_cells):
        lines.extend(
            [
                "## Curriculum Metadata",
                "",
                _render_table(
                    [
                        "curriculum_enabled",
                        "phase_1_delays",
                        "phase_2_delays",
                        "switch_generation",
                    ],
                    [
                        [
                            "true",
                            ", ".join(sorted({record.curriculum_phase_1_delays for record in archive_cells if record.curriculum_enabled})),
                            ", ".join(sorted({record.curriculum_phase_2_delays for record in archive_cells if record.curriculum_enabled})),
                            ", ".join(
                                str(value)
                                for value in sorted(
                                    {record.curriculum_switch_generation for record in archive_cells if record.curriculum_enabled}
                                )
                            ),
                        ]
                    ],
                ),
                "",
            ]
        )

    if len(delays) > 1:
        lines.extend(
            [
                "## By Delay",
                "",
                _render_table(
                    [
                        "delay",
                        "occupied_cells",
                        "coverage",
                        "best_elite_score",
                        "mean_elite_score",
                        profile.secondary_axis_label + "_bins",
                    ],
                    [
                        [
                            str(delay),
                            str(len(group)),
                            _format_stat(len(group) / profile.total_cells_per_delay, precision=3),
                            _format_stat(max(cell.elite_score for cell in group), precision=6),
                            _format_stat(mean(cell.elite_score for cell in group), precision=6),
                            str(len({int(_descriptor_payload(cell).get(profile.secondary_axis_bin_key, 0)) for cell in group})),
                        ]
                        for delay, group in _group_by_delay(archive_cells)
                    ],
                ),
                "",
            ]
        )
    if qd_profile == QD_PROFILE_CURRICULUM_PROGRESS:
        lines.extend(
            [
                "## By Phase",
                "",
                _render_table(
                    [
                        "phase",
                        "occupied_cells",
                        "best_elite_score",
                        "mean_elite_score",
                    ],
                    [
                        [
                            phase,
                            str(len(group)),
                            _format_stat(max(cell.elite_score for cell in group), precision=6),
                            _format_stat(mean(cell.elite_score for cell in group), precision=6),
                        ]
                        for phase, group in _group_by_curriculum_phase(archive_cells)
                    ],
                ),
                "",
            ]
        )

    if event_records:
        event_counts = _event_type_counts(event_records)
        lines.extend(
            [
                "## Archive Events",
                "",
                _render_table(
                    ["insert", "replace", "skip"],
                    [[str(event_counts.get("insert", 0)), str(event_counts.get("replace", 0)), str(event_counts.get("skip", 0))]],
                ),
                "",
            ]
        )

    lines.extend(_render_top_cells_section(archive_cells, features_by_candidate, qd_profile=qd_profile))

    lines.extend(
        [
            "## Hints",
            "",
        ]
    )
    lines.extend(f"- {hint}" for hint in derive_archive_hints(archive_cells, feature_records=feature_records))
    return "\n".join(lines).rstrip() + "\n"


def derive_archive_hints(
    archive_cells: Sequence[ArchiveCellRecord],
    *,
    feature_records: Sequence[CandidateFeatureRecord],
) -> list[str]:
    if not archive_cells:
        return ["No archive cells found."]

    qd_profiles = sorted({record.qd_profile for record in archive_cells})
    if len(qd_profiles) != 1:
        return ["Das Archiv enthaelt mehrere QD-Profile; bitte erst nach Profil filtern."]

    qd_profile = qd_profiles[0] or DEFAULT_QD_PROFILE
    profile = archive_profile_definition(qd_profile)
    payloads = [_descriptor_payload(record) for record in archive_cells]
    mean_norm_score = mean(float(payload.get("normalized_score", 0.0)) for payload in payloads)
    unique_secondary_bins = {int(payload.get(profile.secondary_axis_bin_key, 0)) for payload in payloads}
    elite_features = [
        feature
        for feature in feature_records
        if any(cell.elite_candidate_id == feature.candidate_id for cell in archive_cells)
    ]

    hints: list[str] = []
    if mean_norm_score >= 0.8:
        hints.append("Das Archiv haelt im Mittel starke Elite-Scores; viele besetzte Zellen liegen bereits im oberen Score-Bereich.")
    else:
        hints.append("Das Archiv deckt mehrere Regionen ab, aber viele Elites liegen noch unter dem oberen Score-Bereich.")

    if len(unique_secondary_bins) >= max(3, profile.secondary_axis_bin_count // 2):
        hints.append(f"Die Elites verteilen sich ueber mehrere {profile.secondary_axis_label}-Bins; es gibt sichtbare unterschiedliche Strategieregionen.")
    else:
        hints.append(f"Die Elites konzentrieren sich auf wenige {profile.secondary_axis_label}-Bins; der Suchraum wirkt noch eher eng.")

    if qd_profile == QD_PROFILE_GENERAL_COMPACTNESS:
        hints.extend(_derive_compactness_hints(archive_cells, elite_features))
    elif qd_profile == QD_PROFILE_CURRICULUM_PROGRESS:
        hints.extend(_derive_curriculum_progress_hints(archive_cells))
    elif qd_profile == QD_PROFILE_DELAY_ROBUSTNESS:
        hints.extend(_derive_delay_robustness_hints(archive_cells))
    elif qd_profile == QD_PROFILE_RETRIEVAL_STRATEGY:
        hints.extend(_derive_retrieval_strategy_hints(archive_cells, elite_features))
    elif qd_profile == QD_PROFILE_RETRIEVAL_MECHANISM:
        hints.extend(_derive_retrieval_mechanism_hints(archive_cells, elite_features))
    else:
        hints.extend(_derive_mechanism_hints(elite_features))
    return hints


def _render_top_cells_section(
    archive_cells: Sequence[ArchiveCellRecord],
    features_by_candidate: dict[str, CandidateFeatureRecord],
    *,
    qd_profile: str,
) -> list[str]:
    profile = archive_profile_definition(qd_profile)
    top_rows: list[list[str]] = []
    for cell in sorted(archive_cells, key=lambda record: (-record.elite_score, record.archive_id))[:12]:
        payload = _descriptor_payload(cell)
        elite_feature = features_by_candidate.get(cell.elite_candidate_id)
        if qd_profile == QD_PROFILE_GENERAL_COMPACTNESS:
            top_rows.append(
                [
                    cell.descriptor_key.replace("|", " / "),
                    str(cell.delay_steps),
                    _format_stat(cell.elite_score, precision=6),
                    _format_stat(float(payload.get("normalized_score", 0.0)), precision=3),
                    str(elite_feature.enabled_conn_count if elite_feature is not None else int(payload.get("enabled_conn_count", 0))),
                    cell.elite_candidate_id,
                ]
            )
        elif qd_profile == QD_PROFILE_DELAY_ROBUSTNESS:
            top_rows.append(
                [
                    cell.descriptor_key.replace("|", " / "),
                    str(cell.delay_steps),
                    _format_stat(cell.elite_score, precision=6),
                    _format_stat(float(payload.get("normalized_score", 0.0)), precision=3),
                    _format_stat(float(payload.get("mean_score_over_delays", cell.elite_score)), precision=6),
                    _format_stat(float(payload.get("delay_score_std", 0.0)), precision=6),
                    _format_stat(elite_feature.score_delay_5 if elite_feature is not None else None, precision=6),
                    _format_stat(elite_feature.score_delay_8 if elite_feature is not None else None, precision=6),
                    cell.elite_candidate_id,
                ]
            )
        elif qd_profile == QD_PROFILE_CURRICULUM_PROGRESS:
            top_rows.append(
                [
                    cell.descriptor_key.replace("|", " / "),
                    str(payload.get("curriculum_phase", "static")),
                    str(cell.delay_steps),
                    _format_stat(cell.elite_score, precision=6),
                    _format_stat(float(payload.get("normalized_score", 0.0)), precision=3),
                    str(payload.get("active_evaluation_delays", "")),
                    _format_stat(float(payload.get("mean_score_over_delays", cell.elite_score)), precision=6),
                    _format_stat(float(payload.get("delay_score_std", 0.0)), precision=6),
                    _format_stat(elite_feature.score_delay_5 if elite_feature is not None else None, precision=6),
                    _format_stat(elite_feature.score_delay_8 if elite_feature is not None else None, precision=6),
                    cell.elite_candidate_id,
                ]
            )
        elif qd_profile == QD_PROFILE_RETRIEVAL_STRATEGY:
            top_rows.append(
                [
                    cell.descriptor_key.replace("|", " / "),
                    str(cell.delay_steps),
                    _format_stat(cell.elite_score, precision=6),
                    _format_stat(float(payload.get("normalized_score", 0.0)), precision=3),
                    _format_stat(
                        elite_feature.retrieval_score if elite_feature is not None else float(payload.get("retrieval_score", 0.0)),
                        precision=3,
                    ),
                    _format_stat(
                        elite_feature.distractor_suppression_ratio
                        if elite_feature is not None
                        else float(payload.get("distractor_suppression_ratio", 0.0)),
                        precision=3,
                    ),
                    _format_stat(elite_feature.query_accuracy if elite_feature is not None else float(payload.get("query_accuracy", 0.0)), precision=3),
                    _format_stat(elite_feature.slow_query_coupling if elite_feature is not None else float(payload.get("slow_query_coupling", 0.0)), precision=3),
                    cell.elite_candidate_id,
                ]
            )
        elif qd_profile == QD_PROFILE_RETRIEVAL_MECHANISM:
            top_rows.append(
                [
                    cell.descriptor_key.replace("|", " / "),
                    str(cell.delay_steps),
                    _format_stat(cell.elite_score, precision=6),
                    _format_stat(float(payload.get("normalized_score", 0.0)), precision=3),
                    _format_stat(
                        elite_feature.retrieval_score if elite_feature is not None else float(payload.get("retrieval_score", 0.0)),
                        precision=3,
                    ),
                    _format_stat(
                        elite_feature.slow_query_coupling if elite_feature is not None else float(payload.get("slow_query_coupling", 0.0)),
                        precision=3,
                    ),
                    _format_stat(
                        elite_feature.query_response_margin if elite_feature is not None else 0.0,
                        precision=3,
                    ),
                    _format_stat(
                        elite_feature.distractor_suppression_ratio if elite_feature is not None else float(payload.get("distractor_suppression_ratio", 0.0)),
                        precision=3,
                    ),
                    cell.elite_candidate_id,
                ]
            )
        else:
            top_rows.append(
                [
                    cell.descriptor_key.replace("|", " / "),
                    str(cell.delay_steps),
                    _format_stat(cell.elite_score, precision=6),
                    _format_stat(float(payload.get("normalized_score", 0.0)), precision=3),
                    _format_stat(float(payload.get(profile.secondary_axis_value_key, payload.get("slow_ratio_capped", 0.0))), precision=3),
                    _format_stat(elite_feature.mean_abs_fast_state if elite_feature is not None else None, precision=3),
                    _format_stat(elite_feature.mean_abs_slow_state if elite_feature is not None else None, precision=3),
                    _format_stat(elite_feature.slow_fast_contribution_ratio if elite_feature is not None else None, precision=3),
                    cell.elite_candidate_id,
                ]
            )

    if qd_profile == QD_PROFILE_GENERAL_COMPACTNESS:
        return [
            "## Top Cells",
            "",
            _render_table(
                [
                    "cell",
                    "delay",
                    "elite_score",
                    "norm_score",
                    "enabled_conn_count",
                    "candidate_id",
                ],
                top_rows,
            ),
            "",
        ]

    if qd_profile == QD_PROFILE_DELAY_ROBUSTNESS:
        return [
            "## Top Cells",
            "",
            _render_table(
                [
                    "cell",
                    "delay",
                    "elite_score",
                    "norm_score",
                    "mean_score_over_delays",
                    "delay_score_std",
                    "score_delay_5",
                    "score_delay_8",
                    "candidate_id",
                ],
                top_rows,
            ),
            "",
        ]

    if qd_profile == QD_PROFILE_CURRICULUM_PROGRESS:
        return [
            "## Top Cells",
            "",
            _render_table(
                [
                    "cell",
                    "phase",
                    "delay",
                    "elite_score",
                    "norm_score",
                    "active_evaluation_delays",
                    "mean_score_over_delays",
                    "delay_score_std",
                    "score_delay_5",
                    "score_delay_8",
                    "candidate_id",
                ],
                top_rows,
            ),
            "",
        ]

    if qd_profile == QD_PROFILE_RETRIEVAL_STRATEGY:
        return [
            "## Top Cells",
            "",
            _render_table(
                [
                    "cell",
                    "delay",
                    "elite_score",
                    "norm_score",
                    "retrieval_score",
                    "distractor_suppression_ratio",
                    "query_accuracy",
                    "slow_query_coupling",
                    "candidate_id",
                ],
                top_rows,
            ),
            "",
        ]

    if qd_profile == QD_PROFILE_RETRIEVAL_MECHANISM:
        return [
            "## Top Cells",
            "",
            _render_table(
                [
                    "cell",
                    "delay",
                    "elite_score",
                    "norm_score",
                    "retrieval_score",
                    "slow_query_coupling",
                    "query_response_margin",
                    "distractor_suppression_ratio",
                    "candidate_id",
                ],
                top_rows,
            ),
            "",
        ]

    return [
        "## Top Cells",
        "",
        _render_table(
            [
                "cell",
                "delay",
                "elite_score",
                "norm_score",
                "descriptor_slow_ratio",
                "fast_state",
                "slow_state",
                "slow_fast_ratio",
                "candidate_id",
            ],
            top_rows,
        ),
        "",
    ]


def _derive_mechanism_hints(elite_features: Sequence[CandidateFeatureRecord]) -> list[str]:
    if not elite_features:
        return []
    mean_fast = mean(feature.mean_abs_fast_state for feature in elite_features)
    mean_slow = mean(feature.mean_abs_slow_state for feature in elite_features)
    if mean_slow > mean_fast:
        return ["Die Archiv-Elites nutzen den Slow-State im Mittel staerker als den Fast-State."]
    return ["Die Archiv-Elites bleiben im Mittel Fast-State-dominant oder balanciert."]


def _derive_compactness_hints(
    archive_cells: Sequence[ArchiveCellRecord],
    elite_features: Sequence[CandidateFeatureRecord],
) -> list[str]:
    hints: list[str] = []
    if elite_features:
        mean_enabled_conn_count = mean(feature.enabled_conn_count for feature in elite_features)
        if mean_enabled_conn_count <= 4.5:
            hints.append("Die Archiv-Elites liegen im Mittel im kompakten Verbindungsbereich.")
        elif mean_enabled_conn_count >= 6.5:
            hints.append("Die Archiv-Elites liegen im Mittel im dichter vernetzten Bereich.")
        else:
            hints.append("Die Archiv-Elites verteilen sich im Mittel ueber einen mittleren Konnektivitaetsbereich.")

    strong_cells = [
        cell
        for cell in archive_cells
        if float(_descriptor_payload(cell).get("normalized_score", 0.0)) >= 0.9
    ]
    if strong_cells:
        strong_conn_bins = {int(_descriptor_payload(cell).get("conn_bin", 0)) for cell in strong_cells}
        if len(strong_conn_bins) >= 2:
            hints.append("Starke Elites erscheinen in mehreren Conn-Bins; gute Leistung haengt nicht nur an einem einzigen Kompaktheitsniveau.")
        else:
            hints.append("Starke Elites haengen noch an einem engen Conn-Bin-Bereich; der Kompaktheitsraum wirkt fuer gute Kandidaten recht fokussiert.")
    return hints


def _derive_delay_robustness_hints(
    archive_cells: Sequence[ArchiveCellRecord],
) -> list[str]:
    hints: list[str] = []
    payloads = [_descriptor_payload(record) for record in archive_cells]
    strong_cells = [
        payload
        for payload in payloads
        if float(payload.get("normalized_score", 0.0)) >= 0.9
    ]
    if strong_cells:
        strong_std_values = [float(payload.get("delay_score_std", 0.0)) for payload in strong_cells]
        mean_strong_std = mean(strong_std_values)
        if mean_strong_std <= 0.2:
            hints.append("Starke Elites bleiben ueber die Delay-Familie sehr konsistent; der Archivraum enthaelt klar robuste Kandidaten.")
        elif mean_strong_std >= 0.75:
            hints.append("Starke Elites streuen deutlich zwischen den Delays; der Archivraum enthaelt eher Spezialisten als robuste Allrounder.")
        else:
            hints.append("Starke Elites zeigen mittlere Delay-Streuung; robustere und spezialisiertere Regionen koexistieren.")
    unique_std_bins = {int(payload.get("delay_std_bin", 0)) for payload in payloads}
    if len(unique_std_bins) >= 3:
        hints.append("Das Archiv belegt mehrere Delay-Std-Bins; robuste und variablere Kandidatentypen werden getrennt sichtbar.")
    return hints


def _derive_curriculum_progress_hints(
    archive_cells: Sequence[ArchiveCellRecord],
) -> list[str]:
    hints = _derive_delay_robustness_hints(archive_cells)
    phase_groups = _group_by_curriculum_phase(archive_cells)
    if len(phase_groups) >= 2:
        phase_scores = {phase: mean(cell.elite_score for cell in group) for phase, group in phase_groups}
        phase_one_score = phase_scores.get("phase_1")
        phase_two_score = phase_scores.get("phase_2")
        if phase_one_score is not None and phase_two_score is not None:
            if phase_two_score < phase_one_score - 0.25:
                hints.append("Phase 2 liegt im Mittel noch unter Phase 1; der Curriculum-Uebergang bleibt ein echter Huerdenpunkt.")
            elif phase_two_score >= phase_one_score:
                hints.append("Phase 2 haelt das Niveau von Phase 1 oder verbessert es; robuste Kandidaten ueberstehen den Uebergang gut.")
            else:
                hints.append("Phase 2 bleibt etwas schwieriger als Phase 1, aber der Abstand wirkt nicht dramatisch.")
    return hints


def _derive_retrieval_strategy_hints(
    archive_cells: Sequence[ArchiveCellRecord],
    elite_features: Sequence[CandidateFeatureRecord],
) -> list[str]:
    hints: list[str] = []
    if elite_features:
        mean_query_accuracy = mean(feature.query_accuracy for feature in elite_features)
        mean_suppression = mean(feature.distractor_suppression_ratio for feature in elite_features)
        mean_coupling = mean(feature.slow_query_coupling for feature in elite_features)
        if mean_query_accuracy >= 0.9:
            hints.append("Die Archiv-Elites loesen die Querys meist korrekt; Retrieval-Leistung ist im Spitzenfeld hoch.")
        elif mean_query_accuracy <= 0.6:
            hints.append("Viele Archiv-Elites bleiben bei den Querys noch fehlerhaft; der Retrieval-Raum ist noch nicht voll ausgereizt.")
        if mean_suppression >= 2.0:
            hints.append("Die Elites unterdruecken Distraktoren deutlich; Filtering und Retention wirken gemeinsam stabil.")
        elif mean_suppression <= 1.0:
            hints.append("Distraktorunterdrueckung bleibt schwach; gute Retention wird noch oft durch Leakage verwischt.")
        if mean_coupling >= 1.0:
            hints.append("Der slow-state-Zweig koppelt sichtbar in die Query-Phase ein; Retrieval wirkt mechanistisch getragen statt rein reaktiv.")
    return hints


def _derive_retrieval_mechanism_hints(
    archive_cells: Sequence[ArchiveCellRecord],
    elite_features: Sequence[CandidateFeatureRecord],
) -> list[str]:
    hints: list[str] = []
    if not elite_features:
        return hints
    mean_coupling = mean(feature.slow_query_coupling for feature in elite_features)
    mean_margin = mean(feature.query_response_margin for feature in elite_features)
    mean_suppression = mean(feature.distractor_suppression_ratio for feature in elite_features)
    if mean_coupling >= 1.2:
        hints.append("Die Elites wirken slow-dominant in der Query-Phase; der Abrufpfad nutzt den langsameren Zustand sichtbar.")
    elif mean_coupling <= 0.5:
        hints.append("Die Elites wirken eher fast-reaktiv; der Query-Pfad bleibt mechanistisch nahe an der schnellen Spur.")
    else:
        hints.append("Die Elites liegen zwischen fast-reaktivem und slow-dominiertem Retrieval; der Archivraum zeigt gemischte Abrufmodi.")
    if mean_suppression >= 2.0 and mean_margin <= 0.2:
        hints.append("Hohe Distraktorunterdrueckung trifft auf enge Antwortmargen; Filtering ist stark, die Value-Selektion aber noch nicht ganz stabil.")
    elif mean_suppression >= 2.0 and mean_margin > 0.2:
        hints.append("Filtering und Antwortmargen sind gemeinsam stabil; der Retrieval-Mechanismus wirkt balanciert und belastbar.")
    elif mean_suppression < 1.0:
        hints.append("Die Distraktorkonkurrenz bleibt hoch; gute Abrufwerte wirken noch anfaellig fuer stoerenden Kontext.")
    return hints


def _group_by_delay(records: Sequence[ArchiveCellRecord]) -> list[tuple[int, list[ArchiveCellRecord]]]:
    grouped: dict[int, list[ArchiveCellRecord]] = {}
    for record in records:
        grouped.setdefault(int(record.delay_steps), []).append(record)
    return sorted(grouped.items(), key=lambda item: item[0])


def _group_by_curriculum_phase(records: Sequence[ArchiveCellRecord]) -> list[tuple[str, list[ArchiveCellRecord]]]:
    grouped: dict[str, list[ArchiveCellRecord]] = {}
    for record in records:
        phase = _curriculum_phase_for_record(record)
        grouped.setdefault(phase, []).append(record)
    return sorted(grouped.items(), key=lambda item: item[0])


def _event_type_counts(records: Sequence[ArchiveEventRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.event_type] = counts.get(record.event_type, 0) + 1
    return counts


def _descriptor_payload(record: ArchiveCellRecord) -> dict[str, object]:
    return json.loads(record.descriptor_values_json)


def _curriculum_phase_for_record(record: ArchiveCellRecord) -> str:
    return str(_descriptor_payload(record).get("curriculum_phase", "static"))


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _format_stat(value: float | None, *, precision: int) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"
