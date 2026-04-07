from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Sequence

from db.models import CandidateFeatureRecord
from utils.serialization import stable_json_dumps


SUMMARY_FEATURES = [
    "curriculum_switch_generation",
    "score_current_phase",
    "score_delay_3",
    "score_delay_5",
    "score_delay_8",
    "mean_score_over_delays",
    "delay_score_std",
    "delay_score_range",
    "query_accuracy",
    "retrieval_score",
    "mean_query_distance",
    "distractor_load",
    "retrieval_margin",
    "retrieval_confusion_rate",
    "relevant_token_retention",
    "query_response_margin",
    "distractor_suppression_ratio",
    "correct_key_selected",
    "correct_value_selected",
    "query_key_match_score",
    "value_margin",
    "distractor_competition_score",
    "mean_eta",
    "mean_plastic_d",
    "plastic_d_at_lower_bound_fraction",
    "plastic_d_at_zero_fraction",
    "mean_abs_delta_w",
    "max_abs_delta_w",
    "mean_abs_decay_term",
    "max_abs_decay_term",
    "decay_effect_ratio",
    "decay_near_zero_fraction",
    "clamp_hit_rate",
    "plasticity_active_fraction",
    "mean_abs_fast_state",
    "mean_abs_slow_state",
    "slow_fast_contribution_ratio",
    "mean_abs_fast_state_during_store",
    "mean_abs_slow_state_during_store",
    "mean_abs_fast_state_during_query",
    "mean_abs_slow_state_during_query",
    "mean_abs_fast_state_during_distractor",
    "mean_abs_slow_state_during_distractor",
    "slow_query_coupling",
    "store_query_state_gap",
    "slow_fast_retrieval_ratio",
    "retrieval_state_alignment",
    "node_count",
    "enabled_conn_count",
]


def load_feature_records_from_jsonl(path: Path) -> list[CandidateFeatureRecord]:
    records: list[CandidateFeatureRecord] = []
    if not path.exists():
        raise FileNotFoundError(f"Feature export not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            records.append(CandidateFeatureRecord(**payload))
    return records


def write_feature_records_jsonl(path: Path, records: Sequence[CandidateFeatureRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json_dumps(asdict(record)))
            handle.write("\n")


def filter_feature_records(
    records: Iterable[CandidateFeatureRecord],
    *,
    benchmark_label: str,
    task_name: str | None = None,
    variant: str | None = None,
    delay_steps: int | None = None,
    curriculum_phase: str | None = None,
) -> list[CandidateFeatureRecord]:
    filtered = [record for record in records if record.benchmark_label == benchmark_label]
    if task_name is not None:
        filtered = [record for record in filtered if record.task_name == task_name]
    if variant is not None:
        filtered = [record for record in filtered if record.variant == variant]
    if delay_steps is not None:
        filtered = [record for record in filtered if record.delay_steps == delay_steps]
    if curriculum_phase is not None:
        filtered = [record for record in filtered if record.curriculum_phase == curriculum_phase]
    filtered.sort(key=lambda record: (record.task_name, record.delay_steps, record.variant, record.seed, record.generation, record.candidate_id))
    return filtered


def render_search_space_report(
    records: Sequence[CandidateFeatureRecord],
    *,
    curriculum_phase_filter: str | None = None,
) -> str:
    if not records:
        return "No candidate features found for the requested filter."

    summary_title = "## Search Space Summary"
    if curriculum_phase_filter is not None:
        summary_title = f"## Search Space Summary ({curriculum_phase_filter})"

    lines = [
        summary_title,
        "",
        _render_table(
            ["candidates", "hof_candidates", "success_rate", "mean_final_max_score"],
            [
                [
                    str(len(records)),
                    str(sum(1 for record in records if record.hof_flag)),
                    f"{sum(1 for record in records if record.success) / len(records):.3f}",
                    _format_stat(_safe_mean([record.final_max_score for record in records])),
                ]
            ],
        ),
        "",
    ]
    if any(record.curriculum_enabled for record in records):
        lines.extend(
            [
                "## Curriculum Metadata",
                "",
                _render_curriculum_metadata_table(records),
                "",
            ]
        )
    if len({record.variant for record in records}) > 1:
        lines.extend(
            [
                "## By Variant",
                "",
                _render_group_summary_table(records, group_by="variant"),
                "",
            ]
        )
    if len({record.delay_steps for record in records}) > 1:
        lines.extend(
            [
                "## By Delay",
                "",
                _render_group_summary_table(records, group_by="delay_steps"),
                "",
            ]
        )
    if len({record.curriculum_phase for record in records}) > 1:
        lines.extend(
            [
                "## By Curriculum Phase",
                "",
                _render_group_summary_table(records, group_by="curriculum_phase"),
                "",
            ]
        )
    if len({record.curriculum_switch_generation for record in records if record.curriculum_enabled}) > 1:
        lines.extend(
            [
                "## By Curriculum Switch Generation",
                "",
                _render_group_summary_table(records, group_by="curriculum_switch_generation"),
                "",
            ]
        )
    lines.extend(
        [
        "## Feature Stats",
        "",
        _render_table(
            ["feature", "mean", "std"],
            [[feature, _format_stat(_safe_mean(_values(records, feature))), _format_stat(_safe_std(_values(records, feature)))] for feature in SUMMARY_FEATURES],
        ),
        "",
        "## HOF vs Non-HOF",
        "",
        _render_group_comparison_table(
            left_label="hof",
            left_records=[record for record in records if record.hof_flag],
            right_label="non_hof",
            right_records=[record for record in records if not record.hof_flag],
        ),
        "",
        "## Success vs Failure",
        "",
        _render_group_comparison_table(
            left_label="success",
            left_records=[record for record in records if record.success],
            right_label="failure",
            right_records=[record for record in records if not record.success],
        ),
        "",
        "## Hints",
        "",
        ]
    )
    lines.extend(f"- {hint}" for hint in derive_search_space_hints(records))
    return "\n".join(lines).rstrip() + "\n"


def derive_search_space_hints(records: Sequence[CandidateFeatureRecord]) -> list[str]:
    hints: list[str] = []
    if not records:
        return ["No rows available."]

    mean_clamp_hit_rate = _safe_mean([record.clamp_hit_rate for record in records])
    mean_abs_delta = _safe_mean([record.mean_abs_delta_w for record in records])
    mean_max_delta = _safe_mean([record.max_abs_delta_w for record in records])
    mean_abs_decay = _safe_mean([record.mean_abs_decay_term for record in records])
    mean_decay_ratio = _safe_mean([record.decay_effect_ratio for record in records])
    mean_decay_near_zero = _safe_mean([record.decay_near_zero_fraction for record in records])
    mean_plastic_d_lower = _safe_mean([record.plastic_d_at_lower_bound_fraction for record in records])
    mean_plastic_d_zero = _safe_mean([record.plastic_d_at_zero_fraction for record in records])
    mean_fast_state = _safe_mean([record.mean_abs_fast_state for record in records])
    mean_slow_state = _safe_mean([record.mean_abs_slow_state for record in records])
    mean_slow_fast_ratio = _safe_mean([record.slow_fast_contribution_ratio for record in records])
    mean_query_accuracy = _safe_mean([record.query_accuracy for record in records])
    mean_retrieval_score = _safe_mean([record.retrieval_score for record in records])
    mean_query_distance = _safe_mean([record.mean_query_distance for record in records])
    mean_distractor_load = _safe_mean([record.distractor_load for record in records])
    mean_retrieval_margin = _safe_mean([record.retrieval_margin for record in records])
    mean_retrieval_confusion_rate = _safe_mean([record.retrieval_confusion_rate for record in records])
    mean_distractor_suppression_ratio = _safe_mean([record.distractor_suppression_ratio for record in records])
    mean_correct_key_selected = _safe_mean([record.correct_key_selected for record in records])
    mean_correct_value_selected = _safe_mean([record.correct_value_selected for record in records])
    mean_query_key_match_score = _safe_mean([record.query_key_match_score for record in records])
    mean_value_margin = _safe_mean([record.value_margin for record in records])
    mean_distractor_competition_score = _safe_mean([record.distractor_competition_score for record in records])
    mean_slow_query_coupling = _safe_mean([record.slow_query_coupling for record in records])
    mean_retrieval_state_alignment = _safe_mean([record.retrieval_state_alignment for record in records])
    mean_store_vs_distractor_write_gap = _safe_mean([record.store_vs_distractor_write_gap for record in records])
    mean_query_value_read_strength = _safe_mean([record.query_value_read_strength for record in records])
    mean_readout_selectivity = _safe_mean([record.readout_selectivity for record in records])
    mean_query_key_alignment_v11 = _safe_mean([record.query_key_alignment for record in records])
    mean_score_over_delays = _safe_mean([record.mean_score_over_delays for record in records])
    mean_delay_score_std = _safe_mean([record.delay_score_std for record in records])
    mean_delay_score_range = _safe_mean([record.delay_score_range for record in records])
    has_multi_delay_signal = any(
        sum(
            1
            for value in (record.score_delay_3, record.score_delay_5, record.score_delay_8)
            if value > 0.0
        )
        >= 2
        for record in records
    )
    phase_one_records = [record for record in records if record.curriculum_phase == "phase_1"]
    phase_two_records = [record for record in records if record.curriculum_phase == "phase_2"]

    if mean_plastic_d_zero >= 0.5:
        hints.append("plastic_d haeuft sich stark bei 0.0; der AD-Suchraum scheint oft in wenig aktiven Decay-Regeln zu landen.")
    elif mean_plastic_d_lower >= 0.5:
        hints.append("plastic_d klebt haeufig an der Untergrenze; die Suche wirkt stark in Richtung aggressiven Decay gedrueckt.")
    else:
        hints.append("plastic_d zeigt keine dominante Randhaeufung; der Suchraum wirkt eher verteilt als komplett festgeklemmt.")

    if mean_clamp_hit_rate >= 0.25:
        hints.append("clamp_hit_rate ist relativ hoch; viele Kandidaten fahren ihre Plastizitaetsdynamik sichtbar gegen die harte Clamp-Grenze.")
    elif mean_clamp_hit_rate <= 0.02:
        hints.append("clamp_hit_rate ist sehr niedrig; die Plastizitaet kollidiert selten mit der Clamp und wirkt eher untersteuert.")
    else:
        hints.append("clamp_hit_rate liegt im mittleren Bereich; die Clamp greift, aber dominiert nicht jede Episode.")

    if mean_abs_delta <= 0.05 and mean_max_delta <= 0.1:
        hints.append("delta_w bleibt meist klein; die plastische Laufzeitspur wirkt eher schwach oder vorsichtig.")
    elif mean_max_delta >= 0.4:
        hints.append("max_abs_delta_w liegt oft nahe an der 0.5-Grenze; einzelne Kandidaten nutzen fast den ganzen Schreibraum aus.")
    else:
        hints.append("delta_w nutzt einen mittleren Bereich; die Laufzeitspur wirkt weder komplett tot noch permanent saturiert.")

    if mean_abs_decay <= 0.01 or mean_decay_ratio <= 0.1:
        hints.append("Der Decay-Beitrag wirkt meist schwach; D greift oft nur gering in die Laufzeitdynamik ein.")
    elif mean_decay_ratio >= 0.5:
        hints.append("Decay traegt substanziell zur Laufzeitdynamik bei; D ist kein reiner Nebenparameter.")
    else:
        hints.append("Decay wirkt messbar, aber nicht dominant gegenueber dem plastischen Schreibterm.")

    if mean_decay_near_zero >= 0.75:
        hints.append("Der Decay-Term ist in vielen Updates praktisch null; die aktuelle Parametrisierung macht D oft inaktiv.")

    if has_multi_delay_signal and mean_score_over_delays is not None and mean_delay_score_std is not None:
        if mean_score_over_delays >= 3.5 and mean_delay_score_std <= 0.25:
            hints.append("Die Kandidaten wirken im Mittel robust ueber die Delay-Familie: hoher mittlerer Score bei niedriger Delay-Varianz.")
        elif mean_delay_score_std >= 0.75 or mean_delay_score_range >= 1.0:
            hints.append("Die Delay-Streuung ist sichtbar; der Suchraum enthaelt eher Spezialisten als gleichmaessig starke Allrounder.")
        else:
            hints.append("Die Delay-Streuung liegt im mittleren Bereich; es gibt weder rein robuste noch extrem spezialisierte Muster.")

    if any(record.task_name == "key_value_memory" for record in records):
        if mean_query_accuracy >= 0.9 and mean_retrieval_confusion_rate <= 0.1:
            hints.append("Die Retrieval-Task wirkt im Mittel gut geloest: hohe Query-Genauigkeit bei niedriger Verwechslungsrate.")
        elif mean_query_accuracy <= 0.6:
            hints.append("Die Retrieval-Task ist noch fordernd; viele Kandidaten verlieren relevante Information vor der Query.")
        else:
            hints.append("Die Retrieval-Task trennt bereits zwischen brauchbarem und fragilen Kontextgedaechtnis.")

        if mean_distractor_suppression_ratio >= 2.0:
            hints.append("Distraktoren werden im Mittel gut unterdrueckt; relevante Information setzt sich klarer gegen Rauschen durch.")
        elif mean_distractor_suppression_ratio <= 1.0:
            hints.append("Distraktorunterdrueckung wirkt noch schwach; irrelevante Token lecken sichtbar in den Antwortpfad.")

        if mean_correct_key_selected >= mean_correct_value_selected + 0.1:
            hints.append("Viele Kandidaten scheinen den richtigen Kontext zu isolieren, verlieren aber noch Praezision bei der eigentlichen Value-Selektion.")
        elif mean_correct_key_selected <= 0.5:
            hints.append("Schon die Key-Selektion bleibt oft unsicher; der Query-Pfad findet relevante Stores noch nicht verlaesslich.")

        if mean_query_key_match_score <= 0.0:
            hints.append("Die Ziel-Store-Spur gewinnt gegen konkurrierende Stores oft nicht klar; falsche Kontexte bleiben bei der Query konkurrenzfaehig.")

        if mean_distractor_competition_score >= 0.5:
            hints.append("Distraktoren konkurrieren sichtbar mit dem Zielsignal; der Retrieval-Pfad ist noch anfaellig fuer falsche Attraktoren.")

        if mean_store_vs_distractor_write_gap <= 0.05:
            hints.append("Schreibpfad trennt Store und Distraktor noch nicht sauber; das Write-Gate bleibt zu undifferenziert.")
        elif mean_store_vs_distractor_write_gap >= 0.2:
            hints.append("Store-Schritte schreiben deutlich staerker als Distraktoren; der Write-Pfad zeigt gute Selektivitaet.")

        if mean_query_key_alignment_v11 <= 0.25:
            hints.append("Query-Match bleibt schwach; die Query koppelt nur locker an den relevanten Key-Zustand.")

        if mean_query_value_read_strength <= 0.2:
            hints.append("Value-Readout ist vorhanden, aber zu instabil; Query-getriebene Auslesestaerke bleibt niedrig.")

        if mean_query_key_alignment_v11 >= 0.5 and mean_query_value_read_strength <= 0.3:
            hints.append("Key/Value-Trennung ist messbar, aber noch nicht funktional ausreichend fuer robustes Value-Retrieval.")

        if mean_readout_selectivity >= 0.25:
            hints.append("Readout bleibt zwischen Query und Distraktor klar getrennt; der Abrufpfad wirkt selektiver.")

        if mean_retrieval_state_alignment >= 0.75:
            hints.append("Store- und Query-Zustand bleiben gut ausgerichtet; relevante Information wird intern relativ konsistent weitergetragen.")
        elif mean_retrieval_state_alignment <= 0.4:
            hints.append("Zwischen Store- und Query-Zustand geht viel interne Ausrichtung verloren; Retention und Abruf sind noch locker gekoppelt.")

        if mean_query_distance >= 5.0 and mean_retrieval_score >= 0.75:
            hints.append("Die Kandidaten halten relevante Information auch ueber laengere Store-Query-Distanzen erstaunlich stabil.")

        if mean_slow_query_coupling >= 1.0:
            hints.append("Der slow-state-Zweig koppelt sichtbar in die Query-Phase ein; Retrieval wirkt nicht rein fast-state-getrieben.")
        else:
            hints.append("Der Query-Pfad bleibt eher fast-state-nah; slow-state-Retrieval ist noch nicht dominant ausgepraegt.")

    if phase_one_records and phase_two_records:
        phase_one_score = _safe_mean([record.score_current_phase for record in phase_one_records])
        phase_two_score = _safe_mean([record.score_current_phase for record in phase_two_records])
        if phase_one_score is not None and phase_two_score is not None:
            if phase_two_score < phase_one_score - 0.25:
                hints.append("Beim Wechsel in Phase 2 faellt der mittlere Phasenscore sichtbar ab; der Curriculum-Schritt ist noch fordernd.")
            elif phase_two_score > phase_one_score + 0.1:
                hints.append("Die Kandidaten halten oder steigern ihren mittleren Phasenscore nach dem Wechsel in Phase 2.")
            else:
                hints.append("Der Uebergang von Phase 1 zu Phase 2 wirkt im Mittel relativ glatt.")

    if mean_fast_state is not None and mean_slow_state is not None and mean_slow_fast_ratio is not None:
        if mean_slow_state <= 0.01 and mean_slow_fast_ratio <= 0.05:
            hints.append("Die langsame Spur bleibt meist klein; der slow-state-Zweig wirkt oft kaum genutzt.")
        elif mean_slow_fast_ratio >= 1.0:
            hints.append("Der slow-state-Zweig traegt im Mittel mindestens so stark bei wie der fast-state-Zweig.")
        else:
            hints.append("Fast- und Slow-State liegen beide im Spiel; der slow-state-Zweig wirkt messbar, aber nicht dominant.")

    hof_records = [record for record in records if record.hof_flag]
    non_hof_records = [record for record in records if not record.hof_flag]
    if hof_records and non_hof_records:
        hof_eta = _safe_mean([record.mean_eta for record in hof_records])
        non_hof_eta = _safe_mean([record.mean_eta for record in non_hof_records])
        if hof_eta > non_hof_eta + 0.02:
            hints.append("HOF-Kandidaten tragen im Mittel hoehere eta-Werte als Nicht-HOF-Kandidaten.")
        elif hof_eta < non_hof_eta - 0.02:
            hints.append("HOF-Kandidaten liegen im Mittel bei konservativerem eta als Nicht-HOF-Kandidaten.")
        else:
            hints.append("HOF- und Nicht-HOF-Kandidaten liegen bei mean_eta nah beieinander.")

    return hints


def _render_group_comparison_table(
    *,
    left_label: str,
    left_records: Sequence[CandidateFeatureRecord],
    right_label: str,
    right_records: Sequence[CandidateFeatureRecord],
) -> str:
    rows = [
        [
            "count",
            str(len(left_records)),
            "0.000",
            str(len(right_records)),
            "0.000",
        ]
    ]
    for feature in ["final_max_score", *SUMMARY_FEATURES]:
        left_values = _values(left_records, feature)
        right_values = _values(right_records, feature)
        rows.append(
            [
                feature,
                _format_stat(_safe_mean(left_values)),
                _format_stat(_safe_std(left_values)),
                _format_stat(_safe_mean(right_values)),
                _format_stat(_safe_std(right_values)),
            ]
        )
    return _render_table(
        ["feature", f"{left_label}_mean", f"{left_label}_std", f"{right_label}_mean", f"{right_label}_std"],
        rows,
    )


def _render_group_summary_table(records: Sequence[CandidateFeatureRecord], *, group_by: str) -> str:
    grouped: dict[str, list[CandidateFeatureRecord]] = {}
    for record in records:
        raw_key = getattr(record, group_by)
        key = str(raw_key)
        grouped.setdefault(key, []).append(record)
    rows: list[list[str]] = []
    for key, group_records in sorted(grouped.items(), key=lambda item: item[0]):
        rows.append(
            [
                key,
                str(len(group_records)),
                str(sum(1 for record in group_records if record.hof_flag)),
                _format_stat(sum(1 for record in group_records if record.success) / len(group_records)),
                _format_stat(_safe_mean([record.final_max_score for record in group_records])),
                _format_stat(_safe_mean([record.score_current_phase for record in group_records])),
                _format_stat(_safe_mean([record.mean_eta for record in group_records])),
                _format_stat(_safe_mean([record.mean_plastic_d for record in group_records])),
                _format_stat(_safe_mean([record.mean_score_over_delays for record in group_records])),
                _format_stat(_safe_mean([record.delay_score_std for record in group_records])),
                _format_stat(_safe_mean([record.mean_abs_delta_w for record in group_records])),
                _format_stat(_safe_mean([record.mean_abs_decay_term for record in group_records])),
                _format_stat(_safe_mean([record.decay_effect_ratio for record in group_records])),
                _format_stat(_safe_mean([record.clamp_hit_rate for record in group_records])),
                _format_stat(_safe_mean([record.plasticity_active_fraction for record in group_records])),
                _format_stat(_safe_mean([record.mean_abs_fast_state for record in group_records])),
                _format_stat(_safe_mean([record.mean_abs_slow_state for record in group_records])),
                _format_stat(_safe_mean([record.slow_fast_contribution_ratio for record in group_records])),
            ]
        )
    if group_by == "variant":
        label = "variant"
    elif group_by == "delay_steps":
        label = "delay"
    elif group_by == "curriculum_switch_generation":
        label = "curriculum_switch_generation"
    else:
        label = "curriculum_phase"
    return _render_table(
        [
            label,
            "candidates",
            "hof_candidates",
            "success_rate",
            "mean_final_max_score",
            "mean_score_current_phase",
            "mean_eta",
            "mean_plastic_d",
            "mean_score_over_delays",
            "mean_delay_score_std",
            "mean_abs_delta_w",
            "mean_abs_decay_term",
            "mean_decay_effect_ratio",
            "mean_clamp_hit_rate",
            "mean_plasticity_active_fraction",
            "mean_abs_fast_state",
            "mean_abs_slow_state",
            "mean_slow_fast_ratio",
        ],
        rows,
    )


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _render_curriculum_metadata_table(records: Sequence[CandidateFeatureRecord]) -> str:
    phase_1_labels = sorted({record.curriculum_phase_1_delays for record in records if record.curriculum_enabled})
    phase_2_labels = sorted({record.curriculum_phase_2_delays for record in records if record.curriculum_enabled})
    switch_generations = sorted({record.curriculum_switch_generation for record in records if record.curriculum_enabled})
    return _render_table(
        [
            "curriculum_enabled",
            "phase_1_delays",
            "phase_2_delays",
            "switch_generation",
        ],
        [
            [
                "true",
                ", ".join(phase_1_labels),
                ", ".join(phase_2_labels),
                ", ".join(str(value) for value in switch_generations),
            ]
        ],
    )


def _values(records: Sequence[CandidateFeatureRecord], feature: str) -> list[float]:
    return [float(getattr(record, feature)) for record in records]


def _safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _safe_std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(pstdev(values))


def _format_stat(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.6f}"
