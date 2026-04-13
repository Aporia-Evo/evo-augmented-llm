"""Cross-candidate retrieval-trace sweep diagnostics.

This module orchestrates repeated ``run_retrieval_trace`` episodes across top
candidates and aggregates mechanistic failure patterns. It is purely
diagnostic: no scoring, math, task, or executor behavior is changed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

from analysis.retrieval_trace import (
    FAILURE_NONE,
    FAILURE_OUTPUT_DECODING,
    FAILURE_QUERY_ALIGNMENT,
    FAILURE_READOUT_COLLAPSE,
    FAILURE_WRITE_SELECTIVITY,
    run_retrieval_trace,
)
from db.models import CandidateFeatureRecord
from evolve.genome_codec import GenomeModel, genome_model_from_blob
from tasks.key_value_memory import KeyValueMemoryTask
from analysis.search_space import filter_feature_records, load_feature_records_from_jsonl


FAILURE_MIXED = "mixed_failures"
FAILURE_MOSTLY_CORRECT_SPARSE = "mostly_correct_but_sparse"

FINAL_VERDICT_WRITE = "dominant-write-selectivity-failure"
FINAL_VERDICT_QUERY = "dominant-query-alignment-failure"
FINAL_VERDICT_READOUT = "dominant-readout-collapse"
FINAL_VERDICT_OUTPUT = "dominant-output-decoding-failure"
FINAL_VERDICT_MIXED = "mixed-failure-regime"
FINAL_VERDICT_INCONSISTENT = "mostly-correct-but-inconsistent"

ALLOWED_FINAL_VERDICTS = {
    FINAL_VERDICT_WRITE,
    FINAL_VERDICT_QUERY,
    FINAL_VERDICT_READOUT,
    FINAL_VERDICT_OUTPUT,
    FINAL_VERDICT_MIXED,
    FINAL_VERDICT_INCONSISTENT,
}


@dataclass(frozen=True)
class EpisodeTraceSummary:
    sample_index: int
    failure_class: str
    query_key_match_score: float
    value_margin: float
    correct_value_selected: bool
    predicted_value_id: int
    target_value_id: int
    query_memory_alignment: float | None
    readout_selectivity: float | None


@dataclass(frozen=True)
class CandidateTraceSummary:
    candidate_id: str
    run_id: str
    final_max_score: float | None
    episodes: tuple[EpisodeTraceSummary, ...]
    dominant_failure_class: str
    consistency_fraction: float


@dataclass(frozen=True)
class RetrievalTraceSweepResult:
    benchmark_label: str
    task_name: str
    variant: str
    candidates_traced: tuple[CandidateTraceSummary, ...]
    episodes_per_candidate: int
    failure_frequency: dict[str, int]
    overall_correct_fraction: float
    cross_candidate_consistency: float
    final_verdict: str
    next_step_hint: str
    candidate_source: str = "unknown"


@dataclass(frozen=True)
class SweepCandidate:
    candidate_id: str
    run_id: str
    final_max_score: float | None
    success: bool | None = None
    task_name: str | None = None
    variant: str | None = None
    delay_steps: int = 8
    seed: int | None = None


@dataclass(frozen=True)
class LocalSweepInputs:
    candidates: tuple[SweepCandidate, ...]
    genomes_by_candidate: dict[str, GenomeModel]
    source_used: str
    checked_files: tuple[str, ...]


def _episode_indices(*, profile: str, delay_steps: int, limit: int | None) -> list[int]:
    task = KeyValueMemoryTask.create(delay_steps=delay_steps, profile=profile)
    indices = list(range(len(task.input_sequences)))
    if limit is not None and limit > 0:
        return indices[: min(len(indices), int(limit))]
    return indices


def _mean_query_metric(trace: list[dict[str, object]], key: str) -> float | None:
    vals = [float(r.get(key, 0.0)) for r in trace if str(r.get("step_role", "")) == "query"]
    if not vals:
        return None
    return float(mean(vals))


def _value_margin_from_trace_output(*, raw_output: np.ndarray, value_levels: tuple[float, ...], query_step_index: int) -> float:
    bounded = np.clip((raw_output[:, 0] + 1.0) / 2.0, 0.0, 1.0)
    query_value = float(bounded[query_step_index])
    distances = sorted(float(abs(query_value - level)) for level in value_levels)
    if len(distances) < 2:
        return 0.0
    return float(distances[1] - distances[0])


def _candidate_dominant_failure(episodes: list[EpisodeTraceSummary]) -> tuple[str, float]:
    if not episodes:
        return FAILURE_MIXED, 0.0
    counts = Counter(ep.failure_class for ep in episodes)
    dominant, count = counts.most_common(1)[0]
    return str(dominant), float(count / len(episodes))


def _pick_final_verdict(*, failure_frequency: Counter[str], overall_correct_fraction: float, feature_success_fraction: float, cross_candidate_consistency: float) -> str:
    if overall_correct_fraction >= 0.60 and feature_success_fraction < 0.50:
        return FINAL_VERDICT_INCONSISTENT

    dominant_failure, dominant_count = failure_frequency.most_common(1)[0]
    total = sum(failure_frequency.values())
    dominant_fraction = dominant_count / max(1, total)

    if dominant_failure == FAILURE_NONE and cross_candidate_consistency < 0.60:
        return FINAL_VERDICT_INCONSISTENT

    if dominant_fraction < 0.55 or cross_candidate_consistency < 0.50:
        return FINAL_VERDICT_MIXED

    if dominant_failure == FAILURE_WRITE_SELECTIVITY:
        return FINAL_VERDICT_WRITE
    if dominant_failure == FAILURE_QUERY_ALIGNMENT:
        return FINAL_VERDICT_QUERY
    if dominant_failure == FAILURE_READOUT_COLLAPSE:
        return FINAL_VERDICT_READOUT
    if dominant_failure == FAILURE_OUTPUT_DECODING:
        return FINAL_VERDICT_OUTPUT
    return FINAL_VERDICT_MIXED


def _next_step_hint(final_verdict: str) -> str:
    if final_verdict == FINAL_VERDICT_WRITE:
        return "Prioritize interventions that improve store-vs-distractor write selectivity."
    if final_verdict == FINAL_VERDICT_QUERY:
        return "Prioritize interventions that improve query-to-memory alignment robustness."
    if final_verdict == FINAL_VERDICT_READOUT:
        return "Prioritize interventions that sharpen readout selectivity at query time."
    if final_verdict == FINAL_VERDICT_OUTPUT:
        return "Prioritize interventions in the post-readout output decoding path."
    if final_verdict == FINAL_VERDICT_INCONSISTENT:
        return "Prioritize interventions that stabilize retrieval policy consistency across episodes."
    return "Prioritize a broad diagnostic intervention to separate mixed failure regimes."


def _coerce_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object, default: int = 8) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _candidate_from_row(row: dict[str, object], *, task_name: str, variant: str) -> SweepCandidate | None:
    candidate_id = str(
        row.get("candidate_id")
        or row.get("best_candidate_id")
        or ""
    ).strip()
    if not candidate_id:
        return None
    run_id = str(row.get("run_id") or "unknown-run").strip() or "unknown-run"
    score = _coerce_float(
        row.get("final_max_score")
        if row.get("final_max_score") is not None
        else row.get("best_score", row.get("score"))
    )
    delay_steps = _coerce_int(row.get("delay_steps"), default=8)
    row_task = str(row.get("task_name") or task_name or "").strip() or None
    row_variant = str(row.get("variant") or variant or "").strip() or None
    if task_name and row_task and row_task != task_name:
        return None
    if variant and row_variant and row_variant != variant:
        return None
    success_raw = row.get("success")
    success = None if success_raw is None else bool(success_raw)
    seed = _coerce_int(row.get("seed"), default=-1)
    return SweepCandidate(
        candidate_id=candidate_id,
        run_id=run_id,
        final_max_score=score,
        success=success,
        task_name=row_task,
        variant=row_variant,
        delay_steps=delay_steps,
        seed=None if seed < 0 else seed,
    )


def resolve_local_sweep_inputs(
    *,
    output_dir: str | Path,
    benchmark_label: str,
    task_name: str,
    variant: str,
) -> LocalSweepInputs:
    """Resolve local candidates/genomes via artifact fallback hierarchy."""
    output_path = Path(output_dir)
    feature_path = output_path / f"{benchmark_label}.candidate-features.jsonl"
    bench_jsonl_path = output_path / f"{benchmark_label}.jsonl"
    genome_path = output_path / f"{benchmark_label}.candidate-genomes.jsonl"
    bench_csv_path = output_path / f"{benchmark_label}.csv"
    checked_paths = (feature_path, bench_jsonl_path, genome_path, bench_csv_path)

    genomes_by_candidate: dict[str, GenomeModel] = {}
    if genome_path.exists():
        for line in genome_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = str(row.get("candidate_id", "")).strip()
            blob = str(row.get("genome_blob", "")).strip()
            if candidate_id and blob:
                genomes_by_candidate[candidate_id] = genome_model_from_blob(blob)

    candidates: list[SweepCandidate] = []
    source_used = "none"

    if feature_path.exists():
        feature_records = load_feature_records_from_jsonl(feature_path)
        filtered = filter_feature_records(
            feature_records,
            benchmark_label=benchmark_label,
            task_name=task_name,
            variant=variant,
        )
        candidates = [
            SweepCandidate(
                candidate_id=record.candidate_id,
                run_id=record.run_id,
                final_max_score=float(record.final_max_score),
                success=bool(record.success),
                task_name=record.task_name,
                variant=record.variant,
                delay_steps=int(record.delay_steps),
                seed=int(record.seed),
            )
            for record in filtered
        ]
        source_used = "candidate-features"

    if not candidates and bench_jsonl_path.exists():
        parsed: list[SweepCandidate] = []
        for line in bench_jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            parsed_candidate = _candidate_from_row(row, task_name=task_name, variant=variant)
            if parsed_candidate is not None:
                parsed.append(parsed_candidate)
        candidates = parsed
        source_used = "benchmark-jsonl"

    if not candidates and bench_csv_path.exists():
        parsed_csv: list[SweepCandidate] = []
        with bench_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed_candidate = _candidate_from_row(dict(row), task_name=task_name, variant=variant)
                if parsed_candidate is not None:
                    parsed_csv.append(parsed_candidate)
        candidates = parsed_csv
        source_used = "benchmark-csv"

    if not candidates and genomes_by_candidate:
        candidates = [
            SweepCandidate(
                candidate_id=candidate_id,
                run_id="unknown-run",
                final_max_score=None,
                success=None,
                task_name=task_name or None,
                variant=variant or None,
            )
            for candidate_id in sorted(genomes_by_candidate.keys())
        ]
        source_used = "candidate-genomes"

    if not candidates or not genomes_by_candidate:
        checked_txt = ", ".join(str(path) for path in checked_paths)
        raise FileNotFoundError(
            "Insufficient local artifacts for retrieval trace sweep. "
            f"Checked: {checked_txt}. "
            "Need at least candidate genomes and one candidate discovery source."
        )

    deduped: dict[str, SweepCandidate] = {}
    for candidate in candidates:
        if candidate.candidate_id not in deduped:
            deduped[candidate.candidate_id] = candidate
    return LocalSweepInputs(
        candidates=tuple(deduped.values()),
        genomes_by_candidate=genomes_by_candidate,
        source_used=source_used,
        checked_files=tuple(str(path) for path in checked_paths),
    )


def run_retrieval_trace_sweep(
    *,
    benchmark_label: str,
    task_name: str,
    variant: str,
    candidate_records: list[CandidateFeatureRecord] | list[SweepCandidate],
    genomes_by_candidate: dict[str, GenomeModel],
    top_k_candidates: int = 5,
    episodes_per_candidate: int | None = None,
    profile: str = "kv_easy",
) -> RetrievalTraceSweepResult:
    """Run a retrieval-trace sweep across top candidates and aggregate diagnostics."""
    normalized: list[SweepCandidate] = []
    for record in candidate_records:
        if isinstance(record, CandidateFeatureRecord):
            if (
                record.benchmark_label != benchmark_label
                or (task_name and record.task_name != task_name)
                or (variant and record.variant != variant)
            ):
                continue
            normalized.append(
                SweepCandidate(
                    candidate_id=record.candidate_id,
                    run_id=record.run_id,
                    final_max_score=float(record.final_max_score),
                    success=bool(record.success),
                    task_name=record.task_name,
                    variant=record.variant,
                    delay_steps=int(record.delay_steps),
                    seed=int(record.seed),
                )
            )
        else:
            if (task_name and record.task_name and record.task_name != task_name) or (
                variant and record.variant and record.variant != variant
            ):
                continue
            normalized.append(record)

    ranked = sorted(
        normalized,
        key=lambda r: r.final_max_score if r.final_max_score is not None else float("-inf"),
        reverse=True,
    )
    chosen = ranked[: max(1, int(top_k_candidates))]

    episode_indices = _episode_indices(
        profile=profile,
        delay_steps=chosen[0].delay_steps if chosen else 8,
        limit=episodes_per_candidate,
    )

    per_candidate: list[CandidateTraceSummary] = []
    global_failures: Counter[str] = Counter()
    all_correct_flags: list[bool] = []

    traced_success_values: list[float] = []
    for record in chosen:
        genome = genomes_by_candidate.get(record.candidate_id)
        if genome is None:
            continue

        episodes: list[EpisodeTraceSummary] = []
        for sample_index in episode_indices:
            trace_result = run_retrieval_trace(
                genome,
                delay_steps=record.delay_steps,
                profile=profile,
                sample_index=sample_index,
            )
            query_indices = [
                idx for idx, role in enumerate(trace_result.step_roles) if role == "query"
            ]
            query_step = query_indices[0] if query_indices else len(trace_result.step_roles) - 1
            query_key_match_score = float(
                _mean_query_metric(trace_result.trace, "key_query_cos_post") or 0.0
            )
            value_margin = _value_margin_from_trace_output(
                raw_output=trace_result.raw_outputs,
                value_levels=trace_result.value_levels,
                query_step_index=query_step,
            )
            predicted_id = int(trace_result.predicted_value_ids[0]) if trace_result.predicted_value_ids else -1
            target_id = int(trace_result.target_value_ids[0]) if trace_result.target_value_ids else -1
            episode = EpisodeTraceSummary(
                sample_index=sample_index,
                failure_class=trace_result.verdict.mode,
                query_key_match_score=query_key_match_score,
                value_margin=value_margin,
                correct_value_selected=predicted_id == target_id,
                predicted_value_id=predicted_id,
                target_value_id=target_id,
                query_memory_alignment=_mean_query_metric(trace_result.trace, "query_memory_alignment"),
                readout_selectivity=_mean_query_metric(trace_result.trace, "readout_selectivity"),
            )
            episodes.append(episode)
            global_failures.update([episode.failure_class])
            all_correct_flags.append(episode.correct_value_selected)

        dominant, consistency = _candidate_dominant_failure(episodes)
        per_candidate.append(
            CandidateTraceSummary(
                candidate_id=record.candidate_id,
                run_id=record.run_id,
                final_max_score=record.final_max_score,
                episodes=tuple(episodes),
                dominant_failure_class=dominant,
                consistency_fraction=consistency,
            )
        )
        if record.success is not None:
            traced_success_values.append(1.0 if record.success else 0.0)

    if not per_candidate:
        raise ValueError("No traceable candidates found (missing genomes for selected records).")

    candidate_dominants = [cand.dominant_failure_class for cand in per_candidate]
    dominant_counts = Counter(candidate_dominants)
    cross_candidate_consistency = dominant_counts.most_common(1)[0][1] / max(1, len(candidate_dominants))
    overall_correct_fraction = sum(1 for x in all_correct_flags if x) / max(1, len(all_correct_flags))
    feature_success_fraction = (
        float(mean(traced_success_values)) if traced_success_values else 0.0
    )

    final_verdict = _pick_final_verdict(
        failure_frequency=global_failures,
        overall_correct_fraction=overall_correct_fraction,
        feature_success_fraction=feature_success_fraction,
        cross_candidate_consistency=cross_candidate_consistency,
    )

    return RetrievalTraceSweepResult(
        benchmark_label=benchmark_label,
        task_name=task_name,
        variant=variant,
        candidates_traced=tuple(per_candidate),
        episodes_per_candidate=len(per_candidate[0].episodes),
        failure_frequency=dict(sorted(global_failures.items(), key=lambda kv: (-kv[1], kv[0]))),
        overall_correct_fraction=overall_correct_fraction,
        cross_candidate_consistency=float(cross_candidate_consistency),
        final_verdict=final_verdict,
        next_step_hint=_next_step_hint(final_verdict),
        candidate_source="unknown",
    )


def render_retrieval_trace_sweep_report(result: RetrievalTraceSweepResult) -> str:
    lines: list[str] = []
    lines.append(f"# Retrieval Trace Sweep — {result.benchmark_label}")
    lines.append("")
    lines.append("## 1. Dataset summary")
    lines.append("")
    lines.append(f"- benchmark label: `{result.benchmark_label}`")
    lines.append(f"- task: `{result.task_name}`")
    lines.append(f"- variant: `{result.variant}`")
    lines.append(f"- number of candidates traced: `{len(result.candidates_traced)}`")
    lines.append(f"- episodes per candidate: `{result.episodes_per_candidate}`")
    lines.append(f"- candidate discovery source: `{result.candidate_source}`")
    lines.append("")

    lines.append("## 2. Candidate table")
    lines.append("")
    lines.append("| candidate_id | run_id | final score | dominant failure class | consistency fraction |")
    lines.append("| --- | --- | ---: | --- | ---: |")
    for candidate in result.candidates_traced:
        score_txt = "n/a" if candidate.final_max_score is None else f"{candidate.final_max_score:.4f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate.candidate_id,
                    candidate.run_id,
                    score_txt,
                    candidate.dominant_failure_class,
                    f"{candidate.consistency_fraction:.3f}",
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## 3. Failure frequency table")
    lines.append("")
    lines.append("| failure class | count |")
    lines.append("| --- | ---: |")
    for failure_class, count in result.failure_frequency.items():
        lines.append(f"| {failure_class} | {count} |")
    lines.append("")

    all_eps = [ep for candidate in result.candidates_traced for ep in candidate.episodes]
    mean_qk = mean(ep.query_key_match_score for ep in all_eps)
    mean_margin = mean(ep.value_margin for ep in all_eps)
    mean_qm = mean((ep.query_memory_alignment or 0.0) for ep in all_eps)
    mean_rs = mean((ep.readout_selectivity or 0.0) for ep in all_eps)

    lines.append("## 4. Aggregate metric summary")
    lines.append("")
    lines.append(f"- overall correct_value_selected fraction: `{result.overall_correct_fraction:.3f}`")
    lines.append(f"- cross-candidate consistency: `{result.cross_candidate_consistency:.3f}`")
    lines.append(f"- mean query_key_match_score: `{mean_qk:.3f}`")
    lines.append(f"- mean value_margin: `{mean_margin:.3f}`")
    lines.append(f"- mean query_memory_alignment: `{mean_qm:.3f}`")
    lines.append(f"- mean readout_selectivity: `{mean_rs:.3f}`")
    lines.append("")

    lines.append("## 5. Final verdict")
    lines.append("")
    lines.append(f"`{result.final_verdict}`")
    lines.append("")

    lines.append("## 6. Next-step hint")
    lines.append("")
    lines.append(result.next_step_hint)
    lines.append("")

    return "\n".join(lines)


def write_retrieval_trace_sweep_report(
    result: RetrievalTraceSweepResult,
    *,
    output_dir: str | Path = "results",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"retrieval-trace-sweep-{result.benchmark_label}.md"
    report_path.write_text(render_retrieval_trace_sweep_report(result), encoding="utf-8")
    return report_path
