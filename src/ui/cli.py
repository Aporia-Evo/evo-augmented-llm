from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.search_space import (
    SearchSpaceSnapshot,
    SUMMARY_FEATURES,
    derive_search_space_hints,
    filter_feature_records,
    load_feature_records_from_jsonl,
    load_search_space_snapshot_from_markdown,
    render_cross_label_snapshot_report,
    render_cross_label_search_space_report,
    render_search_space_report,
    write_feature_records_jsonl,
)
from analysis.archive import (
    filter_archive_cells,
    filter_archive_events,
    load_archive_cells_from_jsonl,
    load_archive_events_from_jsonl,
    render_archive_report,
    write_archive_cells_jsonl,
    write_archive_events_jsonl,
)
from analysis.curriculum_boundaries import render_curriculum_boundary_report
from analysis.fitness_landscape import (
    analyze_fitness_landscape,
    render_landscape_report,
    write_landscape_report,
)
from analysis.retrieval_trace import (
    render_trace_report,
    run_retrieval_trace,
    write_trace_report,
)
from analysis.retrieval_trace_sweep import (
    SweepCandidate,
    resolve_local_sweep_inputs,
    run_retrieval_trace_sweep,
    write_retrieval_trace_sweep_report,
)
from config import (
    AppConfig,
    curriculum_phase_delay_labels,
    curriculum_switch_generation,
    evaluation_delay_label,
    load_config,
)
from db.client import SpacetimeHttpClient
from db.models import (
    ArchiveCellRecord,
    ArchiveEventRecord,
    CandidateFeatureRecord,
    EventRecord,
    GenerationCommitResult,
    GenerationRecord,
    RunRecord,
)
from db.reducers import InMemoryRepository, RunRepository, SpacetimeRepository
from evolve.benchmark_runner import run_online_benchmark
from evolve.archive import DEFAULT_QD_PROFILE, archive_profile_names
from evolve.evaluator import score_ceiling_for_task
from evolve.genome_codec import GenomeModel, genome_model_from_blob
from evolve.online_loop import execute_online_run
from evolve.run_loop import execute_run
from ui.compare_report import build_online_benchmark_aggregates, build_online_compare_summary
from ui.online_cli import OnlineCliObserver
from utils.serialization import stable_json_dumps, utc_now_iso
from utils.scoring import resolve_success


DEFAULT_CONFIGS = ["configs/base.yaml", "configs/xor.yaml", "configs/local.yaml"]
TASK_CHOICES = ["xor", "delayed_xor", "bit_memory", "key_value_memory", "event_memory", "event_decision"]
VARIANT_CHOICES = [
    "stateful",
    "stateful_v2",
    "stateful_v2_gated",
    "stateful_v3_kv",
    "stateful_v4_slots",
    "stateful_v5_addressed_slots",
    "stateful_v6_delta_memory",
    "stateful_v6_delta_memory_v16a",
    "stateful_v6_delta_memory_v16b",
    "stateful_v6_delta_memory_v16c",
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
GENERATION_SUITE_DEFAULT_TASKS = ["delayed_xor", "bit_memory"]
TASK_DEFAULT_CONFIGS = {
    "xor": ["configs/base.yaml", "configs/xor.yaml", "configs/local.yaml"],
    "delayed_xor": ["configs/base.yaml", "configs/delayed_xor.yaml", "configs/local.yaml"],
    "bit_memory": ["configs/base.yaml", "configs/bit_memory.yaml", "configs/local.yaml"],
    "key_value_memory": ["configs/base.yaml", "configs/key_value_memory.yaml", "configs/local.yaml"],
    "event_memory": ["configs/base.yaml", "configs/event_memory.yaml", "configs/local.yaml"],
    "event_decision": ["configs/base.yaml", "configs/event_decision.yaml", "configs/local.yaml"],
}


@dataclass(frozen=True)
class CompareSummary:
    run_id: str
    variant: str
    seed: int
    task_name: str
    delay_steps: int
    status: str
    first_max_generation: int | None
    first_success_generation: int | None
    final_max_score: float | None
    score_ceiling: float
    success: bool
    avg_score_path: str
    resume_hint: str
    evaluation_delay_steps_label: str = ""
    curriculum_enabled: bool = False
    curriculum_phase_1_delays: str = ""
    curriculum_phase_2_delays: str = ""
    curriculum_switch_generation: int = 0


@dataclass(frozen=True)
class BenchmarkAggregate:
    delay_steps: int
    variant: str
    run_count: int
    completed_count: int
    success_rate: float
    mean_final_max_score: float | None
    mean_first_max_generation: float | None
    mean_first_success_generation: float | None
    best_final_max_score: float | None


@dataclass(frozen=True)
class GenerationBenchmarkRow:
    task_name: str
    delay_steps: int
    variant: str
    seed: int
    run_id: str
    status: str
    generation_budget: int
    population_size: int
    score_ceiling: float
    success: bool
    first_success_generation: int | None
    best_generation_id: int | None
    best_candidate_id: str | None
    final_max_score: float | None
    final_avg_score: float | None
    completed_generations: int
    best_node_count: int | None
    best_enabled_connection_count: int | None
    success_node_count: int | None
    success_enabled_connection_count: int | None
    avg_score_path: str
    resume_hint: str
    evaluation_delay_steps_label: str = ""
    curriculum_enabled: bool = False
    curriculum_phase_1_delays: str = ""
    curriculum_phase_2_delays: str = ""
    curriculum_switch_generation: int = 0
    query_accuracy: float | None = None
    retrieval_score: float | None = None
    mean_query_distance: float | None = None
    distractor_load: float | None = None
    correct_key_selected: float | None = None
    correct_value_selected: float | None = None
    query_key_match_score: float | None = None
    value_margin: float | None = None
    distractor_competition_score: float | None = None
    store_vs_distractor_write_gap: float | None = None
    query_value_read_strength: float | None = None
    slot_write_focus: float | None = None
    slot_query_focus: float | None = None
    slot_readout_selectivity: float | None = None
    slot_utilization: float | None = None
    query_slot_match_max: float | None = None
    slot_distractor_leak: float | None = None
    mean_write_address_focus: float | None = None
    mean_read_address_focus: float | None = None
    write_read_address_gap: float | None = None
    readout_address_concentration: float | None = None
    store_vs_distractor_beta_gap: float | None = None
    query_memory_alignment: float | None = None
    delta_correction_magnitude: float | None = None
    mean_memory_frobenius_norm: float | None = None


@dataclass(frozen=True)
class GenerationSuiteAggregate:
    task_name: str
    delay_steps: int
    variant: str
    run_count: int
    completed_count: int
    success_rate: float
    mean_final_max_score: float | None
    mean_first_success_generation: float | None
    median_first_success_generation: float | None
    mean_best_node_count: float | None
    mean_best_enabled_connection_count: float | None
    evaluation_delay_steps_label: str = ""
    curriculum_enabled: bool = False
    curriculum_phase_1_delays: str = ""
    curriculum_phase_2_delays: str = ""
    curriculum_switch_generation: int = 0
    mean_query_accuracy: float | None = None
    mean_retrieval_score: float | None = None
    mean_query_distance: float | None = None
    mean_distractor_load: float | None = None
    mean_correct_key_selected: float | None = None
    mean_correct_value_selected: float | None = None
    mean_query_key_match_score: float | None = None
    mean_value_margin: float | None = None
    mean_distractor_competition_score: float | None = None
    mean_store_vs_distractor_write_gap: float | None = None
    mean_query_value_read_strength: float | None = None
    mean_slot_write_focus: float | None = None
    mean_slot_query_focus: float | None = None
    mean_slot_readout_selectivity: float | None = None
    mean_slot_utilization: float | None = None
    mean_query_slot_match_max: float | None = None
    mean_slot_distractor_leak: float | None = None
    mean_write_address_focus: float | None = None
    mean_read_address_focus: float | None = None
    mean_write_read_address_gap: float | None = None
    mean_readout_address_concentration: float | None = None
    mean_store_vs_distractor_beta_gap: float | None = None
    mean_query_memory_alignment: float | None = None
    mean_delta_correction_magnitude: float | None = None
    mean_memory_frobenius_norm: float | None = None


@dataclass(frozen=True)
class CandidateGenomeExportRecord:
    benchmark_label: str
    run_id: str
    generation_id: int
    candidate_id: str
    task_name: str
    variant: str
    seed: int
    genome_blob: str


class CliObserver:
    def on_run_started(self, run: RunRecord) -> None:
        variant = _variant_from_run(run)
        print(
            f"run_id={run.run_id} task={run.task_name} variant={variant} seed={run.seed} status={run.status}"
        )

    def on_generation_committed(self, result: GenerationCommitResult) -> None:
        elite_candidate_ids = [elite.candidate_id for elite in result.elites]
        print(
            "generation_id="
            f"{result.generation.generation_id} "
            f"best_score={result.generation.best_score:.6f} "
            f"avg_score={result.generation.avg_score:.6f} "
            f"eval_duration_ms={result.generation.eval_duration_ms} "
            f"commit_duration_ms={result.generation.commit_duration_ms} "
            f"elite_candidate_ids={elite_candidate_ids}"
        )

    def on_run_finished(self, run: RunRecord) -> None:
        print(f"run_id={run.run_id} status={run.status} finished_at={run.finished_at}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TensorNEAT + SpacetimeDB local demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start or resume a local evolution run")
    _add_common_run_args(run_parser)
    run_parser.add_argument("--resume-run-id", default=None, help="Resume an existing run id")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run a small multi-seed benchmark")
    _add_common_run_args(benchmark_parser)
    benchmark_parser.add_argument(
        "--seeds",
        default="7,11,13",
        help="Comma-separated list of seeds, for example: 7,11,13",
    )
    benchmark_parser.add_argument("--generations", type=int, default=None)
    benchmark_parser.add_argument("--population-size", type=int, default=None)
    benchmark_parser.add_argument("--elite-top-k", type=int, default=None)
    benchmark_parser.add_argument(
        "--variants",
        default="stateful,stateless",
        help="Comma-separated generation variants, for example: stateful,stateless,stateful_plastic_hebb,stateful_plastic_ad",
    )
    benchmark_parser.add_argument(
        "--delay-sweep",
        default=None,
        help="Optional comma-separated temporal delays, for example: 1,3,5,8",
    )

    suite_parser = subparsers.add_parser(
        "benchmark-suite",
        help="Run a reproducible generation benchmark suite and export results",
    )
    suite_parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        default=[],
        help="Optional extra config file path applied after the built-in task configs.",
    )
    suite_parser.add_argument(
        "--store",
        choices=["memory", "spacetimedb"],
        default="memory",
        help="Persistence backend.",
    )
    suite_parser.add_argument("--server-url", default=None, help="SpacetimeDB HTTP server URL")
    suite_parser.add_argument("--database-name", default=None, help="SpacetimeDB database name")
    suite_parser.add_argument(
        "--tasks",
        default="delayed_xor,bit_memory",
        help="Comma-separated generation tasks, for example: delayed_xor,bit_memory",
    )
    suite_parser.add_argument(
        "--seeds",
        default="7,11,13,17,19",
        help="Comma-separated list of seeds.",
    )
    suite_parser.add_argument("--generations", type=int, default=None)
    suite_parser.add_argument("--population-size", type=int, default=None)
    suite_parser.add_argument("--elite-top-k", type=int, default=None)
    suite_parser.add_argument(
        "--variants",
        default="stateful,stateless",
        help="Comma-separated generation variants, for example: stateful,stateless,stateful_plastic_hebb,stateful_plastic_ad",
    )
    suite_parser.add_argument(
        "--task-delay-sweep",
        action="append",
        default=[],
        help="Optional task-specific delay sweep, for example: bit_memory=1,3,5,8. Can be repeated.",
    )
    suite_parser.add_argument(
        "--temporal-delay-steps",
        type=int,
        default=None,
        help="Optional override for the base temporal delay used by the task config.",
    )
    suite_parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for jsonl/csv/markdown exports.",
    )
    suite_parser.add_argument(
        "--label",
        default=None,
        help="Optional label for export filenames.",
    )
    suite_parser.add_argument(
        "--evaluation-delay-steps",
        default=None,
        help="Optional comma-separated evaluation delays for tasks that support multi-delay evaluation, for example: 5,8",
    )
    suite_parser.add_argument(
        "--curriculum-enabled",
        action="store_true",
        help="Enable curriculum evaluation for supported tasks.",
    )
    suite_parser.add_argument(
        "--curriculum-phase-switch-generation",
        "--curriculum-phase-split-generation",
        type=int,
        dest="curriculum_phase_switch_generation",
        default=None,
        help="Generation index where curriculum switches from phase 1 to phase 2.",
    )
    suite_parser.add_argument(
        "--curriculum-phase-1-delays",
        default=None,
        help="Comma-separated evaluation delays for curriculum phase 1, for example: 5",
    )
    suite_parser.add_argument(
        "--curriculum-phase-2-delays",
        default=None,
        help="Comma-separated evaluation delays for curriculum phase 2, for example: 5,8",
    )
    suite_parser.add_argument(
        "--key-value-profile",
        choices=["kv_easy", "kv_mid", "kv_full"],
        default=None,
        help="Complexity profile for key_value_memory.",
    )
    suite_parser.add_argument(
        "--curriculum-phase-1-key-value-profile",
        choices=["kv_easy", "kv_mid", "kv_full"],
        default=None,
        help="Complexity profile for key_value_memory in curriculum phase 1.",
    )
    suite_parser.add_argument(
        "--curriculum-phase-2-key-value-profile",
        choices=["kv_easy", "kv_mid", "kv_full"],
        default=None,
        help="Complexity profile for key_value_memory in curriculum phase 2.",
    )

    online_run_parser = subparsers.add_parser("run-online", help="Start or resume an online rtNEAT-like run")
    _add_common_run_args(online_run_parser)
    online_run_parser.add_argument("--resume-run-id", default=None, help="Resume an existing online run id")
    online_run_parser.add_argument("--online-steps", type=int, default=None)
    online_run_parser.add_argument("--replacement-interval", type=int, default=None)
    online_run_parser.add_argument("--metrics-interval", type=int, default=None)
    online_run_parser.add_argument("--active-population-size", type=int, default=None)

    online_status_parser = subparsers.add_parser("status-online", help="Show online run status")
    online_status_parser.add_argument("run_id")
    online_status_parser.add_argument("--server-url", required=True)
    online_status_parser.add_argument("--database-name", required=True)

    online_benchmark_parser = subparsers.add_parser("benchmark-online", help="Run a small online benchmark")
    _add_common_run_args(online_benchmark_parser)
    online_benchmark_parser.add_argument("--seeds", default="7,11,13")
    online_benchmark_parser.add_argument("--online-steps", type=int, default=None)
    online_benchmark_parser.add_argument("--replacement-interval", type=int, default=None)
    online_benchmark_parser.add_argument("--metrics-interval", type=int, default=None)
    online_benchmark_parser.add_argument("--active-population-size", type=int, default=None)

    online_compare_parser = subparsers.add_parser("compare-online", help="Compare two online runs")
    online_compare_parser.add_argument("--server-url", required=True)
    online_compare_parser.add_argument("--database-name", required=True)
    online_compare_parser.add_argument("--seed", type=int, default=None)
    online_compare_parser.add_argument("--task-name", choices=TASK_CHOICES, default=None)
    online_compare_parser.add_argument("--stateful-run-id", default=None)
    online_compare_parser.add_argument("--stateless-run-id", default=None)
    online_compare_parser.add_argument("--search-limit", type=int, default=100)

    runs_parser = subparsers.add_parser("runs", help="List recent persisted runs")
    runs_parser.add_argument("--server-url", required=True)
    runs_parser.add_argument("--database-name", required=True)
    runs_parser.add_argument("--limit", type=int, default=10)

    status_parser = subparsers.add_parser("status", help="Show run and generation status")
    status_parser.add_argument("run_id")
    status_parser.add_argument("--server-url", required=True)
    status_parser.add_argument("--database-name", required=True)

    compare_parser = subparsers.add_parser("compare", help="Compare stateful vs stateless runs")
    compare_parser.add_argument("--server-url", required=True)
    compare_parser.add_argument("--database-name", required=True)
    compare_parser.add_argument("--seed", type=int, default=None)
    compare_parser.add_argument("--task-name", choices=TASK_CHOICES, default=None)
    compare_parser.add_argument("--stateful-run-id", default=None)
    compare_parser.add_argument("--stateless-run-id", default=None)
    compare_parser.add_argument("--search-limit", type=int, default=100)

    elite_parser = subparsers.add_parser("elites", help="Show archived elites")
    elite_parser.add_argument("run_id")
    elite_parser.add_argument("--server-url", required=True)
    elite_parser.add_argument("--database-name", required=True)
    elite_parser.add_argument("--limit", type=int, default=10)

    event_parser = subparsers.add_parser("events", help="Show recent events")
    event_parser.add_argument("run_id")
    event_parser.add_argument("--server-url", required=True)
    event_parser.add_argument("--database-name", required=True)
    event_parser.add_argument("--limit", type=int, default=20)

    analyze_parser = subparsers.add_parser(
        "analyze-search-space",
        help="Summarize persisted candidate features for a benchmark label",
    )
    analyze_parser.add_argument(
        "--store",
        choices=["memory", "spacetimedb"],
        default="memory",
        help="Feature source: local feature export or SpacetimeDB tables.",
    )
    analyze_parser.add_argument("--benchmark-label", required=True)
    analyze_parser.add_argument("--task", choices=TASK_CHOICES, default=None)
    analyze_parser.add_argument("--variant", choices=VARIANT_CHOICES, default=None)
    analyze_parser.add_argument("--delay", type=int, default=None)
    analyze_parser.add_argument(
        "--curriculum-phase",
        choices=["phase_1", "phase_2", "static"],
        default=None,
        help="Optional curriculum phase filter for candidate-feature analysis.",
    )
    analyze_parser.add_argument("--output-dir", default="results")
    analyze_parser.add_argument("--server-url", default=None)
    analyze_parser.add_argument("--database-name", default=None)

    cross_label_parser = subparsers.add_parser(
        "analyze-search-space-cross-labels",
        help="Compare search-space diagnostics across multiple benchmark labels",
    )
    cross_label_parser.add_argument(
        "--benchmark-labels",
        required=True,
        help="Comma-separated benchmark labels to compare, for example: v14r-delta,v14t-delta,v14u-delta",
    )
    cross_label_parser.add_argument("--task", choices=TASK_CHOICES, default=None)
    cross_label_parser.add_argument("--variant", choices=VARIANT_CHOICES, default=None)
    cross_label_parser.add_argument("--delay", type=int, default=None)
    cross_label_parser.add_argument(
        "--curriculum-phase",
        choices=["phase_1", "phase_2", "static"],
        default=None,
        help="Optional curriculum phase filter for candidate-feature analysis.",
    )
    cross_label_parser.add_argument("--output-dir", default="results")
    cross_label_parser.add_argument(
        "--report-name",
        default="v15e-search-space-cross-variant.md",
        help="Output markdown filename written under --output-dir.",
    )
    cross_label_parser.add_argument("--top-hints", type=int, default=3)

    archive_parser = subparsers.add_parser(
        "analyze-archive",
        help="Summarize persisted QD archive cells for a benchmark label",
    )
    archive_parser.add_argument(
        "--store",
        choices=["memory", "spacetimedb"],
        default="memory",
        help="Archive source: local export or SpacetimeDB tables.",
    )
    archive_parser.add_argument("--benchmark-label", required=True)
    archive_parser.add_argument("--task", choices=TASK_CHOICES, default=None)
    archive_parser.add_argument("--variant", choices=VARIANT_CHOICES, default=None)
    archive_parser.add_argument("--delay", type=int, default=None)
    archive_parser.add_argument(
        "--curriculum-phase",
        choices=["phase_1", "phase_2", "static"],
        default=None,
        help="Optional curriculum phase filter for archive analysis.",
    )
    archive_parser.add_argument(
        "--qd-profile",
        choices=list(archive_profile_names(task_name="key_value_memory")),
        default=DEFAULT_QD_PROFILE,
        help="QD descriptor profile to analyze. Defaults to mechanism_v2.",
    )
    archive_parser.add_argument("--output-dir", default="results")
    archive_parser.add_argument("--server-url", default=None)
    archive_parser.add_argument("--database-name", default=None)

    curriculum_compare_parser = subparsers.add_parser(
        "analyze-curriculum-boundaries",
        help="Compare curriculum boundary runs across exported benchmark labels",
    )
    curriculum_compare_parser.add_argument(
        "--benchmark-labels",
        required=True,
        help="Comma-separated benchmark labels, for example: v8c-boundary4,v8c-boundary6,v8c-boundary8",
    )
    curriculum_compare_parser.add_argument("--task", choices=TASK_CHOICES, default="bit_memory")
    curriculum_compare_parser.add_argument(
        "--variants",
        default="stateful,stateful_v2,stateful_v2_gated,stateful_v3_kv,stateful_v4_slots,stateful_v5_addressed_slots,stateful_v6_delta_memory,content_gated,stateful_plastic_hebb",
        help="Comma-separated variants to include in the comparison report.",
    )
    curriculum_compare_parser.add_argument(
        "--focus-variant",
        choices=VARIANT_CHOICES,
        default="stateful_v2",
        help="Variant used for the boundary decision hint.",
    )
    curriculum_compare_parser.add_argument("--output-dir", default="results")

    # -- Retrieval trace (mechanistic per-episode diagnosis) --

    retrieval_trace_parser = subparsers.add_parser(
        "analyze-retrieval-trace",
        help="Run a single genome through kv_easy with full per-step instrumentation",
    )
    retrieval_trace_parser.add_argument(
        "--store",
        choices=["memory", "spacetimedb"],
        default="memory",
        help="Genome source: local feature export or SpacetimeDB.",
    )
    retrieval_trace_parser.add_argument("--benchmark-label", required=True)
    retrieval_trace_parser.add_argument(
        "--candidate-id",
        default=None,
        help="Specific candidate ID to trace. If omitted, uses the top scorer.",
    )
    retrieval_trace_parser.add_argument("--delay", type=int, default=8)
    retrieval_trace_parser.add_argument("--profile", default="kv_easy")
    retrieval_trace_parser.add_argument("--sample-index", type=int, default=0)
    retrieval_trace_parser.add_argument("--output-dir", default="results")
    retrieval_trace_parser.add_argument("--server-url", default=None)
    retrieval_trace_parser.add_argument("--database-name", default=None)

    retrieval_trace_sweep_parser = subparsers.add_parser(
        "analyze-retrieval-trace-sweep",
        help="Run retrieval trace diagnostics across top candidates and episodes",
    )
    retrieval_trace_sweep_parser.add_argument(
        "--store",
        choices=["memory", "spacetimedb"],
        default="memory",
        help="Genome source: local feature export or SpacetimeDB.",
    )
    retrieval_trace_sweep_parser.add_argument("--benchmark-label", required=True)
    retrieval_trace_sweep_parser.add_argument("--task", choices=TASK_CHOICES, default="key_value_memory")
    retrieval_trace_sweep_parser.add_argument("--variant", choices=VARIANT_CHOICES, default="stateful_v6_delta_memory")
    retrieval_trace_sweep_parser.add_argument("--top-k-candidates", type=int, default=5)
    retrieval_trace_sweep_parser.add_argument(
        "--episodes-per-candidate",
        type=int,
        default=0,
        help="Number of deterministic kv_easy samples per candidate; <=0 means all samples.",
    )
    retrieval_trace_sweep_parser.add_argument("--output-dir", default="results")
    retrieval_trace_sweep_parser.add_argument("--server-url", default=None)
    retrieval_trace_sweep_parser.add_argument("--database-name", default=None)

    # -- Fitness landscape (population-level retrieval diagnosis) --

    fitness_landscape_parser = subparsers.add_parser(
        "analyze-fitness-landscape",
        help="Analyze fitness-vs-retrieval correlations from persisted candidate features",
    )
    fitness_landscape_parser.add_argument(
        "--store",
        choices=["memory", "spacetimedb"],
        default="memory",
        help="Feature source: local feature export or SpacetimeDB tables.",
    )
    fitness_landscape_parser.add_argument("--benchmark-label", required=True)
    fitness_landscape_parser.add_argument("--task", choices=TASK_CHOICES, default=None)
    fitness_landscape_parser.add_argument("--variant", choices=VARIANT_CHOICES, default=None)
    fitness_landscape_parser.add_argument("--delay", type=int, default=None)
    fitness_landscape_parser.add_argument("--top-k", type=int, default=20)
    fitness_landscape_parser.add_argument("--bins", type=int, default=5)
    fitness_landscape_parser.add_argument("--output-dir", default="results")
    fitness_landscape_parser.add_argument("--server-url", default=None)
    fitness_landscape_parser.add_argument("--database-name", default=None)

    return parser


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        default=[],
        help="Config file path. Can be repeated.",
    )
    parser.add_argument(
        "--store",
        choices=["memory", "spacetimedb"],
        default="memory",
        help="Persistence backend.",
    )
    parser.add_argument("--server-url", default=None, help="SpacetimeDB HTTP server URL")
    parser.add_argument("--database-name", default=None, help="SpacetimeDB database name")
    parser.add_argument("--task-name", choices=TASK_CHOICES, default=None)
    parser.add_argument("--variant", choices=VARIANT_CHOICES, default=None)
    parser.add_argument("--temporal-delay-steps", type=int, default=None)
    parser.add_argument(
        "--evaluation-delay-steps",
        default=None,
        help="Optional comma-separated evaluation delays for supported temporal tasks, for example: 5,8",
    )
    parser.add_argument(
        "--curriculum-enabled",
        action="store_true",
        help="Enable curriculum evaluation for supported tasks.",
    )
    parser.add_argument(
        "--curriculum-phase-switch-generation",
        "--curriculum-phase-split-generation",
        type=int,
        dest="curriculum_phase_switch_generation",
        default=None,
        help="Generation index where curriculum switches from phase 1 to phase 2.",
    )
    parser.add_argument(
        "--curriculum-phase-1-delays",
        default=None,
        help="Comma-separated evaluation delays for curriculum phase 1, for example: 5",
    )
    parser.add_argument(
        "--curriculum-phase-2-delays",
        default=None,
        help="Comma-separated evaluation delays for curriculum phase 2, for example: 5,8",
    )
    parser.add_argument(
        "--key-value-profile",
        choices=["kv_easy", "kv_mid", "kv_full"],
        default=None,
        help="Complexity profile for key_value_memory.",
    )
    parser.add_argument(
        "--curriculum-phase-1-key-value-profile",
        choices=["kv_easy", "kv_mid", "kv_full"],
        default=None,
        help="Complexity profile for key_value_memory in curriculum phase 1.",
    )
    parser.add_argument(
        "--curriculum-phase-2-key-value-profile",
        choices=["kv_easy", "kv_mid", "kv_full"],
        default=None,
        help="Complexity profile for key_value_memory in curriculum phase 2.",
    )


def _load_runtime_config(config_paths: Sequence[str]) -> AppConfig:
    paths = list(config_paths) if config_paths else list(DEFAULT_CONFIGS)
    return load_config(paths)


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    task = config.task
    run = config.run
    online = config.online
    evolution = config.evolution

    if getattr(args, "task_name", None) is not None:
        task = replace(task, name=args.task_name)
    if getattr(args, "temporal_delay_steps", None) is not None:
        task = replace(task, temporal_delay_steps=args.temporal_delay_steps)
    if getattr(args, "evaluation_delay_steps", None) is not None:
        task = replace(task, evaluation_delay_steps=_parse_optional_delay_list(args.evaluation_delay_steps))
    if getattr(args, "curriculum_enabled", False):
        task = replace(task, curriculum_enabled=True)
    if getattr(args, "curriculum_phase_switch_generation", None) is not None:
        task = replace(
            task,
            curriculum_phase_switch_generation=args.curriculum_phase_switch_generation,
            curriculum_phase_split_generation=args.curriculum_phase_switch_generation,
        )
    if getattr(args, "curriculum_phase_1_delays", None) is not None:
        task = replace(task, curriculum_phase_1_delay_steps=_parse_optional_delay_list(args.curriculum_phase_1_delays))
    if getattr(args, "curriculum_phase_2_delays", None) is not None:
        task = replace(task, curriculum_phase_2_delay_steps=_parse_optional_delay_list(args.curriculum_phase_2_delays))
    if getattr(args, "key_value_profile", None) is not None:
        task = replace(task, key_value_profile=str(args.key_value_profile))
    if getattr(args, "curriculum_phase_1_key_value_profile", None) is not None:
        task = replace(task, curriculum_phase_1_key_value_profile=str(args.curriculum_phase_1_key_value_profile))
    if getattr(args, "curriculum_phase_2_key_value_profile", None) is not None:
        task = replace(task, curriculum_phase_2_key_value_profile=str(args.curriculum_phase_2_key_value_profile))
    if getattr(args, "variant", None) is not None:
        run = replace(run, variant=args.variant)
    if getattr(args, "generations", None) is not None:
        run = replace(run, generations=args.generations)
    if getattr(args, "elite_top_k", None) is not None:
        run = replace(run, elite_top_k=args.elite_top_k)
    if getattr(args, "population_size", None) is not None:
        evolution = replace(evolution, population_size=args.population_size)
    if getattr(args, "online_steps", None) is not None:
        online = replace(online, max_steps=args.online_steps)
    if getattr(args, "replacement_interval", None) is not None:
        online = replace(online, replacement_interval=args.replacement_interval)
    if getattr(args, "metrics_interval", None) is not None:
        online = replace(online, metrics_interval=args.metrics_interval)
    if getattr(args, "active_population_size", None) is not None:
        online = replace(online, active_population_size=args.active_population_size)

    return replace(config, task=task, run=run, online=online, evolution=evolution)


def _spacetime_repository(config: AppConfig, server_url: str | None, database_name: str | None) -> SpacetimeRepository:
    client = SpacetimeHttpClient(
        server_url=server_url or config.spacetimedb.server_url,
        database_name=database_name or config.spacetimedb.database_name,
        timeout_seconds=config.spacetimedb.timeout_seconds,
    )
    return SpacetimeRepository(client=client, run_id_prefix=config.run.run_id_prefix)


def _print_run_status(repository: RunRepository, run_id: str) -> int:
    run = repository.get_run(run_id)
    if run is None:
        print(f"Run not found: {run_id}")
        return 1

    variant = _variant_from_run(run)
    print(
        f"run_id={run.run_id} task={run.task_name} variant={variant} seed={run.seed} "
        f"status={run.status} created_at={run.created_at} finished_at={run.finished_at}"
    )
    for generation in repository.list_generations(run_id):
        checkpoint = repository.get_checkpoint(run_id, generation.generation_id)
        print(
            f"generation_id={generation.generation_id} state={generation.state} "
            f"best_candidate_id={generation.best_candidate_id} "
            f"best_score={generation.best_score} avg_score={generation.avg_score} "
            f"eval_duration_ms={generation.eval_duration_ms} "
            f"commit_duration_ms={generation.commit_duration_ms} "
            f"checkpoint={'yes' if checkpoint is not None else 'no'}"
        )
    return 0


def _print_compare_report(repository: RunRepository, args: argparse.Namespace) -> int:
    stateful_run, stateless_run = _resolve_compare_runs(repository, args)
    if stateful_run is None or stateless_run is None:
        print("Could not find both stateful and stateless runs for the requested comparison.")
        return 1

    stateful_summary = _build_compare_summary(repository, stateful_run)
    stateless_summary = _build_compare_summary(repository, stateless_run)

    print(
        f"seed={stateful_summary.seed} task={stateful_summary.task_name} "
        f"delay_steps={stateful_summary.delay_steps} "
        f"stateful_run_id={stateful_summary.run_id} stateless_run_id={stateless_summary.run_id}"
    )
    for summary in (stateful_summary, stateless_summary):
        print(
            f"variant={summary.variant} run_id={summary.run_id} status={summary.status} "
            f"first_max_generation={summary.first_max_generation} "
            f"first_success_generation={summary.first_success_generation} "
            f"final_max_score={summary.final_max_score} score_ceiling={summary.score_ceiling} "
            f"success={summary.success} avg_score_path={summary.avg_score_path} "
            f"resume_hint={summary.resume_hint}"
        )
    return 0


def _print_benchmark_report(args: argparse.Namespace) -> int:
    initial_config = _load_runtime_config(args.configs)
    base_config = _apply_overrides(initial_config, args)
    seeds = _parse_seed_list(args.seeds)
    variants = _parse_variants(args.variants)
    delay_values = _parse_delay_sweep(args.delay_sweep, base_config.task.temporal_delay_steps)
    repository: RunRepository
    if args.store == "memory":
        repository = InMemoryRepository(run_id_prefix=base_config.run.run_id_prefix)
    else:
        repository = _spacetime_repository(base_config, args.server_url, args.database_name)

    print(
        f"benchmark_task={base_config.task.name} store={args.store} seeds={seeds} "
        f"variants={variants} "
        f"delays={delay_values} generations={base_config.run.generations} "
        f"population_size={base_config.evolution.population_size}",
        flush=True,
    )

    summaries = _run_generation_benchmark(
        base_config=base_config,
        repository=repository,
        seeds=seeds,
        variants=variants,
        delay_values=delay_values,
        on_summary=_print_generation_benchmark_progress,
    )

    for aggregate in _build_benchmark_aggregates(summaries):
        mean_first_success = (
            f"{aggregate.mean_first_success_generation:.2f}"
            if aggregate.mean_first_success_generation is not None
            else "n/a"
        )
        mean_final = (
            f"{aggregate.mean_final_max_score:.6f}"
            if aggregate.mean_final_max_score is not None
            else "n/a"
        )
        best_final = (
            f"{aggregate.best_final_max_score:.6f}"
            if aggregate.best_final_max_score is not None
            else "n/a"
        )
        print(
            f"aggregate delay_steps={aggregate.delay_steps} variant={aggregate.variant} "
            f"runs={aggregate.run_count} completed_runs={aggregate.completed_count} "
            f"success_rate={aggregate.success_rate:.3f} "
            f"mean_final_max_score={mean_final} "
            f"mean_first_success_generation={mean_first_success} "
            f"best_final_max_score={best_final}",
            flush=True,
        )

    for delay_summary in _pair_delay_summaries(summaries):
        print(
            f"delay_compare delay_steps={delay_summary['delay_steps']} "
            f"left_variant={delay_summary['left_variant']} "
            f"right_variant={delay_summary['right_variant']} "
            f"left_success_rate={delay_summary['left_success_rate']:.3f} "
            f"right_success_rate={delay_summary['right_success_rate']:.3f} "
            f"delta_success_rate={delay_summary['delta_success_rate']:.3f} "
            f"left_mean_first_success_generation={delay_summary['left_mean_first_success_generation']} "
            f"right_mean_first_success_generation={delay_summary['right_mean_first_success_generation']} "
            f"delta_mean_final_max_score={delay_summary['delta_mean_final_max_score']:.6f}",
            flush=True,
        )
    return 0


def _run_generation_benchmark(
    *,
    base_config: AppConfig,
    repository: RunRepository,
    seeds: list[int],
    variants: list[str],
    delay_values: list[int],
    benchmark_label: str | None = None,
    on_summary: Callable[[CompareSummary, int, int], None] | None = None,
) -> list[CompareSummary]:
    summaries: list[CompareSummary] = []
    total_runs = len(seeds) * len(delay_values) * len(variants)
    completed_runs = 0
    for delay_steps in delay_values:
        for seed in seeds:
            for variant in variants:
                benchmark_config = replace(
                    base_config,
                    task=replace(base_config.task, temporal_delay_steps=delay_steps),
                    run=replace(base_config.run, seed=seed, variant=variant),
                )
                result = execute_run(
                    config=benchmark_config,
                    repository=repository,
                    benchmark_label=benchmark_label,
                )
                summary = _build_compare_summary(repository, result.run)
                summaries.append(summary)
                completed_runs += 1
                if on_summary is not None:
                    on_summary(summary, completed_runs, total_runs)
    return summaries


def _print_generation_benchmark_progress(
    summary: CompareSummary,
    completed_runs: int,
    total_runs: int,
) -> None:
    delay_label = summary.evaluation_delay_steps_label or str(summary.delay_steps)
    print(
        f"benchmark_result progress={completed_runs}/{total_runs} "
        f"delay_steps={delay_label} seed={summary.seed} variant={summary.variant} "
        f"run_id={summary.run_id} status={summary.status} "
        f"first_success_generation={summary.first_success_generation} "
        f"final_max_score={summary.final_max_score} score_ceiling={summary.score_ceiling} "
        f"success={summary.success}",
        flush=True,
    )


def _print_generation_benchmark_suite_report(args: argparse.Namespace) -> int:
    tasks = _parse_task_list(args.tasks)
    seeds = _parse_seed_list(args.seeds)
    variants = _parse_variants(args.variants)
    task_delay_sweeps = _parse_task_delay_sweeps(args.task_delay_sweep)
    output_dir = Path(args.output_dir)
    all_rows: list[GenerationBenchmarkRow] = []
    all_feature_records: list[CandidateFeatureRecord] = []
    all_archive_cells: list[ArchiveCellRecord] = []
    all_archive_events: list[ArchiveEventRecord] = []
    all_candidate_genomes: list[CandidateGenomeExportRecord] = []
    label = args.label or f"generation-suite-{utc_now_iso().replace(':', '').replace('-', '')}"

    print(
        f"benchmark_suite tasks={tasks} seeds={seeds} variants={variants} store={args.store} output_dir={output_dir}",
        flush=True,
    )

    for task_name in tasks:
        task_config = _load_task_benchmark_config(task_name, args.configs)
        task_args = argparse.Namespace(
            task_name=None,
            temporal_delay_steps=args.temporal_delay_steps,
            evaluation_delay_steps=args.evaluation_delay_steps,
            curriculum_enabled=args.curriculum_enabled,
            curriculum_phase_switch_generation=args.curriculum_phase_switch_generation,
            curriculum_phase_1_delays=args.curriculum_phase_1_delays,
            curriculum_phase_2_delays=args.curriculum_phase_2_delays,
            variant=None,
            generations=args.generations,
            elite_top_k=args.elite_top_k,
            population_size=args.population_size,
            online_steps=None,
            replacement_interval=None,
            metrics_interval=None,
            active_population_size=None,
        )
        task_config = _apply_overrides(task_config, task_args)
        repository: RunRepository
        if args.store == "memory":
            repository = InMemoryRepository(run_id_prefix=task_config.run.run_id_prefix)
        else:
            repository = _spacetime_repository(task_config, args.server_url, args.database_name)

        print(
            f"suite_task task={task_name} delay_steps={task_config.task.temporal_delay_steps} "
            f"evaluation_delays={evaluation_delay_label(task_config.task)} "
            f"generations={task_config.run.generations} population_size={task_config.evolution.population_size}",
            flush=True,
        )

        delay_values = task_delay_sweeps.get(task_name, [task_config.task.temporal_delay_steps])
        task_summaries = _run_generation_benchmark(
            base_config=task_config,
            repository=repository,
            seeds=seeds,
            variants=variants,
            delay_values=delay_values,
            benchmark_label=label,
            on_summary=_print_generation_benchmark_progress,
        )
        task_rows = [
            _build_generation_benchmark_row(
                repository=repository,
                summary=summary,
                generation_budget=task_config.run.generations,
                population_size=task_config.evolution.population_size,
            )
            for summary in task_summaries
        ]
        all_rows.extend(task_rows)
        all_candidate_genomes.extend(
            _collect_candidate_genome_exports(
                repository=repository,
                benchmark_label=label,
                rows=task_rows,
            )
        )
        all_feature_records.extend(
            repository.list_candidate_features(
                benchmark_label=label,
                task_name=task_name,
            )
        )
        all_archive_cells.extend(
            repository.list_archive_cells(
                benchmark_label=label,
                task_name=task_name,
            )
        )
        all_archive_events.extend(
            repository.list_archive_events(
                benchmark_label=label,
                task_name=task_name,
            )
        )
    aggregates = _build_generation_suite_aggregates(all_rows)
    markdown_summary = _render_generation_suite_markdown(label=label, rows=all_rows, aggregates=aggregates)
    for line in markdown_summary.splitlines():
        if line.startswith("|"):
            print(line, flush=True)

    export_paths = _write_generation_suite_exports(
        output_dir=output_dir,
        label=label,
        rows=all_rows,
        aggregates=aggregates,
        tasks=tasks,
        seeds=seeds,
    )
    feature_path = _write_generation_feature_exports(
        output_dir=output_dir,
        label=label,
        records=all_feature_records,
    )
    archive_cell_path = _write_archive_cell_exports(
        output_dir=output_dir,
        label=label,
        records=all_archive_cells,
    )
    archive_event_path = _write_archive_event_exports(
        output_dir=output_dir,
        label=label,
        records=all_archive_events,
    )
    candidate_genomes_path = _write_candidate_genome_exports(
        output_dir=output_dir,
        label=label,
        records=all_candidate_genomes,
    )
    print(
        "generation_suite_exports "
        f"jsonl={export_paths['jsonl']} csv={export_paths['csv']} markdown={export_paths['markdown']} "
        f"features_jsonl={feature_path} archive_cells_jsonl={archive_cell_path} "
        f"archive_events_jsonl={archive_event_path} candidate_genomes_jsonl={candidate_genomes_path}",
        flush=True,
    )
    return 0


def _resolve_compare_runs(
    repository: RunRepository,
    args: argparse.Namespace,
) -> tuple[RunRecord | None, RunRecord | None]:
    if args.stateful_run_id or args.stateless_run_id:
        stateful = repository.get_run(args.stateful_run_id) if args.stateful_run_id else None
        stateless = repository.get_run(args.stateless_run_id) if args.stateless_run_id else None
        return stateful, stateless

    runs = repository.list_runs(limit=args.search_limit)
    if args.seed is not None:
        runs = [run for run in runs if run.seed == args.seed]
    if args.task_name is not None:
        runs = [run for run in runs if run.task_name == args.task_name]

    stateful = next((run for run in runs if _variant_from_run(run) == "stateful"), None)
    stateless = next((run for run in runs if _variant_from_run(run) == "stateless"), None)
    return stateful, stateless


def _resolve_online_compare_runs(
    repository: RunRepository,
    args: argparse.Namespace,
) -> tuple[RunRecord | None, RunRecord | None]:
    runs = [run for run in repository.list_runs(limit=args.search_limit) if run.mode == "online"]
    if args.seed is not None:
        runs = [run for run in runs if run.seed == args.seed]
    if args.task_name is not None:
        runs = [run for run in runs if run.task_name == args.task_name]
    stateful = next((run for run in runs if _variant_from_run(run) == "stateful"), None)
    stateless = next((run for run in runs if _variant_from_run(run) == "stateless"), None)
    return stateful, stateless


def _build_compare_summary(repository: RunRepository, run: RunRecord) -> CompareSummary:
    config = _config_from_run(run)
    curriculum_phase_1_delays, curriculum_phase_2_delays = curriculum_phase_delay_labels(config.task)
    switch_generation = curriculum_switch_generation(config.task) if config.task.curriculum_enabled else 0
    score_ceiling = score_ceiling_for_task(config.task)
    generations = [generation for generation in repository.list_generations(run.run_id) if generation.state == "committed"]
    final_max_score = max((generation.best_score for generation in generations if generation.best_score is not None), default=None)
    first_max_generation = None
    if final_max_score is not None:
        for generation in generations:
            if generation.best_score is None:
                continue
            if math.isclose(generation.best_score, final_max_score, rel_tol=0.0, abs_tol=1e-6):
                first_max_generation = generation.generation_id
                break
    success_generation_ids = [
        generation.generation_id
        for generation in generations
        if _generation_is_success(repository, generation, score_ceiling)
    ]

    return CompareSummary(
        run_id=run.run_id,
        variant=_variant_from_run(run),
        seed=run.seed,
        task_name=run.task_name,
        delay_steps=config.task.temporal_delay_steps,
        status=run.status,
        first_max_generation=first_max_generation,
        first_success_generation=success_generation_ids[0] if success_generation_ids else None,
        final_max_score=final_max_score,
        score_ceiling=score_ceiling,
        success=bool(success_generation_ids),
        avg_score_path=_avg_score_path(generations),
        resume_hint=_resume_hint(repository.list_events(run.run_id, limit=200)),
        evaluation_delay_steps_label=evaluation_delay_label(config.task),
        curriculum_enabled=config.task.curriculum_enabled,
        curriculum_phase_1_delays=curriculum_phase_1_delays,
        curriculum_phase_2_delays=curriculum_phase_2_delays,
        curriculum_switch_generation=switch_generation,
    )


def _build_benchmark_aggregates(summaries: list[CompareSummary]) -> list[BenchmarkAggregate]:
    aggregates: list[BenchmarkAggregate] = []
    variants = sorted({summary.variant for summary in summaries})
    for delay_steps in sorted({summary.delay_steps for summary in summaries}):
        for variant in variants:
            rows = [
                summary
                for summary in summaries
                if summary.variant == variant and summary.delay_steps == delay_steps
            ]
            final_scores = [summary.final_max_score for summary in rows if summary.final_max_score is not None]
            first_max_generations = [
                float(summary.first_max_generation)
                for summary in rows
                if summary.first_max_generation is not None
            ]
            first_success_generations = [
                float(summary.first_success_generation)
                for summary in rows
                if summary.first_success_generation is not None
            ]
            aggregates.append(
                BenchmarkAggregate(
                    delay_steps=delay_steps,
                    variant=variant,
                    run_count=len(rows),
                    completed_count=sum(1 for summary in rows if summary.status == "finished"),
                    success_rate=(sum(1 for summary in rows if summary.success) / len(rows)) if rows else 0.0,
                    mean_final_max_score=(sum(final_scores) / len(final_scores)) if final_scores else None,
                    mean_first_max_generation=(
                        sum(first_max_generations) / len(first_max_generations)
                        if first_max_generations
                        else None
                    ),
                    mean_first_success_generation=(
                        sum(first_success_generations) / len(first_success_generations)
                        if first_success_generations
                        else None
                    ),
                    best_final_max_score=max(final_scores) if final_scores else None,
                )
            )
    return aggregates


def _pair_delay_summaries(summaries: list[CompareSummary]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    aggregates = _build_benchmark_aggregates(summaries)
    variants = sorted({summary.variant for summary in summaries})
    for delay_steps in sorted({summary.delay_steps for summary in summaries}):
        for left_index, left_variant in enumerate(variants):
            for right_variant in variants[left_index + 1 :]:
                left = next(
                    (
                        aggregate
                        for aggregate in aggregates
                        if aggregate.delay_steps == delay_steps and aggregate.variant == left_variant
                    ),
                    None,
                )
                right = next(
                    (
                        aggregate
                        for aggregate in aggregates
                        if aggregate.delay_steps == delay_steps and aggregate.variant == right_variant
                    ),
                    None,
                )
                if left is None or right is None:
                    continue
                rows.append(
                    {
                        "delay_steps": delay_steps,
                        "left_variant": left.variant,
                        "right_variant": right.variant,
                        "left_success_rate": left.success_rate,
                        "right_success_rate": right.success_rate,
                        "delta_success_rate": left.success_rate - right.success_rate,
                        "left_mean_first_success_generation": left.mean_first_success_generation,
                        "right_mean_first_success_generation": right.mean_first_success_generation,
                        "delta_mean_final_max_score": (left.mean_final_max_score or 0.0) - (right.mean_final_max_score or 0.0),
                    }
                )
    return rows


def _parse_task_list(raw_tasks: str) -> list[str]:
    values = [item.strip() for item in raw_tasks.split(",")]
    parsed = [item for item in values if item]
    if not parsed:
        raise ValueError("At least one task is required.")
    invalid = [task_name for task_name in parsed if task_name not in TASK_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported task(s): {', '.join(invalid)}")
    return parsed


def _parse_variants(raw_variants: str) -> list[str]:
    values = [item.strip() for item in raw_variants.split(",")]
    parsed = [item for item in values if item]
    if not parsed:
        raise ValueError("At least one variant is required.")
    invalid = [variant for variant in parsed if variant not in VARIANT_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported variant(s): {', '.join(invalid)}")
    unique_variants: list[str] = []
    for variant in parsed:
        if variant not in unique_variants:
            unique_variants.append(variant)
    return unique_variants


def _parse_task_delay_sweeps(raw_mappings: Sequence[str]) -> dict[str, list[int]]:
    parsed: dict[str, list[int]] = {}
    for raw_mapping in raw_mappings:
        if "=" not in raw_mapping:
            raise ValueError(f"Invalid task delay sweep mapping: {raw_mapping}")
        task_name, raw_values = raw_mapping.split("=", 1)
        task_name = task_name.strip()
        if task_name not in TASK_CHOICES:
            raise ValueError(f"Unsupported task in delay sweep: {task_name}")
        parsed[task_name] = _parse_delay_sweep(raw_values, default_delay=1)
    return parsed


def _load_task_benchmark_config(task_name: str, extra_paths: Sequence[str]) -> AppConfig:
    if task_name not in TASK_DEFAULT_CONFIGS:
        raise ValueError(f"No default benchmark config registered for task: {task_name}")
    paths: list[str] = []
    for path in [*TASK_DEFAULT_CONFIGS[task_name], *extra_paths]:
        if path not in paths:
            paths.append(path)
    return load_config(paths)


def _build_generation_benchmark_row(
    *,
    repository: RunRepository,
    summary: CompareSummary,
    generation_budget: int,
    population_size: int,
) -> GenerationBenchmarkRow:
    generations = [generation for generation in repository.list_generations(summary.run_id) if generation.state == "committed"]
    best_generation = _resolve_best_generation(generations, summary.final_max_score)
    success_generation = next(
        (
            generation
            for generation in generations
            if generation.generation_id == summary.first_success_generation
        ),
        None,
    )
    best_node_count, best_enabled_connection_count = _candidate_complexity(
        repository,
        summary.run_id,
        best_generation,
    )
    best_candidate_metrics = _candidate_raw_metrics(
        repository,
        summary.run_id,
        best_generation,
    )
    success_node_count, success_enabled_connection_count = _candidate_complexity(
        repository,
        summary.run_id,
        success_generation,
    )
    final_generation = generations[-1] if generations else None
    return GenerationBenchmarkRow(
        task_name=summary.task_name,
        delay_steps=summary.delay_steps,
        variant=summary.variant,
        seed=summary.seed,
        run_id=summary.run_id,
        status=summary.status,
        generation_budget=generation_budget,
        population_size=population_size,
        score_ceiling=summary.score_ceiling,
        success=summary.success,
        first_success_generation=summary.first_success_generation,
        best_generation_id=best_generation.generation_id if best_generation is not None else None,
        best_candidate_id=best_generation.best_candidate_id if best_generation is not None else None,
        final_max_score=summary.final_max_score,
        final_avg_score=final_generation.avg_score if final_generation is not None else None,
        completed_generations=len(generations),
        best_node_count=best_node_count,
        best_enabled_connection_count=best_enabled_connection_count,
        success_node_count=success_node_count,
        success_enabled_connection_count=success_enabled_connection_count,
        avg_score_path=summary.avg_score_path,
        resume_hint=summary.resume_hint,
        evaluation_delay_steps_label=summary.evaluation_delay_steps_label,
        curriculum_enabled=summary.curriculum_enabled,
        curriculum_phase_1_delays=summary.curriculum_phase_1_delays,
        curriculum_phase_2_delays=summary.curriculum_phase_2_delays,
        curriculum_switch_generation=summary.curriculum_switch_generation,
        query_accuracy=_coerce_optional_metric(best_candidate_metrics.get("query_accuracy")),
        retrieval_score=_coerce_optional_metric(best_candidate_metrics.get("retrieval_score")),
        mean_query_distance=_coerce_optional_metric(best_candidate_metrics.get("mean_query_distance")),
        distractor_load=_coerce_optional_metric(best_candidate_metrics.get("distractor_load")),
        correct_key_selected=_coerce_optional_metric(best_candidate_metrics.get("correct_key_selected")),
        correct_value_selected=_coerce_optional_metric(best_candidate_metrics.get("correct_value_selected")),
        query_key_match_score=_coerce_optional_metric(best_candidate_metrics.get("query_key_match_score")),
        value_margin=_coerce_optional_metric(best_candidate_metrics.get("value_margin")),
        distractor_competition_score=_coerce_optional_metric(best_candidate_metrics.get("distractor_competition_score")),
        store_vs_distractor_write_gap=_coerce_optional_metric(best_candidate_metrics.get("store_vs_distractor_write_gap")),
        query_value_read_strength=_coerce_optional_metric(best_candidate_metrics.get("query_value_read_strength")),
        slot_write_focus=_coerce_optional_metric(best_candidate_metrics.get("slot_write_focus")),
        slot_query_focus=_coerce_optional_metric(best_candidate_metrics.get("slot_query_focus")),
        slot_readout_selectivity=_coerce_optional_metric(best_candidate_metrics.get("slot_readout_selectivity")),
        slot_utilization=_coerce_optional_metric(best_candidate_metrics.get("slot_utilization")),
        query_slot_match_max=_coerce_optional_metric(best_candidate_metrics.get("query_slot_match_max")),
        slot_distractor_leak=_coerce_optional_metric(best_candidate_metrics.get("slot_distractor_leak")),
        mean_write_address_focus=_coerce_optional_metric(best_candidate_metrics.get("mean_write_address_focus")),
        mean_read_address_focus=_coerce_optional_metric(best_candidate_metrics.get("mean_read_address_focus")),
        write_read_address_gap=_coerce_optional_metric(best_candidate_metrics.get("write_read_address_gap")),
        readout_address_concentration=_coerce_optional_metric(best_candidate_metrics.get("readout_address_concentration")),
        store_vs_distractor_beta_gap=_coerce_optional_metric(best_candidate_metrics.get("store_vs_distractor_beta_gap")),
        query_memory_alignment=_coerce_optional_metric(best_candidate_metrics.get("query_memory_alignment")),
        delta_correction_magnitude=_coerce_optional_metric(best_candidate_metrics.get("delta_correction_magnitude")),
        mean_memory_frobenius_norm=_coerce_optional_metric(best_candidate_metrics.get("mean_memory_frobenius_norm")),
    )


def _resolve_best_generation(
    generations: list[GenerationRecord],
    final_max_score: float | None,
) -> GenerationRecord | None:
    if final_max_score is None:
        return None
    for generation in generations:
        if generation.best_score is None:
            continue
        if math.isclose(generation.best_score, final_max_score, rel_tol=0.0, abs_tol=1e-6):
            return generation
    return None


def _candidate_complexity(
    repository: RunRepository,
    run_id: str,
    generation: GenerationRecord | None,
) -> tuple[int | None, int | None]:
    if generation is None or generation.best_candidate_id is None:
        return None, None
    candidates = repository.list_candidates(run_id, generation.generation_id)
    candidate = next(
        (row for row in candidates if row.candidate_id == generation.best_candidate_id),
        None,
    )
    if candidate is None:
        return None, None
    genome = genome_model_from_blob(candidate.genome_blob)
    enabled_connections = sum(1 for conn in genome.connections if conn.enabled)
    return len(genome.nodes), enabled_connections


def _candidate_raw_metrics(
    repository: RunRepository,
    run_id: str,
    generation: GenerationRecord | None,
) -> dict[str, object]:
    if generation is None or generation.best_candidate_id is None:
        return {}
    fitness_rows = repository.list_fitness(run_id, generation.generation_id)
    matching = next(
        (row for row in fitness_rows if row.candidate_id == generation.best_candidate_id),
        None,
    )
    if matching is None:
        return {}
    return dict(matching.raw_metrics)


def _build_generation_suite_aggregates(rows: list[GenerationBenchmarkRow]) -> list[GenerationSuiteAggregate]:
    aggregates: list[GenerationSuiteAggregate] = []
    keys = sorted(
        {
            (
                row.task_name,
                row.delay_steps,
                row.evaluation_delay_steps_label,
                row.variant,
                row.curriculum_enabled,
                row.curriculum_phase_1_delays,
                row.curriculum_phase_2_delays,
                row.curriculum_switch_generation,
            )
            for row in rows
        }
    )
    for (
        task_name,
        delay_steps,
        evaluation_delay_steps_label,
        variant,
        curriculum_enabled,
        curriculum_phase_1_delays,
        curriculum_phase_2_delays,
        curriculum_switch_generation_value,
    ) in keys:
        grouped_rows = [
            row
            for row in rows
            if (
                row.task_name == task_name
                and row.delay_steps == delay_steps
                and row.evaluation_delay_steps_label == evaluation_delay_steps_label
                and row.variant == variant
                and row.curriculum_enabled == curriculum_enabled
                and row.curriculum_phase_1_delays == curriculum_phase_1_delays
                and row.curriculum_phase_2_delays == curriculum_phase_2_delays
                and row.curriculum_switch_generation == curriculum_switch_generation_value
            )
        ]
        final_scores = [row.final_max_score for row in grouped_rows if row.final_max_score is not None]
        first_success_generations = [
            float(row.first_success_generation)
            for row in grouped_rows
            if row.first_success_generation is not None
        ]
        best_node_counts = [float(row.best_node_count) for row in grouped_rows if row.best_node_count is not None]
        best_enabled_connection_counts = [
            float(row.best_enabled_connection_count)
            for row in grouped_rows
            if row.best_enabled_connection_count is not None
        ]
        query_accuracies = [float(row.query_accuracy) for row in grouped_rows if row.query_accuracy is not None]
        retrieval_scores = [float(row.retrieval_score) for row in grouped_rows if row.retrieval_score is not None]
        query_distances = [float(row.mean_query_distance) for row in grouped_rows if row.mean_query_distance is not None]
        distractor_loads = [float(row.distractor_load) for row in grouped_rows if row.distractor_load is not None]
        correct_key_selected_values = [float(row.correct_key_selected) for row in grouped_rows if row.correct_key_selected is not None]
        correct_value_selected_values = [float(row.correct_value_selected) for row in grouped_rows if row.correct_value_selected is not None]
        query_key_match_scores = [float(row.query_key_match_score) for row in grouped_rows if row.query_key_match_score is not None]
        value_margins = [float(row.value_margin) for row in grouped_rows if row.value_margin is not None]
        distractor_competition_scores = [
            float(row.distractor_competition_score)
            for row in grouped_rows
            if row.distractor_competition_score is not None
        ]
        store_vs_distractor_write_gaps = [
            float(row.store_vs_distractor_write_gap)
            for row in grouped_rows
            if row.store_vs_distractor_write_gap is not None
        ]
        query_value_read_strengths = [
            float(row.query_value_read_strength)
            for row in grouped_rows
            if row.query_value_read_strength is not None
        ]
        slot_write_focus_values = [float(row.slot_write_focus) for row in grouped_rows if row.slot_write_focus is not None]
        slot_query_focus_values = [float(row.slot_query_focus) for row in grouped_rows if row.slot_query_focus is not None]
        slot_readout_selectivity_values = [
            float(row.slot_readout_selectivity) for row in grouped_rows if row.slot_readout_selectivity is not None
        ]
        slot_utilization_values = [float(row.slot_utilization) for row in grouped_rows if row.slot_utilization is not None]
        query_slot_match_max_values = [
            float(row.query_slot_match_max) for row in grouped_rows if row.query_slot_match_max is not None
        ]
        slot_distractor_leak_values = [
            float(row.slot_distractor_leak) for row in grouped_rows if row.slot_distractor_leak is not None
        ]
        write_address_focus_values = [
            float(row.mean_write_address_focus) for row in grouped_rows if row.mean_write_address_focus is not None
        ]
        read_address_focus_values = [
            float(row.mean_read_address_focus) for row in grouped_rows if row.mean_read_address_focus is not None
        ]
        write_read_gap_values = [
            float(row.write_read_address_gap) for row in grouped_rows if row.write_read_address_gap is not None
        ]
        readout_concentration_values = [
            float(row.readout_address_concentration) for row in grouped_rows if row.readout_address_concentration is not None
        ]
        beta_gap_values = [float(row.store_vs_distractor_beta_gap) for row in grouped_rows if row.store_vs_distractor_beta_gap is not None]
        query_memory_alignment_values = [float(row.query_memory_alignment) for row in grouped_rows if row.query_memory_alignment is not None]
        delta_correction_values = [float(row.delta_correction_magnitude) for row in grouped_rows if row.delta_correction_magnitude is not None]
        memory_norm_values = [float(row.mean_memory_frobenius_norm) for row in grouped_rows if row.mean_memory_frobenius_norm is not None]
        aggregates.append(
            GenerationSuiteAggregate(
                task_name=task_name,
                delay_steps=delay_steps,
                variant=variant,
                run_count=len(grouped_rows),
                completed_count=sum(1 for row in grouped_rows if row.status == "finished"),
                success_rate=(sum(1 for row in grouped_rows if row.success) / len(grouped_rows)) if grouped_rows else 0.0,
                mean_final_max_score=(sum(final_scores) / len(final_scores)) if final_scores else None,
                mean_first_success_generation=(
                    sum(first_success_generations) / len(first_success_generations)
                    if first_success_generations
                    else None
                ),
                median_first_success_generation=(
                    float(median(first_success_generations))
                    if first_success_generations
                    else None
                ),
                mean_best_node_count=(sum(best_node_counts) / len(best_node_counts)) if best_node_counts else None,
                mean_best_enabled_connection_count=(
                    sum(best_enabled_connection_counts) / len(best_enabled_connection_counts)
                    if best_enabled_connection_counts
                    else None
                ),
                evaluation_delay_steps_label=evaluation_delay_steps_label,
                curriculum_enabled=curriculum_enabled,
                curriculum_phase_1_delays=curriculum_phase_1_delays,
                curriculum_phase_2_delays=curriculum_phase_2_delays,
                curriculum_switch_generation=curriculum_switch_generation_value,
                mean_query_accuracy=(sum(query_accuracies) / len(query_accuracies)) if query_accuracies else None,
                mean_retrieval_score=(sum(retrieval_scores) / len(retrieval_scores)) if retrieval_scores else None,
                mean_query_distance=(sum(query_distances) / len(query_distances)) if query_distances else None,
                mean_distractor_load=(sum(distractor_loads) / len(distractor_loads)) if distractor_loads else None,
                mean_correct_key_selected=(
                    (sum(correct_key_selected_values) / len(correct_key_selected_values))
                    if correct_key_selected_values
                    else None
                ),
                mean_correct_value_selected=(
                    (sum(correct_value_selected_values) / len(correct_value_selected_values))
                    if correct_value_selected_values
                    else None
                ),
                mean_query_key_match_score=(
                    (sum(query_key_match_scores) / len(query_key_match_scores))
                    if query_key_match_scores
                    else None
                ),
                mean_value_margin=(sum(value_margins) / len(value_margins)) if value_margins else None,
                mean_distractor_competition_score=(
                    (sum(distractor_competition_scores) / len(distractor_competition_scores))
                    if distractor_competition_scores
                    else None
                ),
                mean_store_vs_distractor_write_gap=(
                    (sum(store_vs_distractor_write_gaps) / len(store_vs_distractor_write_gaps))
                    if store_vs_distractor_write_gaps
                    else None
                ),
                mean_query_value_read_strength=(
                    (sum(query_value_read_strengths) / len(query_value_read_strengths))
                    if query_value_read_strengths
                    else None
                ),
                mean_slot_write_focus=(
                    (sum(slot_write_focus_values) / len(slot_write_focus_values))
                    if slot_write_focus_values
                    else None
                ),
                mean_slot_query_focus=(
                    (sum(slot_query_focus_values) / len(slot_query_focus_values))
                    if slot_query_focus_values
                    else None
                ),
                mean_slot_readout_selectivity=(
                    (sum(slot_readout_selectivity_values) / len(slot_readout_selectivity_values))
                    if slot_readout_selectivity_values
                    else None
                ),
                mean_slot_utilization=(
                    (sum(slot_utilization_values) / len(slot_utilization_values))
                    if slot_utilization_values
                    else None
                ),
                mean_query_slot_match_max=(
                    (sum(query_slot_match_max_values) / len(query_slot_match_max_values))
                    if query_slot_match_max_values
                    else None
                ),
                mean_slot_distractor_leak=(
                    (sum(slot_distractor_leak_values) / len(slot_distractor_leak_values))
                    if slot_distractor_leak_values
                    else None
                ),
                mean_write_address_focus=(
                    (sum(write_address_focus_values) / len(write_address_focus_values))
                    if write_address_focus_values
                    else None
                ),
                mean_read_address_focus=(
                    (sum(read_address_focus_values) / len(read_address_focus_values))
                    if read_address_focus_values
                    else None
                ),
                mean_write_read_address_gap=(
                    (sum(write_read_gap_values) / len(write_read_gap_values))
                    if write_read_gap_values
                    else None
                ),
                mean_readout_address_concentration=(
                    (sum(readout_concentration_values) / len(readout_concentration_values))
                    if readout_concentration_values
                    else None
                ),
                mean_store_vs_distractor_beta_gap=(sum(beta_gap_values) / len(beta_gap_values)) if beta_gap_values else None,
                mean_query_memory_alignment=(sum(query_memory_alignment_values) / len(query_memory_alignment_values)) if query_memory_alignment_values else None,
                mean_delta_correction_magnitude=(sum(delta_correction_values) / len(delta_correction_values)) if delta_correction_values else None,
                mean_memory_frobenius_norm=(sum(memory_norm_values) / len(memory_norm_values)) if memory_norm_values else None,
            )
        )
    return aggregates


def _write_generation_suite_exports(
    *,
    output_dir: Path,
    label: str,
    rows: list[GenerationBenchmarkRow],
    aggregates: list[GenerationSuiteAggregate],
    tasks: list[str],
    seeds: list[int],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{label}.jsonl"
    csv_path = output_dir / f"{label}.csv"
    markdown_path = output_dir / f"{label}.md"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(stable_json_dumps(asdict(row)))
            handle.write("\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_name",
                "delay_steps",
                "evaluation_delay_steps_label",
                "curriculum_enabled",
                "curriculum_phase_1_delays",
                "curriculum_phase_2_delays",
                "curriculum_switch_generation",
                "variant",
                "run_count",
                "completed_count",
                "success_rate",
                "mean_final_max_score",
                "mean_first_success_generation",
                "median_first_success_generation",
                "mean_best_node_count",
                "mean_best_enabled_connection_count",
                "mean_query_accuracy",
                "mean_retrieval_score",
                "mean_query_distance",
                "mean_distractor_load",
                "mean_correct_key_selected",
                "mean_correct_value_selected",
                "mean_query_key_match_score",
                "mean_value_margin",
                "mean_distractor_competition_score",
                "mean_store_vs_distractor_write_gap",
                "mean_query_value_read_strength",
                "mean_slot_write_focus",
                "mean_slot_query_focus",
                "mean_slot_readout_selectivity",
                "mean_slot_utilization",
                "mean_query_slot_match_max",
                "mean_slot_distractor_leak",
                "mean_write_address_focus",
                "mean_read_address_focus",
                "mean_write_read_address_gap",
                "mean_readout_address_concentration",
                "mean_store_vs_distractor_beta_gap",
                "mean_query_memory_alignment",
                "mean_delta_correction_magnitude",
                "mean_memory_frobenius_norm",
            ],
        )
        writer.writeheader()
        for aggregate in aggregates:
            writer.writerow(asdict(aggregate))

    markdown_path.write_text(
        _render_generation_suite_markdown(label=label, rows=rows, aggregates=aggregates, tasks=tasks, seeds=seeds),
        encoding="utf-8",
    )

    return {
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
    }


def _feature_export_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"{label}.candidate-features.jsonl"


def _candidate_genome_export_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"{label}.candidate-genomes.jsonl"


def _write_generation_feature_exports(
    *,
    output_dir: Path,
    label: str,
    records: list[CandidateFeatureRecord],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _feature_export_path(output_dir, label)
    write_feature_records_jsonl(path, records)
    return str(path)


def _collect_candidate_genome_exports(
    *,
    repository: RunRepository,
    benchmark_label: str,
    rows: Sequence[GenerationBenchmarkRow],
) -> list[CandidateGenomeExportRecord]:
    exports: list[CandidateGenomeExportRecord] = []
    seen_candidate_ids: set[str] = set()
    for row in rows:
        if row.best_generation_id is None or row.best_candidate_id is None:
            continue
        if row.best_candidate_id in seen_candidate_ids:
            continue
        candidates = repository.list_candidates(row.run_id, row.best_generation_id)
        matching = next((cand for cand in candidates if cand.candidate_id == row.best_candidate_id), None)
        if matching is None:
            continue
        exports.append(
            CandidateGenomeExportRecord(
                benchmark_label=benchmark_label,
                run_id=row.run_id,
                generation_id=row.best_generation_id,
                candidate_id=row.best_candidate_id,
                task_name=row.task_name,
                variant=row.variant,
                seed=row.seed,
                genome_blob=matching.genome_blob,
            )
        )
        seen_candidate_ids.add(row.best_candidate_id)
    return exports


def _write_candidate_genome_exports(
    *,
    output_dir: Path,
    label: str,
    records: list[CandidateGenomeExportRecord],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _candidate_genome_export_path(output_dir, label)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json_dumps(asdict(record)))
            handle.write("\n")
    return str(path)


def _archive_cell_export_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"{label}.archive-cells.jsonl"


def _archive_event_export_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"{label}.archive-events.jsonl"


def _write_archive_cell_exports(
    *,
    output_dir: Path,
    label: str,
    records: list[ArchiveCellRecord],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _archive_cell_export_path(output_dir, label)
    write_archive_cells_jsonl(path, records)
    return str(path)


def _write_archive_event_exports(
    *,
    output_dir: Path,
    label: str,
    records: list[ArchiveEventRecord],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _archive_event_export_path(output_dir, label)
    write_archive_events_jsonl(path, records)
    return str(path)


def _render_generation_suite_markdown(
    *,
    label: str,
    rows: list[GenerationBenchmarkRow],
    aggregates: list[GenerationSuiteAggregate],
    tasks: list[str] | None = None,
    seeds: list[int] | None = None,
) -> str:
    header = [
        f"# Generation Benchmark Suite: {label}",
        "",
        f"- tasks: {', '.join(tasks or sorted({row.task_name for row in rows}))}",
        f"- seeds: {', '.join(str(seed) for seed in (seeds or sorted({row.seed for row in rows})))}",
        "",
        "| task | delay | variant | runs | success_rate | mean_final_max_score | mean_first_success_generation | median_first_success_generation | mean_best_nodes | mean_best_enabled_conns |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    curriculum_rows = [aggregate for aggregate in aggregates if aggregate.curriculum_enabled]
    if curriculum_rows:
        curriculum_summaries = sorted(
            {
                f"{aggregate.curriculum_phase_1_delays}->{aggregate.curriculum_phase_2_delays}@g{aggregate.curriculum_switch_generation}"
                for aggregate in curriculum_rows
            }
        )
        header[4:4] = [f"- curriculum: {', '.join(curriculum_summaries)}", ""]
    table_rows = [
        "| "
        + " | ".join(
            [
                aggregate.task_name,
                aggregate.evaluation_delay_steps_label or str(aggregate.delay_steps),
                aggregate.variant,
                str(aggregate.run_count),
                f"{aggregate.success_rate:.3f}",
                _format_optional_float(aggregate.mean_final_max_score, precision=6),
                _format_optional_float(aggregate.mean_first_success_generation, precision=2),
                _format_optional_float(aggregate.median_first_success_generation, precision=2),
                _format_optional_float(aggregate.mean_best_node_count, precision=2),
                _format_optional_float(aggregate.mean_best_enabled_connection_count, precision=2),
            ]
        )
        + " |"
        for aggregate in aggregates
    ]
    sections = [*header, *table_rows, ""]
    retrieval_aggregates = [
        aggregate
        for aggregate in aggregates
        if aggregate.mean_query_accuracy is not None
        or aggregate.mean_retrieval_score is not None
        or aggregate.mean_query_distance is not None
        or aggregate.mean_distractor_load is not None
    ]
    if retrieval_aggregates:
        sections.extend(
            [
                "## Retrieval Metrics",
                "",
                "| task | delay | variant | mean_query_accuracy | mean_retrieval_score | mean_query_distance | mean_distractor_load |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        sections.extend(
            [
                "| "
                + " | ".join(
                    [
                        aggregate.task_name,
                        aggregate.evaluation_delay_steps_label or str(aggregate.delay_steps),
                        aggregate.variant,
                        _format_optional_float(aggregate.mean_query_accuracy, precision=3),
                        _format_optional_float(aggregate.mean_retrieval_score, precision=3),
                        _format_optional_float(aggregate.mean_query_distance, precision=3),
                        _format_optional_float(aggregate.mean_distractor_load, precision=3),
                    ]
                )
                + " |"
                for aggregate in retrieval_aggregates
            ]
        )
        sections.append("")
    retrieval_diagnostic_aggregates = [
        aggregate
        for aggregate in aggregates
        if aggregate.mean_correct_key_selected is not None
        or aggregate.mean_correct_value_selected is not None
        or aggregate.mean_query_key_match_score is not None
        or aggregate.mean_value_margin is not None
        or aggregate.mean_distractor_competition_score is not None
    ]
    if retrieval_diagnostic_aggregates:
        sections.extend(
            [
                "## Retrieval Diagnostics",
                "",
                "| task | delay | variant | mean_correct_key_selected | mean_correct_value_selected | mean_query_key_match_score | mean_value_margin | mean_distractor_competition_score |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        sections.extend(
            [
                "| "
                + " | ".join(
                    [
                        aggregate.task_name,
                        aggregate.evaluation_delay_steps_label or str(aggregate.delay_steps),
                        aggregate.variant,
                        _format_optional_float(aggregate.mean_correct_key_selected, precision=3),
                        _format_optional_float(aggregate.mean_correct_value_selected, precision=3),
                        _format_optional_float(aggregate.mean_query_key_match_score, precision=3),
                        _format_optional_float(aggregate.mean_value_margin, precision=3),
                        _format_optional_float(aggregate.mean_distractor_competition_score, precision=3),
                    ]
                )
                + " |"
                for aggregate in retrieval_diagnostic_aggregates
            ]
        )
        sections.append("")
        if any(
            aggregate.mean_store_vs_distractor_write_gap is not None
            or aggregate.mean_query_value_read_strength is not None
            for aggregate in retrieval_diagnostic_aggregates
        ):
            sections.extend(
                [
                    "## KV Selectivity Diagnostics",
                    "",
                    "| task | delay | variant | mean_store_vs_distractor_write_gap | mean_query_value_read_strength |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            sections.extend(
                [
                    "| "
                    + " | ".join(
                        [
                            aggregate.task_name,
                            aggregate.evaluation_delay_steps_label or str(aggregate.delay_steps),
                            aggregate.variant,
                            _format_optional_float(aggregate.mean_store_vs_distractor_write_gap, precision=3),
                            _format_optional_float(aggregate.mean_query_value_read_strength, precision=3),
                        ]
                    )
                    + " |"
                    for aggregate in retrieval_diagnostic_aggregates
                ]
            )
            sections.append("")
        if any(
            aggregate.mean_slot_write_focus is not None
            or aggregate.mean_slot_query_focus is not None
            or aggregate.mean_slot_readout_selectivity is not None
            or aggregate.mean_query_slot_match_max is not None
            or aggregate.mean_read_address_focus is not None
            for aggregate in retrieval_diagnostic_aggregates
        ):
            sections.extend(
                [
                    "## Slot Retrieval Diagnostics",
                    "",
                    "| task | delay | variant | mean_slot_write_focus | mean_slot_query_focus | mean_slot_readout_selectivity | mean_slot_utilization | mean_query_slot_match_max | mean_slot_distractor_leak | mean_write_address_focus | mean_read_address_focus | mean_write_read_address_gap | mean_readout_address_concentration |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            sections.extend(
                [
                    "| "
                    + " | ".join(
                        [
                            aggregate.task_name,
                            aggregate.evaluation_delay_steps_label or str(aggregate.delay_steps),
                            aggregate.variant,
                            _format_optional_float(aggregate.mean_slot_write_focus, precision=3),
                            _format_optional_float(aggregate.mean_slot_query_focus, precision=3),
                            _format_optional_float(aggregate.mean_slot_readout_selectivity, precision=3),
                            _format_optional_float(aggregate.mean_slot_utilization, precision=3),
                            _format_optional_float(aggregate.mean_query_slot_match_max, precision=3),
                            _format_optional_float(aggregate.mean_slot_distractor_leak, precision=3),
                            _format_optional_float(aggregate.mean_write_address_focus, precision=3),
                            _format_optional_float(aggregate.mean_read_address_focus, precision=3),
                            _format_optional_float(aggregate.mean_write_read_address_gap, precision=3),
                            _format_optional_float(aggregate.mean_readout_address_concentration, precision=3),
                        ]
                    )
                    + " |"
                    for aggregate in retrieval_diagnostic_aggregates
                ]
            )
            sections.append("")
        if any(
            aggregate.mean_store_vs_distractor_beta_gap is not None
            or aggregate.mean_query_memory_alignment is not None
            or aggregate.mean_delta_correction_magnitude is not None
            or aggregate.mean_memory_frobenius_norm is not None
            for aggregate in retrieval_diagnostic_aggregates
        ):
            sections.extend(
                [
                    "## Delta Memory Diagnostics",
                    "",
                    "| task | delay | variant | mean_store_vs_distractor_beta_gap | mean_query_memory_alignment | mean_delta_correction_magnitude | mean_memory_frobenius_norm |",
                    "| --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            sections.extend(
                [
                    "| "
                    + " | ".join(
                        [
                            aggregate.task_name,
                            aggregate.evaluation_delay_steps_label or str(aggregate.delay_steps),
                            aggregate.variant,
                            _format_optional_float(aggregate.mean_store_vs_distractor_beta_gap, precision=3),
                            _format_optional_float(aggregate.mean_query_memory_alignment, precision=3),
                            _format_optional_float(aggregate.mean_delta_correction_magnitude, precision=3),
                            _format_optional_float(aggregate.mean_memory_frobenius_norm, precision=3),
                        ]
                    )
                    + " |"
                    for aggregate in retrieval_diagnostic_aggregates
                ]
            )
            sections.append("")
    return "\n".join(sections)


def _format_optional_float(value: float | None, *, precision: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _coerce_optional_metric(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg_score_path(generations: list[GenerationRecord]) -> str:
    scores = [generation.avg_score for generation in generations if generation.avg_score is not None]
    if not scores:
        return "n/a"
    first = scores[0]
    middle = scores[len(scores) // 2]
    last = scores[-1]
    return f"{first:.3f}->{middle:.3f}->{last:.3f}"


def _resume_hint(events: list[EventRecord]) -> str:
    resume_events = [event for event in events if event.type == "run_resumed"]
    if not resume_events:
        return "not_resumed"
    latest_payload = json.loads(resume_events[0].payload_json)
    return f"resumed x{len(resume_events)} next_generation={latest_payload.get('next_generation_id')}"


def _variant_from_run(run: RunRecord) -> str:
    payload = json.loads(run.config_json)
    return payload.get("run", {}).get("variant", "stateful")


def _config_from_run(run: RunRecord) -> AppConfig:
    payload = json.loads(run.config_json)
    task_payload = dict(payload.get("task", {}))
    task_payload.setdefault("name", run.task_name)
    payload = {**payload, "task": task_payload}
    return AppConfig.from_dict(payload)


def _generation_is_success(
    repository: RunRepository,
    generation: GenerationRecord,
    score_ceiling: float,
) -> bool:
    if generation.best_candidate_id is not None:
        fitness_rows = repository.list_fitness(generation.run_id, generation.generation_id)
        matching = next(
            (row for row in fitness_rows if row.candidate_id == generation.best_candidate_id),
            None,
        )
        if matching is not None:
            return resolve_success(matching.raw_metrics, matching.score, score_ceiling)
    if generation.best_score is None:
        return False
    return resolve_success(None, generation.best_score, score_ceiling)


def _parse_seed_list(raw_seeds: str) -> list[int]:
    seeds = [item.strip() for item in raw_seeds.split(",")]
    parsed = [int(item) for item in seeds if item]
    if not parsed:
        raise ValueError("At least one seed is required.")
    return parsed


def _parse_delay_sweep(raw_delay_sweep: str | None, default_delay: int) -> list[int]:
    if raw_delay_sweep is None:
        return [default_delay]
    values = [item.strip() for item in raw_delay_sweep.split(",")]
    parsed = [int(item) for item in values if item]
    if not parsed:
        raise ValueError("At least one delay value is required.")
    return parsed


def _parse_optional_delay_list(raw_delay_values: str | None) -> tuple[int, ...]:
    if raw_delay_values is None:
        return ()
    values = [item.strip() for item in raw_delay_values.split(",")]
    parsed = tuple(int(item) for item in values if item)
    if not parsed:
        raise ValueError("At least one evaluation delay value is required.")
    return parsed


def _print_online_status(repository: SpacetimeRepository, run_id: str) -> int:
    run = repository.get_run(run_id)
    if run is None:
        print(f"Run not found: {run_id}")
        return 1
    state = repository.get_online_state(run_id)
    active_candidates = repository.list_active_candidates(run_id)
    jobs = repository.list_evaluation_jobs(run_id, limit=500)
    hall_of_fame = repository.list_hall_of_fame(run_id, limit=20)
    metrics = repository.list_online_metrics(run_id, limit=1)
    metric = metrics[0] if metrics else None
    print(
        f"run_id={run.run_id} mode={run.mode} task={run.task_name} seed={run.seed} status={run.status} "
        f"step={state.step if state else 0} active_population={len([c for c in active_candidates if c.status != 'retired'])} "
        f"queued_jobs={sum(1 for job in jobs if job.status == 'queued')} "
        f"claimed_jobs={sum(1 for job in jobs if job.status == 'claimed')} "
        f"hall_of_fame={len(hall_of_fame)}"
    )
    if metric is not None:
        print(
            f"rolling_best_score={metric.rolling_best_score:.6f} rolling_avg_score={metric.rolling_avg_score:.6f} "
            f"replacement_count={metric.replacement_count} success_rate_window={metric.success_rate_window:.3f}"
        )
    for candidate in active_candidates[: min(10, len(active_candidates))]:
        print(
            f"slot={candidate.slot_index} candidate_id={candidate.candidate_id} status={candidate.status} "
            f"rolling_score={candidate.rolling_score:.6f} eval_count={candidate.eval_count} birth_step={candidate.birth_step}"
        )
    return 0


def _print_online_compare_report(repository: SpacetimeRepository, args: argparse.Namespace) -> int:
    stateful_run, stateless_run = _resolve_online_compare_runs(repository, args)
    if stateful_run is None or stateless_run is None:
        print("Could not find both stateful and stateless online runs for the requested comparison.")
        return 1
    score_ceiling = score_ceiling_for_task(_config_from_run(stateful_run).task)
    for summary in (
        build_online_compare_summary(repository, stateful_run, score_ceiling),
        build_online_compare_summary(repository, stateless_run, score_ceiling),
    ):
        print(
            f"variant={summary.variant} run_id={summary.run_id} status={summary.status} steps={summary.steps} "
            f"success_observed={summary.success_observed} "
            f"rolling_best_score={summary.rolling_best_score:.6f} rolling_avg_score={summary.rolling_avg_score:.6f} "
            f"replacement_count={summary.replacement_count} success_rate_window={summary.success_rate_window:.3f} "
            f"time_to_first_success={summary.time_to_first_success} "
            f"replacements_until_first_success={summary.replacements_until_first_success} "
            f"hall_of_fame_size={summary.hall_of_fame_size} hall_of_fame_growth={summary.hall_of_fame_growth} "
            f"resumed_jobs={summary.resumed_jobs}"
        )
    return 0


def _print_online_benchmark_report(args: argparse.Namespace) -> int:
    initial_config = _load_runtime_config(args.configs)
    config = _apply_overrides(initial_config, args)
    config = replace(config, run=replace(config.run, mode="online"))
    repository = (
        InMemoryRepository(run_id_prefix=config.run.run_id_prefix)
        if args.store == "memory"
        else _spacetime_repository(config, args.server_url, args.database_name)
    )
    summaries = run_online_benchmark(
        base_config=config,
        repository=repository,
        seeds=_parse_seed_list(args.seeds),
    )
    for summary in summaries:
        print(
            f"benchmark_online_result seed={summary.seed} variant={summary.variant} run_id={summary.run_id} status={summary.status} "
            f"success_observed={summary.success_observed} "
            f"steps={summary.steps} rolling_best_score={summary.rolling_best_score:.6f} "
            f"rolling_avg_score={summary.rolling_avg_score:.6f} replacement_count={summary.replacement_count} "
            f"success_rate_window={summary.success_rate_window:.3f} "
            f"time_to_first_success={summary.time_to_first_success} "
            f"replacements_until_first_success={summary.replacements_until_first_success} "
            f"hall_of_fame_size={summary.hall_of_fame_size} hall_of_fame_growth={summary.hall_of_fame_growth} "
            f"resumed_jobs={summary.resumed_jobs}"
        )
    for seed_summary in _pair_online_seed_summaries(summaries):
        print(
            f"seed_compare seed={seed_summary['seed']} "
            f"stateful_success_observed={seed_summary['stateful_success_observed']} "
            f"stateless_success_observed={seed_summary['stateless_success_observed']} "
            f"stateful_final_best_score={seed_summary['stateful_final_best_score']:.6f} "
            f"stateless_final_best_score={seed_summary['stateless_final_best_score']:.6f} "
            f"stateful_time_to_first_success={seed_summary['stateful_time_to_first_success']} "
            f"stateless_time_to_first_success={seed_summary['stateless_time_to_first_success']} "
            f"stateful_replacements_until_first_success={seed_summary['stateful_replacements_until_first_success']} "
            f"stateless_replacements_until_first_success={seed_summary['stateless_replacements_until_first_success']}"
        )
    for aggregate in build_online_benchmark_aggregates(summaries):
        mean_first_success = (
            f"{aggregate.mean_time_to_first_success:.2f}"
            if aggregate.mean_time_to_first_success is not None
            else "n/a"
        )
        mean_replacements_until_success = (
            f"{aggregate.mean_replacements_until_first_success:.2f}"
            if aggregate.mean_replacements_until_first_success is not None
            else "n/a"
        )
        print(
            f"aggregate variant={aggregate.variant} runs={aggregate.run_count} "
            f"run_success_rate={aggregate.run_success_rate:.3f} "
            f"mean_final_best_score={aggregate.mean_final_best_score:.6f} "
            f"mean_final_rolling_avg_score={aggregate.mean_final_rolling_avg_score:.6f} "
            f"mean_success_rate_window={aggregate.mean_success_rate_window:.3f} "
            f"mean_replacement_count={aggregate.mean_replacement_count:.2f} "
            f"mean_time_to_first_success={mean_first_success} "
            f"mean_replacements_until_first_success={mean_replacements_until_success} "
            f"mean_hall_of_fame_size={aggregate.mean_hall_of_fame_size:.2f} "
            f"mean_hall_of_fame_growth={aggregate.mean_hall_of_fame_growth:.2f}"
        )
    return 0


def _pair_online_seed_summaries(summaries: list) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in sorted({summary.seed for summary in summaries}):
        stateful = next(
            (summary for summary in summaries if summary.seed == seed and summary.variant == "stateful"),
            None,
        )
        stateless = next(
            (summary for summary in summaries if summary.seed == seed and summary.variant == "stateless"),
            None,
        )
        if stateful is None or stateless is None:
            continue
        rows.append(
            {
                "seed": seed,
                "stateful_success_observed": stateful.success_observed,
                "stateless_success_observed": stateless.success_observed,
                "stateful_final_best_score": stateful.rolling_best_score,
                "stateless_final_best_score": stateless.rolling_best_score,
                "stateful_time_to_first_success": stateful.time_to_first_success,
                "stateless_time_to_first_success": stateless.time_to_first_success,
                "stateful_replacements_until_first_success": stateful.replacements_until_first_success,
                "stateless_replacements_until_first_success": stateless.replacements_until_first_success,
            }
        )
    return rows


def _load_search_space_records(args: argparse.Namespace) -> list[CandidateFeatureRecord]:
    if args.store == "memory":
        feature_path = _feature_export_path(Path(args.output_dir), args.benchmark_label)
        return load_feature_records_from_jsonl(feature_path)

    config = _load_runtime_config(DEFAULT_CONFIGS)
    repository = _spacetime_repository(config, args.server_url, args.database_name)
    return repository.list_candidate_features(
        benchmark_label=args.benchmark_label,
        task_name=args.task,
        variant=args.variant,
        delay_steps=args.delay,
    )


def _print_search_space_report(args: argparse.Namespace) -> int:
    records = _load_search_space_records(args)
    records = filter_feature_records(
        records,
        benchmark_label=args.benchmark_label,
        task_name=args.task,
        variant=args.variant,
        delay_steps=args.delay,
        curriculum_phase=args.curriculum_phase,
    )
    if not records:
        print("No candidate features found for the requested filter.")
        return 1
    print(
        render_search_space_report(records, curriculum_phase_filter=args.curriculum_phase),
        end="",
    )
    return 0


def _parse_label_list(raw_labels: str) -> list[str]:
    labels = [item.strip() for item in raw_labels.split(",")]
    parsed = [item for item in labels if item]
    if not parsed:
        raise ValueError("At least one benchmark label is required.")
    return parsed


def _print_cross_label_search_space_report(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    labels = _parse_label_list(args.benchmark_labels)
    records_by_label: dict[str, list[CandidateFeatureRecord]] = {}
    snapshot_fallback = False
    snapshots = []
    missing_labels: list[str] = []
    for label in labels:
        feature_path = _feature_export_path(output_dir, label)
        if feature_path.exists():
            records = load_feature_records_from_jsonl(feature_path)
            records_by_label[label] = filter_feature_records(
                records,
                benchmark_label=label,
                task_name=args.task,
                variant=args.variant,
                delay_steps=args.delay,
                curriculum_phase=args.curriculum_phase,
            )
            continue
        snapshot_fallback = True
        markdown_path = output_dir / f"{label}-search-space.md"
        if markdown_path.exists():
            snapshots.append(
                load_search_space_snapshot_from_markdown(
                    markdown_path,
                    label=label,
                )
            )
        else:
            missing_labels.append(label)
    if snapshot_fallback:
        for label, records in records_by_label.items():
            feature_means = {
                feature: (sum(getattr(record, feature) for record in records) / len(records))
                for feature in SUMMARY_FEATURES
            } if records else {}
            snapshots.append(
                SearchSpaceSnapshot(
                    label=label,
                    candidate_count=len(records),
                    feature_means=feature_means,
                    hints=derive_search_space_hints(records),
                )
            )
        snapshots.sort(key=lambda snapshot: labels.index(snapshot.label))
        report = render_cross_label_snapshot_report(
            snapshots,
            top_hint_count=max(1, args.top_hints),
        )
    else:
        report = render_cross_label_search_space_report(
            records_by_label,
            top_hint_count=max(1, args.top_hints),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / args.report_name
    if missing_labels:
        report = (
            report.rstrip()
            + "\n\n## Missing Labels\n\n"
            + "\n".join(f"- {label}: no candidate-feature export or search-space markdown found" for label in missing_labels)
            + "\n"
        )
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote cross-label search-space report: {report_path}")
    return 0


def _load_archive_records(args: argparse.Namespace) -> tuple[list[ArchiveCellRecord], list[ArchiveEventRecord], list[CandidateFeatureRecord]]:
    if args.store == "memory":
        output_dir = Path(args.output_dir)
        archive_cells = load_archive_cells_from_jsonl(_archive_cell_export_path(output_dir, args.benchmark_label))
        archive_events = load_archive_events_from_jsonl(_archive_event_export_path(output_dir, args.benchmark_label))
        feature_records = load_feature_records_from_jsonl(_feature_export_path(output_dir, args.benchmark_label))
        return archive_cells, archive_events, feature_records

    config = _load_runtime_config(DEFAULT_CONFIGS)
    repository = _spacetime_repository(config, args.server_url, args.database_name)
    archive_cells = repository.list_archive_cells(
        benchmark_label=args.benchmark_label,
        task_name=args.task,
        variant=args.variant,
        delay_steps=args.delay,
        qd_profile=args.qd_profile,
    )
    archive_events = repository.list_archive_events(
        benchmark_label=args.benchmark_label,
        task_name=args.task,
        variant=args.variant,
        delay_steps=args.delay,
        qd_profile=args.qd_profile,
    )
    feature_records = repository.list_candidate_features(
        benchmark_label=args.benchmark_label,
        task_name=args.task,
        variant=args.variant,
        delay_steps=args.delay,
    )
    return archive_cells, archive_events, feature_records


def _print_archive_report(args: argparse.Namespace) -> int:
    archive_cells, archive_events, feature_records = _load_archive_records(args)
    archive_cells = filter_archive_cells(
        archive_cells,
        benchmark_label=args.benchmark_label,
        task_name=args.task,
        variant=args.variant,
        delay_steps=args.delay,
        qd_profile=args.qd_profile,
        curriculum_phase=args.curriculum_phase,
    )
    feature_records = filter_feature_records(
        feature_records,
        benchmark_label=args.benchmark_label,
        task_name=args.task,
        variant=args.variant,
        delay_steps=args.delay,
        curriculum_phase=args.curriculum_phase,
    )
    if args.curriculum_phase is None:
        archive_events = filter_archive_events(
            archive_events,
            benchmark_label=args.benchmark_label,
            task_name=args.task,
            variant=args.variant,
            delay_steps=args.delay,
            qd_profile=args.qd_profile,
        )
    else:
        archive_events = []
    if not archive_cells:
        print("No archive cells found for the requested filter.")
        return 1
    print(
        render_archive_report(
            archive_cells,
            feature_records=feature_records,
            archive_events=archive_events,
            curriculum_phase_filter=args.curriculum_phase,
        ),
        end="",
    )
    return 0


def _print_curriculum_boundary_report(args: argparse.Namespace) -> int:
    labels = [label.strip() for label in args.benchmark_labels.split(",") if label.strip()]
    if not labels:
        print("At least one benchmark label is required.")
        return 1
    variants = _parse_variants(args.variants)
    print(
        render_curriculum_boundary_report(
            output_dir=Path(args.output_dir),
            benchmark_labels=labels,
            task_name=args.task,
            variants=variants,
            focus_variant=args.focus_variant,
        ),
        end="",
    )
    return 0


def _print_retrieval_trace_report(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)

    # Load candidate features.
    if args.store == "memory":
        feature_path = _feature_export_path(output_dir, args.benchmark_label)
        records = load_feature_records_from_jsonl(feature_path)
    else:
        config = _load_runtime_config(DEFAULT_CONFIGS)
        repository = _spacetime_repository(config, args.server_url, args.database_name)
        records = repository.list_candidate_features(
            benchmark_label=args.benchmark_label,
        )

    if not records:
        print("No candidate feature records found for the requested benchmark label.")
        return 1

    # Pick candidate by ID or top scorer.
    if args.candidate_id:
        matches = [r for r in records if r.candidate_id == args.candidate_id]
        if not matches:
            print(f"Candidate {args.candidate_id} not found in {len(records)} records.")
            return 1
        chosen = matches[0]
    else:
        chosen = max(records, key=lambda r: r.final_max_score)
        print(f"Using top scorer: candidate_id={chosen.candidate_id} score={chosen.final_max_score:.4f}")

    genome = None
    if args.store == "memory":
        genome_path = _candidate_genome_export_path(output_dir, args.benchmark_label)
        if genome_path.exists():
            raw_rows = [json.loads(line) for line in genome_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            matching = next(
                (row for row in raw_rows if row.get("candidate_id") == chosen.candidate_id),
                None,
            )
            if matching is not None and str(matching.get("genome_blob", "")).strip():
                genome = genome_model_from_blob(str(matching["genome_blob"]))
        if genome is None:
            print(
                "Genome blob not available in local artifacts. "
                f"Expected candidate mapping in {genome_path}. "
                "Re-run benchmark-suite to generate *.candidate-genomes.jsonl, "
                "or use --store spacetimedb."
            )
            return 1
    else:
        config = _load_runtime_config(DEFAULT_CONFIGS)
        repo = _spacetime_repository(config, args.server_url, args.database_name)
        elites = repo.list_elites(chosen.run_id, limit=50)
        for elite in elites:
            if elite.candidate_id == chosen.candidate_id:
                genome = genome_model_from_blob(elite.frozen_genome_blob)
                break
        if genome is None:
            candidates = repo.list_candidates(chosen.run_id, chosen.generation)
            for cand in candidates:
                if cand.candidate_id == chosen.candidate_id:
                    genome = genome_model_from_blob(cand.genome_blob)
                    break

    if genome is None:
        print("Failed to resolve genome blob for the requested candidate.")
        return 1

    result = run_retrieval_trace(
        genome,
        delay_steps=args.delay,
        profile=args.profile,
        sample_index=args.sample_index,
        variant=chosen.variant,
    )
    report_path = write_trace_report(
        result,
        output_dir=str(output_dir),
        label=chosen.candidate_id,
    )
    print(render_trace_report(result, label=chosen.candidate_id), end="")
    print(f"\nWrote retrieval trace report: {report_path}")
    return 0


def _print_fitness_landscape_report(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)

    if args.store == "memory":
        feature_path = _feature_export_path(output_dir, args.benchmark_label)
        records = load_feature_records_from_jsonl(feature_path)
    else:
        config = _load_runtime_config(DEFAULT_CONFIGS)
        repository = _spacetime_repository(config, args.server_url, args.database_name)
        records = repository.list_candidate_features(
            benchmark_label=args.benchmark_label,
            task_name=args.task,
            variant=args.variant,
            delay_steps=args.delay,
        )

    records = filter_feature_records(
        records,
        benchmark_label=args.benchmark_label,
        task_name=args.task,
        variant=args.variant,
        delay_steps=args.delay,
    )
    if not records:
        print("No candidate features found for the requested filter.")
        return 1

    result = analyze_fitness_landscape(
        records,
        label=args.benchmark_label,
        top_k=args.top_k,
        num_bins=args.bins,
    )
    report_path = write_landscape_report(result, output_dir=str(output_dir))
    print(render_landscape_report(result), end="")
    print(f"\nWrote fitness landscape report: {report_path}")
    return 0


def _print_retrieval_trace_sweep_report(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    records: list[CandidateFeatureRecord] | list[SweepCandidate]
    genomes_by_candidate: dict[str, GenomeModel]
    candidate_source = "spacetimedb"
    try:
        if args.store == "memory":
            local_inputs = resolve_local_sweep_inputs(
                output_dir=output_dir,
                benchmark_label=args.benchmark_label,
                task_name=args.task,
                variant=args.variant,
            )
            records = list(local_inputs.candidates)
            genomes_by_candidate = dict(local_inputs.genomes_by_candidate)
            candidate_source = local_inputs.source_used
            print(
                "Using local artifact source "
                f"'{local_inputs.source_used}' for candidate discovery."
            )
        else:
            config = _load_runtime_config(DEFAULT_CONFIGS)
            repository = _spacetime_repository(config, args.server_url, args.database_name)
            records = repository.list_candidate_features(
                benchmark_label=args.benchmark_label,
                task_name=args.task,
                variant=args.variant,
            )
            filtered_records = filter_feature_records(
                records,
                benchmark_label=args.benchmark_label,
                task_name=args.task,
                variant=args.variant,
            )
            if not filtered_records:
                print("No candidate feature records found for retrieval-trace sweep.")
                return 1
            records = filtered_records
            genomes_by_candidate = {}
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    if args.store != "memory":
        config = _load_runtime_config(DEFAULT_CONFIGS)
        repository = _spacetime_repository(config, args.server_url, args.database_name)
        ranked = sorted(records, key=lambda r: r.final_max_score, reverse=True)  # type: ignore[attr-defined]
        for record in ranked[: max(1, int(args.top_k_candidates))]:
            genome = None
            elites = repository.list_elites(record.run_id, limit=50)
            for elite in elites:
                if elite.candidate_id == record.candidate_id:
                    genome = genome_model_from_blob(elite.frozen_genome_blob)
                    break
            if genome is None:
                candidates = repository.list_candidates(record.run_id, record.generation)
                for candidate in candidates:
                    if candidate.candidate_id == record.candidate_id:
                        genome = genome_model_from_blob(candidate.genome_blob)
                        break
            if genome is not None:
                genomes_by_candidate[record.candidate_id] = genome

    result = run_retrieval_trace_sweep(
        benchmark_label=args.benchmark_label,
        task_name=args.task,
        variant=args.variant,
        candidate_records=records,
        genomes_by_candidate=genomes_by_candidate,
        top_k_candidates=args.top_k_candidates,
        episodes_per_candidate=None if args.episodes_per_candidate <= 0 else args.episodes_per_candidate,
    )
    result = replace(result, candidate_source=candidate_source)
    report_path = write_retrieval_trace_sweep_report(
        result,
        output_dir=output_dir,
    )
    print(report_path.read_text(encoding="utf-8"), end="")
    print(f"\nWrote retrieval trace sweep report: {report_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        initial_config = _load_runtime_config(args.configs)
        if args.store == "memory":
            repository = InMemoryRepository(run_id_prefix=initial_config.run.run_id_prefix)
            config = _apply_overrides(initial_config, args)
        else:
            repository = _spacetime_repository(initial_config, args.server_url, args.database_name)
            if args.resume_run_id is not None:
                existing_run = repository.get_run(args.resume_run_id)
                if existing_run is None:
                    print(f"Run not found: {args.resume_run_id}")
                    return 1
                config = AppConfig.from_dict(json.loads(existing_run.config_json))
            else:
                config = _apply_overrides(initial_config, args)
                repository = _spacetime_repository(config, args.server_url, args.database_name)

        execute_run(
            config=config,
            repository=repository,
            observer=CliObserver(),
            resume_run_id=args.resume_run_id,
        )
        return 0

    if args.command == "run-online":
        initial_config = _load_runtime_config(args.configs)
        config = _apply_overrides(initial_config, args)
        config = replace(config, run=replace(config.run, mode="online"))
        if args.store == "memory":
            repository = InMemoryRepository(run_id_prefix=config.run.run_id_prefix)
        else:
            repository = _spacetime_repository(config, args.server_url, args.database_name)
        execute_online_run(
            config=config,
            repository=repository,
            observer=OnlineCliObserver(),
            resume_run_id=args.resume_run_id,
        )
        return 0

    if args.command == "benchmark":
        return _print_benchmark_report(args)

    if args.command == "benchmark-suite":
        return _print_generation_benchmark_suite_report(args)

    if args.command == "benchmark-online":
        return _print_online_benchmark_report(args)

    if args.command == "analyze-search-space":
        return _print_search_space_report(args)
    if args.command == "analyze-search-space-cross-labels":
        return _print_cross_label_search_space_report(args)

    if args.command == "analyze-archive":
        return _print_archive_report(args)

    if args.command == "analyze-curriculum-boundaries":
        return _print_curriculum_boundary_report(args)

    if args.command == "analyze-retrieval-trace":
        return _print_retrieval_trace_report(args)

    if args.command == "analyze-retrieval-trace-sweep":
        return _print_retrieval_trace_sweep_report(args)

    if args.command == "analyze-fitness-landscape":
        return _print_fitness_landscape_report(args)

    config = _load_runtime_config(DEFAULT_CONFIGS)
    repository = _spacetime_repository(config, args.server_url, args.database_name)

    if args.command == "runs":
        for run in repository.list_runs(limit=args.limit):
            variant = _variant_from_run(run)
            print(
                f"run_id={run.run_id} task={run.task_name} variant={variant} seed={run.seed} "
                f"status={run.status} created_at={run.created_at}"
            )
        return 0

    if args.command == "status":
        return _print_run_status(repository, args.run_id)

    if args.command == "status-online":
        return _print_online_status(repository, args.run_id)

    if args.command == "compare":
        return _print_compare_report(repository, args)

    if args.command == "compare-online":
        return _print_online_compare_report(repository, args)

    if args.command == "elites":
        elites = repository.list_elites(args.run_id, limit=args.limit)
        for elite in elites:
            print(
                f"source_generation={elite.source_generation} rank={elite.rank} "
                f"candidate_id={elite.candidate_id} score={elite.score}"
            )
        return 0

    if args.command == "events":
        events = repository.list_events(args.run_id, limit=args.limit)
        for event in events:
            print(
                f"created_at={event.created_at} type={event.type} payload_json={event.payload_json}"
            )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
