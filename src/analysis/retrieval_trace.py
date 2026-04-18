"""Per-episode mechanistic trace for delta-memory retrieval diagnosis.

Runs a single genome through a ``KeyValueMemoryTask`` episode with the
``trace_sink`` hook enabled, then classifies the failure mode and renders
a Markdown report.  This module never changes any math — it is a read-only
diagnostic consumer of ``StatefulV6DeltaMemoryNetworkExecutor.run_sequence``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Sequence

import numpy as np

from evolve.custom_neuron import StatefulV6DeltaMemoryNetworkExecutor
from evolve.genome_codec import GenomeModel
from tasks.key_value_memory import KeyValueMemoryTask


# ---------------------------------------------------------------------------
# Failure-mode classification
# ---------------------------------------------------------------------------

FAILURE_WRITE_SELECTIVITY = "write_selectivity"
FAILURE_QUERY_ALIGNMENT = "query_alignment"
FAILURE_READOUT_COLLAPSE = "readout_collapse"
FAILURE_OUTPUT_DECODING = "output_decoding"
FAILURE_NONE = "none"


@dataclass(frozen=True)
class FailureVerdict:
    mode: str
    metric_name: str
    metric_value: float
    threshold: float


def classify_failure_mode(
    trace: Sequence[dict[str, object]],
    *,
    predicted_value_ids: Sequence[int],
    target_value_ids: Sequence[int],
) -> FailureVerdict:
    """Classify the dominant retrieval failure from a per-step trace.

    The trace is a list of dicts produced by the ``trace_sink`` hook of
    ``StatefulV6DeltaMemoryNetworkExecutor.run_sequence``.  Each dict
    represents one step x node snapshot.

    Returns a ``FailureVerdict`` naming the most severe failure layer
    (or ``"none"`` if retrieval was correct).
    """
    store_betas: list[float] = []
    distractor_betas: list[float] = []
    query_alignment_vals: list[float] = []
    readout_selectivity_vals: list[float] = []

    for record in trace:
        role = str(record.get("step_role", ""))
        if role == "store":
            store_betas.append(float(record.get("beta_t", 0.0)))
        elif role == "distractor":
            distractor_betas.append(float(record.get("beta_t", 0.0)))
        elif role == "query":
            query_alignment_vals.append(
                float(record.get("query_memory_alignment", 0.0))
            )
            readout_selectivity_vals.append(
                float(record.get("readout_selectivity", 0.0))
            )

    mean_store_beta = mean(store_betas) if store_betas else 0.0
    mean_distractor_beta = mean(distractor_betas) if distractor_betas else 0.0
    beta_gap = mean_store_beta - mean_distractor_beta

    mean_query_alignment = (
        mean(query_alignment_vals) if query_alignment_vals else 0.0
    )
    mean_readout_selectivity = (
        mean(readout_selectivity_vals) if readout_selectivity_vals else 0.0
    )

    correct_count = sum(
        1 for p, t in zip(predicted_value_ids, target_value_ids) if p == t
    )
    all_correct = correct_count == len(target_value_ids) and len(target_value_ids) > 0

    if all_correct:
        return FailureVerdict(
            mode=FAILURE_NONE,
            metric_name="correct_value_selected",
            metric_value=1.0,
            threshold=1.0,
        )
    if beta_gap < 0.05:
        return FailureVerdict(
            mode=FAILURE_WRITE_SELECTIVITY,
            metric_name="store_vs_distractor_beta_gap",
            metric_value=beta_gap,
            threshold=0.05,
        )
    if mean_query_alignment < 0.20:
        return FailureVerdict(
            mode=FAILURE_QUERY_ALIGNMENT,
            metric_name="query_memory_alignment",
            metric_value=mean_query_alignment,
            threshold=0.20,
        )
    if mean_readout_selectivity < 0.10:
        return FailureVerdict(
            mode=FAILURE_READOUT_COLLAPSE,
            metric_name="readout_selectivity",
            metric_value=mean_readout_selectivity,
            threshold=0.10,
        )
    return FailureVerdict(
        mode=FAILURE_OUTPUT_DECODING,
        metric_name="correct_value_selected",
        metric_value=correct_count / max(1, len(target_value_ids)),
        threshold=1.0,
    )


# ---------------------------------------------------------------------------
# Trace runner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceResult:
    """Full per-episode diagnostic output."""

    trace: list[dict[str, object]]
    raw_outputs: np.ndarray
    predicted_value_ids: list[int]
    target_value_ids: list[int]
    value_levels: tuple[float, ...]
    step_roles: tuple[str, ...]
    store_value_ids: list[int]
    verdict: FailureVerdict


def run_retrieval_trace(
    genome: GenomeModel,
    *,
    delay_steps: int = 8,
    profile: str = "kv_easy",
    activation_steps: int = 4,
    sample_index: int = 0,
    variant: str = "stateful_v6_delta_memory",
) -> TraceResult:
    """Execute a single sample from a KV task and return trace diagnostics."""
    task = KeyValueMemoryTask.create(delay_steps=delay_steps, profile=profile)

    if sample_index >= len(task.input_sequences):
        sample_index = 0

    executor = StatefulV6DeltaMemoryNetworkExecutor(
        activation_steps=activation_steps,
        sub_variant=variant,
    )
    sink: list[dict[str, object]] = []
    raw_output = executor.run_sequence(
        genome,
        task.input_sequences[sample_index],
        step_roles=task.step_roles[sample_index],
        trace_sink=sink,
    )

    bounded = np.clip((raw_output[:, 0] + 1.0) / 2.0, 0.0, 1.0)
    value_levels = np.asarray(task.value_levels, dtype=np.float64)
    query_step_indices = [
        i for i, role in enumerate(task.step_roles[sample_index]) if role == "query"
    ]
    predicted_ids: list[int] = []
    for qi in query_step_indices:
        distances = np.abs(value_levels - float(bounded[qi]))
        predicted_ids.append(int(np.argmin(distances)))

    target_ids = [int(task.query_target_ids[sample_index])]
    store_vids = [int(v) for v in task.store_value_ids[sample_index]]

    verdict = classify_failure_mode(
        sink,
        predicted_value_ids=predicted_ids,
        target_value_ids=target_ids,
    )

    return TraceResult(
        trace=sink,
        raw_outputs=raw_output,
        predicted_value_ids=predicted_ids,
        target_value_ids=target_ids,
        value_levels=task.value_levels,
        step_roles=task.step_roles[sample_index],
        store_value_ids=store_vids,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Markdown report renderer
# ---------------------------------------------------------------------------

_VERDICT_LABELS = {
    FAILURE_NONE: "Retrieval correct",
    FAILURE_WRITE_SELECTIVITY: "(a) Write selectivity — distractors written as strongly as real stores",
    FAILURE_QUERY_ALIGNMENT: "(b) Query alignment — query does not align with stored memory",
    FAILURE_READOUT_COLLAPSE: "(c) Readout collapse — memory returns diffuse average, not a peak",
    FAILURE_OUTPUT_DECODING: "(d) Output decoding — readout appears targeted but output node collapses",
}


def render_trace_report(result: TraceResult, *, label: str = "") -> str:
    """Render a Markdown retrieval-trace report."""
    lines: list[str] = []
    lines.append(f"# Retrieval Trace Report{(' — ' + label) if label else ''}")
    lines.append("")

    # --- meta ---
    lines.append("## Episode Meta")
    lines.append("")
    lines.append(f"- value_levels: `{list(result.value_levels)}`")
    lines.append(f"- step_roles: `{list(result.step_roles)}`")
    lines.append(f"- store_value_ids: `{result.store_value_ids}`")
    lines.append(f"- target_value_ids: `{result.target_value_ids}`")
    lines.append(f"- predicted_value_ids: `{result.predicted_value_ids}`")
    lines.append("")

    # --- per-step table ---
    lines.append("## Per-Step Diagnostics")
    lines.append("")
    header_cols = [
        "step", "role", "node", "beta_t", "store_sig", "query_sig",
        "kq_cos", "mem_frob", "qm_align", "readout_sel", "update_frob",
        "node_out",
    ]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in header_cols) + " |")
    for record in result.trace:
        row = [
            str(record.get("step_index", "")),
            str(record.get("step_role", "")),
            str(record.get("node_id", "")),
            f"{float(record.get('beta_t', 0)):.3f}",
            f"{float(record.get('store_signal', 0)):.3f}",
            f"{float(record.get('query_signal', 0)):.3f}",
            f"{float(record.get('key_query_cos_post', 0)):.3f}",
            f"{float(record.get('memory_frob_post', 0)):.3f}",
            f"{float(record.get('query_memory_alignment', 0)):.3f}",
            f"{float(record.get('readout_selectivity', 0)):.3f}",
            f"{float(record.get('update_frob', 0)):.3f}",
            f"{float(record.get('node_output', 0)):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- query readout table ---
    query_records = [r for r in result.trace if r.get("step_role") == "query"]
    if query_records:
        lines.append("## Query-Step Readout Detail")
        lines.append("")
        for qr in query_records:
            read_t = qr.get("read_t", [])
            q_focus = qr.get("q_focus", [])
            lines.append(f"**Node {qr.get('node_id')} (step {qr.get('step_index')})**")
            lines.append("")
            lines.append(f"- read_t: `[{', '.join(f'{v:.4f}' for v in read_t)}]`")
            lines.append(f"- q_focus: `[{', '.join(f'{v:.4f}' for v in q_focus)}]`")
            lines.append(f"- selective_readout: `{float(qr.get('selective_readout', 0)):.4f}`")
            lines.append(f"- read_contrast: `{float(qr.get('read_contrast', 0)):.4f}`")
            lines.append(f"- readout_scalar: `{float(qr.get('readout_scalar', 0)):.4f}`")
            lines.append(f"- node_output: `{float(qr.get('node_output', 0)):.4f}`")
            lines.append("")

        # projection table
        value_levels = np.asarray(result.value_levels)
        bounded = np.clip((result.raw_outputs[:, 0] + 1.0) / 2.0, 0.0, 1.0)
        query_step_indices = [
            i for i, role in enumerate(result.step_roles) if role == "query"
        ]
        if query_step_indices:
            lines.append("### Value-Level Projections")
            lines.append("")
            proj_header = ["value_id", "value", "distance", "winner", "target"]
            lines.append("| " + " | ".join(proj_header) + " |")
            lines.append("| " + " | ".join("---" for _ in proj_header) + " |")
            for qi_idx, qi in enumerate(query_step_indices):
                predicted = float(bounded[qi])
                target_id = (
                    result.target_value_ids[qi_idx]
                    if qi_idx < len(result.target_value_ids)
                    else -1
                )
                predicted_id = (
                    result.predicted_value_ids[qi_idx]
                    if qi_idx < len(result.predicted_value_ids)
                    else -1
                )
                for vid, vl in enumerate(result.value_levels):
                    dist = abs(predicted - vl)
                    is_winner = vid == predicted_id
                    is_target = vid == target_id
                    row = [
                        str(vid),
                        f"{vl:.3f}",
                        f"{dist:.4f}",
                        "**>>**" if is_winner else "",
                        "TARGET" if is_target else "",
                    ]
                    lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    # --- verdict ---
    lines.append("## Verdict")
    lines.append("")
    v = result.verdict
    lines.append(f"**{_VERDICT_LABELS.get(v.mode, v.mode)}**")
    lines.append("")
    lines.append(
        f"Triggered by `{v.metric_name}` = {v.metric_value:.4f} "
        f"(threshold: {v.threshold:.4f})"
    )
    lines.append("")

    return "\n".join(lines)


def write_trace_report(
    result: TraceResult,
    *,
    output_dir: str | Path = "results",
    label: str = "trace",
) -> Path:
    """Write trace report to ``results/retrieval-trace-<label>.md``."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"retrieval-trace-{label}.md"
    report_path.write_text(render_trace_report(result, label=label), encoding="utf-8")
    return report_path
