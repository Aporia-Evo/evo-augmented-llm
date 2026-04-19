"""Tests for the retrieval diagnosis tools (trace + landscape)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis.retrieval_trace import (
    FAILURE_NONE,
    FAILURE_OUTPUT_DECODING,
    FAILURE_QUERY_ALIGNMENT,
    FAILURE_READOUT_COLLAPSE,
    FAILURE_WRITE_SELECTIVITY,
    classify_failure_mode,
    render_trace_report,
    run_retrieval_trace,
)
from analysis.fitness_landscape import (
    VERDICT_DECEPTIVE,
    VERDICT_FLAT,
    VERDICT_WEAK_MONOTONE,
    analyze_fitness_landscape,
    render_landscape_report,
)
from analysis import retrieval_trace_sweep as sweep
from db.models import CandidateFeatureRecord
from evolve.genome_codec import (
    ConnectionGeneModel,
    GenomeModel,
    NodeGeneModel,
    genome_model_to_blob,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _delta_genome() -> GenomeModel:
    """Minimal delta-memory genome for kv_easy (input_size=8, output_size=1)."""
    nodes = [
        NodeGeneModel(node_id=i, bias=0.0, alpha=0.5, is_input=True)
        for i in range(8)
    ]
    nodes.append(
        NodeGeneModel(
            node_id=8,
            bias=0.1,
            alpha=0.3,
            content_w_key=0.3,
            content_b_key=0.1,
            content_w_query=0.25,
            content_b_query=-0.05,
            content_temperature=1.1,
            content_b_match=0.0,
            alpha_slow=0.4,
            slow_input_gain=0.2,
            slow_output_gain=0.1,
        )
    )
    nodes.append(
        NodeGeneModel(
            node_id=9,
            bias=0.0,
            alpha=0.5,
            is_output=True,
            content_temperature=1.0,
        )
    )
    conns = [
        ConnectionGeneModel(
            in_id=i, out_id=8, historical_marker=i, weight=0.2, enabled=True,
        )
        for i in range(8)
    ]
    conns.append(
        ConnectionGeneModel(
            in_id=8, out_id=9, historical_marker=100, weight=0.8, enabled=True,
        )
    )
    return GenomeModel(
        input_ids=tuple(range(8)),
        output_ids=(9,),
        nodes=tuple(nodes),
        connections=tuple(conns),
    )


def _feature_record(**overrides) -> CandidateFeatureRecord:
    """Create a minimal CandidateFeatureRecord with the given overrides."""
    defaults = dict(
        candidate_id="c-test",
        run_id="r-test",
        benchmark_label="test",
        task_name="key_value_memory",
        delay_steps=8,
        variant="stateful_v6_delta_memory",
        seed=7,
        generation=1,
        hof_flag=False,
        success=False,
        final_max_score=5.0,
        first_success_generation=None,
        mean_alpha=0.5,
        std_alpha=0.01,
        mean_eta=0.0,
        std_eta=0.0,
        mean_plastic_d=0.0,
        std_plastic_d=0.0,
        plastic_d_at_lower_bound_fraction=0.0,
        plastic_d_at_zero_fraction=0.0,
        node_count=10,
        enabled_conn_count=8,
        mean_abs_delta_w=0.0,
        max_abs_delta_w=0.0,
        clamp_hit_rate=0.0,
        plasticity_active_fraction=0.0,
    )
    defaults.update(overrides)
    return CandidateFeatureRecord(**defaults)


# ---------------------------------------------------------------------------
# classify_failure_mode tests
# ---------------------------------------------------------------------------


def _make_trace(
    store_beta: float = 0.5,
    distractor_beta: float = 0.3,
    query_alignment: float = 0.5,
    readout_selectivity: float = 0.3,
) -> list[dict[str, object]]:
    """Build a minimal trace matching the classifier's expectations."""
    return [
        {"step_role": "store", "beta_t": store_beta},
        {"step_role": "store", "beta_t": store_beta},
        {"step_role": "distractor", "beta_t": distractor_beta},
        {
            "step_role": "query",
            "query_memory_alignment": query_alignment,
            "readout_selectivity": readout_selectivity,
        },
    ]


def test_classify_failure_mode_none() -> None:
    trace = _make_trace()
    verdict = classify_failure_mode(
        trace,
        predicted_value_ids=[1],
        target_value_ids=[1],
    )
    assert verdict.mode == FAILURE_NONE


def test_classify_failure_mode_write_selectivity() -> None:
    trace = _make_trace(store_beta=0.300, distractor_beta=0.295)
    verdict = classify_failure_mode(
        trace,
        predicted_value_ids=[0],
        target_value_ids=[1],
    )
    assert verdict.mode == FAILURE_WRITE_SELECTIVITY
    assert verdict.metric_value < 0.02


def test_classify_failure_mode_query_alignment() -> None:
    trace = _make_trace(store_beta=0.5, distractor_beta=0.2, query_alignment=0.05)
    verdict = classify_failure_mode(
        trace,
        predicted_value_ids=[0],
        target_value_ids=[1],
    )
    assert verdict.mode == FAILURE_QUERY_ALIGNMENT


def test_classify_failure_mode_readout_collapse() -> None:
    trace = _make_trace(
        store_beta=0.5,
        distractor_beta=0.2,
        query_alignment=0.5,
        readout_selectivity=0.02,
    )
    verdict = classify_failure_mode(
        trace,
        predicted_value_ids=[0],
        target_value_ids=[1],
    )
    assert verdict.mode == FAILURE_READOUT_COLLAPSE


def test_classify_failure_mode_output_decoding() -> None:
    trace = _make_trace(
        store_beta=0.5,
        distractor_beta=0.2,
        query_alignment=0.5,
        readout_selectivity=0.3,
    )
    verdict = classify_failure_mode(
        trace,
        predicted_value_ids=[0],
        target_value_ids=[1],
    )
    assert verdict.mode == FAILURE_OUTPUT_DECODING


# ---------------------------------------------------------------------------
# run_retrieval_trace smoke test
# ---------------------------------------------------------------------------


def test_run_retrieval_trace_returns_trace_result() -> None:
    genome = _delta_genome()
    result = run_retrieval_trace(genome, delay_steps=8, profile="kv_easy")
    assert len(result.trace) > 0
    assert result.verdict.mode in (
        FAILURE_NONE,
        FAILURE_WRITE_SELECTIVITY,
        FAILURE_QUERY_ALIGNMENT,
        FAILURE_READOUT_COLLAPSE,
        FAILURE_OUTPUT_DECODING,
    )
    report = render_trace_report(result, label="smoke")
    assert "Verdict" in report
    assert "Per-Step Diagnostics" in report


# ---------------------------------------------------------------------------
# fitness landscape tests
# ---------------------------------------------------------------------------


def test_analyze_empty_records() -> None:
    result = analyze_fitness_landscape([], label="empty")
    assert result.verdict == VERDICT_FLAT
    assert result.total_records == 0


def test_analyze_flat_landscape() -> None:
    records = [
        _feature_record(
            candidate_id=f"c-{i}",
            generation=i // 5,
            final_max_score=5.0 + np.random.RandomState(i).uniform(-0.01, 0.01),
            correct_value_selected=0.5,
            query_key_match_score=-0.1,
            store_vs_distractor_beta_gap=0.1,
            query_memory_alignment=0.7,
        )
        for i in range(20)
    ]
    result = analyze_fitness_landscape(records, label="flat-test")
    report = render_landscape_report(result)
    assert "Verdict" in report
    assert result.total_records == 20
    assert len(result.generations) > 0


def test_analyze_with_gradient() -> None:
    records = []
    for i in range(30):
        gen = i // 6
        cvs = 0.2 + 0.02 * i
        score = 4.0 + 2.0 * cvs
        records.append(
            _feature_record(
                candidate_id=f"c-grad-{i}",
                generation=gen,
                final_max_score=score,
                correct_value_selected=cvs,
                query_key_match_score=0.1 * cvs,
                store_vs_distractor_beta_gap=0.05 + 0.1 * cvs,
                query_memory_alignment=0.5 + 0.3 * cvs,
            )
        )
    result = analyze_fitness_landscape(records, label="gradient-test")
    report = render_landscape_report(result)
    assert "Verdict" in report
    assert result.verdict in (VERDICT_WEAK_MONOTONE, VERDICT_FLAT, VERDICT_DECEPTIVE)


def test_landscape_report_writes_to_file(tmp_path: Path) -> None:
    records = [
        _feature_record(
            candidate_id=f"c-{i}",
            generation=0,
            final_max_score=5.0,
            correct_value_selected=0.5,
            query_key_match_score=-0.1,
        )
        for i in range(10)
    ]
    result = analyze_fitness_landscape(records, label="write-test")
    from analysis.fitness_landscape import write_landscape_report

    path = write_landscape_report(result, output_dir=str(tmp_path))
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "write-test" in content


def test_retrieval_trace_sweep_verdict_is_allowed(monkeypatch) -> None:
    records = [
        _feature_record(candidate_id="c-1", run_id="r-1", benchmark_label="test", final_max_score=6.0, success=False),
        _feature_record(candidate_id="c-2", run_id="r-2", benchmark_label="test", final_max_score=5.8, success=False),
    ]

    class _FakeVerdict:
        def __init__(self, mode: str) -> None:
            self.mode = mode

    class _FakeTraceResult:
        def __init__(self, sample_index: int) -> None:
            mode = FAILURE_QUERY_ALIGNMENT if sample_index % 2 == 0 else FAILURE_READOUT_COLLAPSE
            self.trace = [
                {
                    "step_role": "query",
                    "key_query_cos_post": 0.1,
                    "query_memory_alignment": 0.1,
                    "readout_selectivity": 0.05,
                }
            ]
            self.raw_outputs = np.asarray([[0.0], [0.1]], dtype=np.float64)
            self.predicted_value_ids = [0]
            self.target_value_ids = [1]
            self.value_levels = (0.0, 0.5, 1.0)
            self.step_roles = ("store", "query")
            self.verdict = _FakeVerdict(mode)

    def _fake_run_retrieval_trace(*args, **kwargs):
        return _FakeTraceResult(int(kwargs.get("sample_index", 0)))

    monkeypatch.setattr(sweep, "run_retrieval_trace", _fake_run_retrieval_trace)

    result = sweep.run_retrieval_trace_sweep(
        benchmark_label="test",
        task_name="key_value_memory",
        variant="stateful_v6_delta_memory",
        candidate_records=records,
        genomes_by_candidate={"c-1": _delta_genome(), "c-2": _delta_genome()},
        top_k_candidates=2,
        episodes_per_candidate=3,
    )
    report = sweep.render_retrieval_trace_sweep_report(result)

    assert result.final_verdict in sweep.ALLOWED_FINAL_VERDICTS
    assert "## 5. Final verdict" in report
    assert "## 2. Candidate table" in report


def test_resolve_local_sweep_inputs_falls_back_to_candidate_genomes(tmp_path: Path) -> None:
    genome = _delta_genome()
    row = {
        "benchmark_label": "v15f-delta",
        "run_id": "run-1",
        "generation_id": 12,
        "candidate_id": "cand-1",
        "task_name": "key_value_memory",
        "variant": "stateful_v6_delta_memory",
        "seed": 7,
        "genome_blob": genome_model_to_blob(genome),
    }
    path = tmp_path / "v15f-delta.candidate-genomes.jsonl"
    import json

    path.write_text(f"{json.dumps(row)}\n", encoding="utf-8")

    resolved = sweep.resolve_local_sweep_inputs(
        output_dir=tmp_path,
        benchmark_label="v15f-delta",
        task_name="key_value_memory",
        variant="stateful_v6_delta_memory",
    )
    assert resolved.source_used == "candidate-genomes"
    assert len(resolved.candidates) == 1
    assert resolved.candidates[0].candidate_id == "cand-1"
    assert "cand-1" in resolved.genomes_by_candidate


def test_resolve_local_sweep_inputs_prefers_benchmark_jsonl_over_genomes(tmp_path: Path) -> None:
    import json

    genome = _delta_genome()
    blob = genome_model_to_blob(genome)
    (tmp_path / "demo.candidate-genomes.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "benchmark_label": "demo",
                    "run_id": "run-g",
                    "generation_id": 1,
                    "candidate_id": candidate_id,
                    "task_name": "key_value_memory",
                    "variant": "stateful_v6_delta_memory",
                    "seed": 7,
                    "genome_blob": blob,
                }
            )
            for candidate_id in ("cand-a", "cand-b")
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "demo.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "run_id": "run-j",
                    "candidate_id": candidate_id,
                    "task_name": "key_value_memory",
                    "variant": "stateful_v6_delta_memory",
                    "final_max_score": score,
                }
            )
            for candidate_id, score in (("cand-a", 1.0), ("cand-b", 2.0))
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = sweep.resolve_local_sweep_inputs(
        output_dir=tmp_path,
        benchmark_label="demo",
        task_name="key_value_memory",
        variant="stateful_v6_delta_memory",
    )
    assert resolved.source_used == "benchmark-jsonl"
    assert {c.candidate_id for c in resolved.candidates} == {"cand-a", "cand-b"}
