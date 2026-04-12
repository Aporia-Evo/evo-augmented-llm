"""Population-level fitness-landscape analysis for retrieval stagnation diagnosis.

Uses persisted ``CandidateFeatureRecord`` data exclusively — no new evaluations.
Computes fitness-vs-retrieval correlations, selection-pressure-bonus variance
decomposition, phenotype diversity of top genomes, and retrieval-axis fitness
gradient.  Emits a structured Markdown report with a verdict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence

import numpy as np

from db.models import CandidateFeatureRecord


# ---------------------------------------------------------------------------
# Retrieval sub-metrics used for correlation analysis
# ---------------------------------------------------------------------------

RETRIEVAL_FEATURES = (
    "correct_value_selected",
    "query_key_match_score",
    "store_vs_distractor_beta_gap",
    "query_memory_alignment",
    "retrieval_margin",
    "readout_selectivity",
    "key_query_cosine_mean",
    "key_query_cosine_at_query",
)

# ---------------------------------------------------------------------------
# Selection-pressure bonus reconstruction (mirrors evaluator.py:1280-1316)
# ---------------------------------------------------------------------------

_BONUS_TERMS = (
    ("correct_value_selected", 0.45),
    ("positive_query_match", 0.22),
    ("positive_beta_gap", 0.16),
    ("correct_key_selected", 0.02),
    ("negative_match_penalty", -0.08),
    ("kq_cos_mean_penalty", -0.012),
    ("kq_cos_query_penalty", -0.012),
    ("key_var_penalty", -0.008),
    ("query_var_penalty", -0.008),
)


def _reconstruct_bonus_terms(record: CandidateFeatureRecord) -> dict[str, float]:
    """Decompose the selection-pressure bonus into its constituent terms."""
    qkm = float(record.query_key_match_score)
    bgap = float(record.store_vs_distractor_beta_gap)
    positive_query_match = float(np.tanh(max(qkm, 0.0)))
    positive_beta_gap = float(np.tanh(max(bgap, 0.0)))

    return {
        "correct_value_selected": 0.45 * float(record.correct_value_selected),
        "positive_query_match": 0.22 * positive_query_match,
        "positive_beta_gap": 0.16 * positive_beta_gap,
        "correct_key_selected": 0.02 * float(record.correct_key_selected),
        "negative_match_penalty": -0.08 * abs(min(qkm, 0.0)),
        "kq_cos_mean_penalty": -0.012 * max(
            0.0, float(record.key_query_cosine_mean) - 0.50
        ),
        "kq_cos_query_penalty": -0.012 * max(
            0.0, float(record.key_query_cosine_at_query) - 0.50
        ),
        "key_var_penalty": -0.008 * max(
            0.0, 0.02 - float(record.key_variance_mean)
        ),
        "query_var_penalty": -0.008 * max(
            0.0, 0.02 - float(record.query_variance_mean)
        ),
    }


def _total_bonus(terms: dict[str, float]) -> float:
    return float(np.clip(sum(terms.values()), -0.15, 0.85))


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3:
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    xm, ym = xa - xa.mean(), ya - ya.mean()
    denom = float(np.linalg.norm(xm) * np.linalg.norm(ym))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(xm, ym) / denom)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3:
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    rx = np.argsort(np.argsort(xa)).astype(np.float64)
    ry = np.argsort(np.argsort(ya)).astype(np.float64)
    return _pearson(rx.tolist(), ry.tolist())


# ---------------------------------------------------------------------------
# Analysis dataclass
# ---------------------------------------------------------------------------

VERDICT_FLAT = "flat-gradient"
VERDICT_DECEPTIVE = "deceptive-local-optimum"
VERDICT_WEAK_MONOTONE = "weak-but-monotone-gradient"


@dataclass(frozen=True)
class LandscapeResult:
    """Structured output of the fitness-landscape analysis."""

    label: str
    total_records: int
    generations: list[int]
    population_per_gen: list[int]

    # Per-generation Pearson correlations (feature -> list[float])
    pearson_by_feature: dict[str, list[float]]
    spearman_by_feature: dict[str, list[float]]

    # Bonus variance decomposition (term_name -> fraction of total variance)
    bonus_variance_fractions: dict[str, float]

    # Top-k phenotype diversity per generation
    top_k_diversity: list[float]

    # Retrieval-axis gradient: (bin_center, mean_fitness) pairs
    retrieval_axis_bins: list[tuple[float, float]]

    verdict: str
    verdict_detail: str


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyze_fitness_landscape(
    records: Sequence[CandidateFeatureRecord],
    *,
    label: str = "",
    top_k: int = 20,
    num_bins: int = 5,
) -> LandscapeResult:
    """Run the full fitness-landscape diagnosis on candidate feature records."""
    if not records:
        return LandscapeResult(
            label=label,
            total_records=0,
            generations=[],
            population_per_gen=[],
            pearson_by_feature={},
            spearman_by_feature={},
            bonus_variance_fractions={},
            top_k_diversity=[],
            retrieval_axis_bins=[],
            verdict=VERDICT_FLAT,
            verdict_detail="No records provided.",
        )

    # Group by generation
    by_gen: dict[int, list[CandidateFeatureRecord]] = {}
    for rec in records:
        by_gen.setdefault(rec.generation, []).append(rec)
    generations = sorted(by_gen.keys())
    pop_per_gen = [len(by_gen[g]) for g in generations]

    # 1. Fitness vs retrieval correlations
    pearson_by_feature: dict[str, list[float]] = {f: [] for f in RETRIEVAL_FEATURES}
    spearman_by_feature: dict[str, list[float]] = {f: [] for f in RETRIEVAL_FEATURES}

    for gen in generations:
        gen_records = by_gen[gen]
        fitness_vals = [float(r.final_max_score) for r in gen_records]
        for feature in RETRIEVAL_FEATURES:
            fvals = [float(getattr(r, feature, 0.0)) for r in gen_records]
            pearson_by_feature[feature].append(_pearson(fitness_vals, fvals))
            spearman_by_feature[feature].append(_spearman(fitness_vals, fvals))

    # 2. Bonus variance decomposition
    all_term_arrays: dict[str, list[float]] = {name: [] for name, _ in _BONUS_TERMS}
    for rec in records:
        terms = _reconstruct_bonus_terms(rec)
        for name, _ in _BONUS_TERMS:
            all_term_arrays[name].append(terms[name])

    total_bonuses = [
        _total_bonus(_reconstruct_bonus_terms(rec)) for rec in records
    ]
    total_var = float(np.var(total_bonuses)) if len(total_bonuses) > 1 else 1e-9
    if total_var < 1e-12:
        total_var = 1e-9

    bonus_variance_fractions: dict[str, float] = {}
    for name, _ in _BONUS_TERMS:
        term_var = float(np.var(all_term_arrays[name])) if all_term_arrays[name] else 0.0
        bonus_variance_fractions[name] = term_var / total_var

    # 3. Top-k phenotype diversity
    top_k_diversity: list[float] = []
    for gen in generations:
        gen_records = sorted(by_gen[gen], key=lambda r: -r.final_max_score)[:top_k]
        if len(gen_records) < 2:
            top_k_diversity.append(0.0)
            continue
        vectors = []
        for rec in gen_records:
            vectors.append(
                [float(getattr(rec, f, 0.0)) for f in RETRIEVAL_FEATURES]
            )
        va = np.asarray(vectors, dtype=np.float64)
        dists: list[float] = []
        for i in range(len(va)):
            for j in range(i + 1, len(va)):
                dists.append(float(np.linalg.norm(va[i] - va[j])))
        top_k_diversity.append(mean(dists) if dists else 0.0)

    # 4. Retrieval-axis gradient (bin correct_value_selected)
    cvs_vals = [float(r.correct_value_selected) for r in records]
    fitness_vals_all = [float(r.final_max_score) for r in records]
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    retrieval_axis_bins: list[tuple[float, float]] = []
    for b in range(num_bins):
        lo, hi = float(bin_edges[b]), float(bin_edges[b + 1])
        center = (lo + hi) / 2.0
        in_bin = [
            f for c, f in zip(cvs_vals, fitness_vals_all)
            if (lo <= c < hi) or (b == num_bins - 1 and c == hi)
        ]
        if in_bin:
            retrieval_axis_bins.append((center, mean(in_bin)))

    # 5. Verdict
    verdict, detail = _compute_verdict(
        pearson_by_feature, retrieval_axis_bins, bonus_variance_fractions,
    )

    return LandscapeResult(
        label=label,
        total_records=len(records),
        generations=generations,
        population_per_gen=pop_per_gen,
        pearson_by_feature=pearson_by_feature,
        spearman_by_feature=spearman_by_feature,
        bonus_variance_fractions=bonus_variance_fractions,
        top_k_diversity=top_k_diversity,
        retrieval_axis_bins=retrieval_axis_bins,
        verdict=verdict,
        verdict_detail=detail,
    )


def _compute_verdict(
    pearson_by_feature: dict[str, list[float]],
    retrieval_axis_bins: list[tuple[float, float]],
    bonus_variance_fractions: dict[str, float],
) -> tuple[str, str]:
    # Check if any retrieval feature has consistent positive Pearson
    mean_pearsons = {
        f: mean(vals) if vals else 0.0 for f, vals in pearson_by_feature.items()
    }
    best_feature = max(mean_pearsons, key=lambda f: mean_pearsons[f]) if mean_pearsons else ""
    best_corr = mean_pearsons.get(best_feature, 0.0)

    # Check retrieval axis monotonicity
    monotone = True
    if len(retrieval_axis_bins) >= 3:
        fitnesses = [f for _, f in retrieval_axis_bins]
        for i in range(len(fitnesses) - 1):
            if fitnesses[i + 1] < fitnesses[i] - 0.01:
                monotone = False
                break

    # Check for dominant deceptive term
    dominant_term = max(bonus_variance_fractions, key=lambda t: bonus_variance_fractions[t]) if bonus_variance_fractions else ""
    dominant_fraction = bonus_variance_fractions.get(dominant_term, 0.0)

    if best_corr < 0.05 and not monotone:
        return (
            VERDICT_FLAT,
            f"No retrieval feature correlates with fitness (best: {best_feature} "
            f"r={best_corr:.3f}). Retrieval-axis fitness curve is not monotone. "
            f"Evolution has no gradient toward better retrieval.",
        )

    if dominant_fraction > 0.70 and dominant_term != "correct_value_selected":
        return (
            VERDICT_DECEPTIVE,
            f"Selection-pressure bonus dominated by `{dominant_term}` "
            f"({dominant_fraction:.0%} of variance). This may create a deceptive "
            f"local optimum that rewards the wrong signal.",
        )

    if best_corr > 0.05 and monotone:
        return (
            VERDICT_WEAK_MONOTONE,
            f"Weak but monotone gradient detected (best: {best_feature} "
            f"r={best_corr:.3f}). Evolution sees a signal but may be too slow. "
            f"Consider increasing selection pressure or population size.",
        )

    if not monotone and dominant_fraction > 0.50 and dominant_term != "correct_value_selected":
        return (
            VERDICT_DECEPTIVE,
            f"Non-monotone fitness curve + dominant bonus term `{dominant_term}` "
            f"({dominant_fraction:.0%}). Likely deceptive local optimum.",
        )

    return (
        VERDICT_FLAT,
        f"Ambiguous landscape. Best correlation: {best_feature} r={best_corr:.3f}. "
        f"Dominant bonus term: {dominant_term} ({dominant_fraction:.0%}). "
        f"More data or stronger selection pressure may be needed.",
    )


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_landscape_report(result: LandscapeResult) -> str:
    """Render a full fitness-landscape Markdown report."""
    lines: list[str] = []
    lines.append(f"# Fitness Landscape Report{(' — ' + result.label) if result.label else ''}")
    lines.append("")

    # --- summary ---
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(f"- benchmark_label: `{result.label}`")
    lines.append(f"- total candidate records: {result.total_records}")
    lines.append(f"- generations: {len(result.generations)}")
    if result.population_per_gen:
        lines.append(f"- population per generation (mean): {mean(result.population_per_gen):.0f}")
    lines.append("")

    # --- correlation table ---
    if result.pearson_by_feature:
        lines.append("## Fitness vs Retrieval Correlations (mean Pearson across generations)")
        lines.append("")
        header = ["feature", "mean_pearson", "mean_spearman", "min_pearson", "max_pearson"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for feature in RETRIEVAL_FEATURES:
            pvals = result.pearson_by_feature.get(feature, [])
            svals = result.spearman_by_feature.get(feature, [])
            if pvals:
                row = [
                    f"`{feature}`",
                    f"{mean(pvals):.3f}",
                    f"{mean(svals):.3f}" if svals else "n/a",
                    f"{min(pvals):.3f}",
                    f"{max(pvals):.3f}",
                ]
            else:
                row = [f"`{feature}`", "n/a", "n/a", "n/a", "n/a"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # --- bonus variance decomposition ---
    if result.bonus_variance_fractions:
        lines.append("## Selection-Pressure Bonus Variance Decomposition")
        lines.append("")
        bv_header = ["term", "variance_fraction"]
        lines.append("| " + " | ".join(bv_header) + " |")
        lines.append("| " + " | ".join("---" for _ in bv_header) + " |")
        sorted_terms = sorted(
            result.bonus_variance_fractions.items(),
            key=lambda item: -item[1],
        )
        for name, frac in sorted_terms:
            lines.append(f"| `{name}` | {frac:.1%} |")
        lines.append("")

    # --- top-k diversity ---
    if result.top_k_diversity:
        lines.append("## Top-k Phenotype Diversity Trend")
        lines.append("")
        div_header = ["generation", "mean_pairwise_L2"]
        lines.append("| " + " | ".join(div_header) + " |")
        lines.append("| " + " | ".join("---" for _ in div_header) + " |")
        for gen, div in zip(result.generations, result.top_k_diversity):
            lines.append(f"| {gen} | {div:.4f} |")
        lines.append("")

    # --- retrieval axis ---
    if result.retrieval_axis_bins:
        lines.append("## Retrieval-Axis Fitness Curve")
        lines.append("")
        lines.append(
            "Bins of `correct_value_selected` with mean fitness per bin."
        )
        lines.append("")
        ra_header = ["bin_center", "mean_fitness"]
        lines.append("| " + " | ".join(ra_header) + " |")
        lines.append("| " + " | ".join("---" for _ in ra_header) + " |")
        for center, fit in result.retrieval_axis_bins:
            lines.append(f"| {center:.2f} | {fit:.4f} |")
        lines.append("")

    # --- verdict ---
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{result.verdict}**")
    lines.append("")
    lines.append(result.verdict_detail)
    lines.append("")

    return "\n".join(lines)


def write_landscape_report(
    result: LandscapeResult,
    *,
    output_dir: str | Path = "results",
) -> Path:
    """Write landscape report to ``results/fitness-landscape-<label>.md``."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_name = f"fitness-landscape-{result.label}.md" if result.label else "fitness-landscape.md"
    report_path = output_path / report_name
    report_path.write_text(render_landscape_report(result), encoding="utf-8")
    return report_path
