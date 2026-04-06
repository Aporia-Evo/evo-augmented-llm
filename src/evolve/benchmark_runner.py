from __future__ import annotations

from dataclasses import replace

from config import AppConfig
from db.online_repository import OnlineRunRepository
from evolve.evaluator import score_ceiling_for_task
from evolve.online_loop import execute_online_run
from ui.compare_report import OnlineCompareSummary, build_online_compare_summary


def run_online_benchmark(
    *,
    base_config: AppConfig,
    repository: OnlineRunRepository,
    seeds: list[int],
    variants: tuple[str, str] = ("stateful", "stateless"),
) -> list[OnlineCompareSummary]:
    summaries: list[OnlineCompareSummary] = []
    score_ceiling = score_ceiling_for_task(base_config.task)
    for seed in seeds:
        for variant in variants:
            config = replace(base_config, run=replace(base_config.run, seed=seed, variant=variant, mode="online"))
            result = execute_online_run(config=config, repository=repository)
            summaries.append(build_online_compare_summary(repository, result.run, score_ceiling))
    return summaries
