from __future__ import annotations

from db.models import ActiveCandidateRecord, EvaluationJobRecord, EvaluationResultRecord, HallOfFameEntryRecord, OnlineMetricRecord, RunRecord


class OnlineCliObserver:
    def on_run_started(self, run: RunRecord) -> None:
        print(f"run_id={run.run_id} mode={run.mode} task={run.task_name} seed={run.seed} status={run.status}")

    def on_job_finished(
        self,
        job: EvaluationJobRecord,
        result: EvaluationResultRecord,
        candidate: ActiveCandidateRecord,
    ) -> None:
        success = result.raw_metrics.get("success")
        print(
            f"step_candidate={candidate.candidate_id} job_id={job.job_id} score={result.score:.6f} "
            f"rolling_score={candidate.rolling_score:.6f} eval_count={candidate.eval_count} success={success}"
        )

    def on_replacement(
        self,
        retired_candidate: ActiveCandidateRecord,
        offspring_candidate: ActiveCandidateRecord,
        hall_of_fame_entry: HallOfFameEntryRecord | None,
    ) -> None:
        print(
            f"replacement retired={retired_candidate.candidate_id} offspring={offspring_candidate.candidate_id} "
            f"slot={offspring_candidate.slot_index} hall_of_fame_entry={hall_of_fame_entry.entry_id if hall_of_fame_entry else 'none'}"
        )

    def on_metrics(self, metric: OnlineMetricRecord) -> None:
        print(
            f"metrics active_population_size={metric.active_population_size} "
            f"rolling_best_score={metric.rolling_best_score:.6f} "
            f"rolling_avg_score={metric.rolling_avg_score:.6f} "
            f"replacement_count={metric.replacement_count} "
            f"success_rate_window={metric.success_rate_window:.3f}"
        )

    def on_run_finished(self, run: RunRecord) -> None:
        print(f"run_id={run.run_id} status={run.status} finished_at={run.finished_at}")
