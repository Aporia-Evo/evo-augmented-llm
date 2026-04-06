from __future__ import annotations

from dataclasses import replace

from config import AppConfig, RunConfig, TaskConfig
from evolve.evaluator import build_evaluator
from tasks.event_memory import EventMemoryTask


def test_event_memory_task_has_expected_recall_targets() -> None:
    task = EventMemoryTask.create(delay_steps=3)

    assert task.input_sequences.shape == (4, 5, 4)
    assert task.target_sequences.shape == (4, 5, 1)
    assert task.target_sequences[:, -1, 0].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_event_memory_evaluator_reports_score_ceiling() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="event_memory", activation_steps=4, temporal_delay_steps=3),
        run=replace(RunConfig(), seed=7, mode="online"),
    )
    evaluator = build_evaluator(config.task)

    assert evaluator.score_ceiling == 4.0
