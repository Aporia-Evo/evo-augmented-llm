from __future__ import annotations

from dataclasses import replace

from config import AppConfig, RunConfig, TaskConfig
from evolve.evaluator import build_evaluator
from tasks.event_decision import EventDecisionTask


def test_event_decision_task_has_expected_shapes_and_targets() -> None:
    task = EventDecisionTask.create(delay_steps=3)

    assert task.input_sequences.shape == (16, 6, 5)
    assert task.target_sequences.shape == (16, 6, 1)
    assert sum(task.target_sequences[:, -1, 0].tolist()) == 8.0


def test_event_decision_task_target_tracks_relevant_event_not_irrelevant_one() -> None:
    task = EventDecisionTask.create(delay_steps=1)

    found = False
    for sequence, target in zip(task.input_sequences, task.target_sequences, strict=True):
        relevant_events = sequence[sequence[:, 1] == 1.0]
        irrelevant_events = sequence[sequence[:, 2] == 1.0]
        if relevant_events.shape[0] == 1 and irrelevant_events.shape[0] == 1:
            relevant_payload = float(relevant_events[0, 0])
            irrelevant_payload = float(irrelevant_events[0, 0])
            if relevant_payload != irrelevant_payload:
                assert float(target[-1, 0]) == relevant_payload
                found = True
                break

    assert found is True


def test_event_decision_evaluator_reports_score_ceiling() -> None:
    config = AppConfig(
        task=replace(TaskConfig(), name="event_decision", activation_steps=4, temporal_delay_steps=3),
        run=replace(RunConfig(), seed=7, mode="online"),
    )
    evaluator = build_evaluator(config.task)

    assert evaluator.score_ceiling == 16.0
