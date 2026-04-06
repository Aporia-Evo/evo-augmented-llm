from __future__ import annotations

import numpy as np

from evolve.evaluator import XorEvaluator
from tasks.xor import XorTask


def test_xor_score_is_max_for_exact_predictions() -> None:
    task = XorTask.create()
    score = XorEvaluator.score_predictions(task.targets, task.targets)
    assert score == 4.0


def test_xor_score_penalizes_wrong_predictions() -> None:
    task = XorTask.create()
    wrong = np.array([[1.0], [0.0], [0.0], [1.0]], dtype=np.float32)
    score = XorEvaluator.score_predictions(wrong, task.targets)
    assert 0.0 <= score < 4.0

