from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventDecisionTask:
    input_sequences: np.ndarray
    target_sequences: np.ndarray
    input_size: int = 5
    output_size: int = 1

    @classmethod
    def create(cls, delay_steps: int = 4) -> "EventDecisionTask":
        sequences: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        distractor_patterns = (
            [float(step % 2) for step in range(delay_steps)],
            [float((step + 1) % 2) for step in range(delay_steps)],
        )

        # A small deterministic decision task:
        # one relevant event and one irrelevant event appear early, in varying order,
        # then distractors, then a decision cue. The correct decision is the payload
        # of the relevant event, not the most recent or most salient distractor.
        for relevant_payload in (0.0, 1.0):
            for irrelevant_payload in (0.0, 1.0):
                for relevant_first in (True, False):
                    for pattern in distractor_patterns:
                        if relevant_first:
                            sequence = [
                                [relevant_payload, 1.0, 0.0, 0.0, 0.0],
                                [irrelevant_payload, 0.0, 1.0, 0.0, 0.0],
                            ]
                        else:
                            sequence = [
                                [irrelevant_payload, 0.0, 1.0, 0.0, 0.0],
                                [relevant_payload, 1.0, 0.0, 0.0, 0.0],
                            ]

                        sequence.extend([[value, 0.0, 0.0, 1.0, 0.0] for value in pattern])
                        sequence.append([0.0, 0.0, 0.0, 0.0, 1.0])

                        target_sequence = [[0.0] for _ in range(len(sequence) - 1)]
                        target_sequence.append([relevant_payload])

                        sequences.append(np.array(sequence, dtype=np.float32))
                        targets.append(np.array(target_sequence, dtype=np.float32))

        return cls(
            input_sequences=np.stack(sequences),
            target_sequences=np.stack(targets),
        )
