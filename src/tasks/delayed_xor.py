from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DelayedXorTask:
    input_sequences: np.ndarray
    target_sequences: np.ndarray
    input_size: int = 3
    output_size: int = 1

    @classmethod
    def create(cls, delay_steps: int = 1) -> "DelayedXorTask":
        sequences: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for left in (0.0, 1.0):
            for right in (0.0, 1.0):
                xor_value = float((left > 0.5) ^ (right > 0.5))
                sequence = [
                    [left, 0.0, 0.0],
                    [0.0, right, 0.0],
                ]
                sequence.extend([[0.0, 0.0, 0.0] for _ in range(delay_steps)])
                sequence.append([0.0, 0.0, 1.0])

                target_sequence = [[0.0] for _ in range(len(sequence) - 1)]
                target_sequence.append([xor_value])

                sequences.append(np.array(sequence, dtype=np.float32))
                targets.append(np.array(target_sequence, dtype=np.float32))

        return cls(
            input_sequences=np.stack(sequences),
            target_sequences=np.stack(targets),
        )
