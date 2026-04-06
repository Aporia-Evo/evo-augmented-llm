from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BitMemoryTask:
    input_sequences: np.ndarray
    target_sequences: np.ndarray
    delay_steps: int = 1
    input_size: int = 3
    output_size: int = 1

    @classmethod
    def create(cls, delay_steps: int = 1) -> "BitMemoryTask":
        sequences: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        distractor_patterns = (
            [float(step % 2) for step in range(delay_steps)],
            [float((step + 1) % 2) for step in range(delay_steps)],
        )

        for bit in (0.0, 1.0):
            for pattern in distractor_patterns:
                sequence = [[bit, 1.0, 0.0]]
                sequence.extend([[value, 0.0, 0.0] for value in pattern])
                sequence.append([0.0, 0.0, 1.0])

                target_sequence = [[0.0] for _ in range(len(sequence) - 1)]
                target_sequence.append([bit])

                sequences.append(np.array(sequence, dtype=np.float32))
                targets.append(np.array(target_sequence, dtype=np.float32))

        return cls(
            input_sequences=np.stack(sequences),
            target_sequences=np.stack(targets),
            delay_steps=int(delay_steps),
        )
