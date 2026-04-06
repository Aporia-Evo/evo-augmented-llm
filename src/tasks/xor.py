from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class XorTask:
    inputs: np.ndarray
    targets: np.ndarray
    input_size: int = 2
    output_size: int = 1

    @classmethod
    def create(cls) -> "XorTask":
        return cls(
            inputs=np.array(
                [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                dtype=np.float32,
            ),
            targets=np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32),
        )
