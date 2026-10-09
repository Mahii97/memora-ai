from statistics import mean, median
from collections.abc import Sequence

import numpy as np


class LatencyCollector:
    def __init__(self):
        self.samples: dict[str, list[float]] = {}

    def add(self, name: str, milliseconds: float | None) -> None:
        if milliseconds is None:
            return
        if milliseconds < 0:
            raise ValueError("latency cannot be negative")
        self.samples.setdefault(name, []).append(float(milliseconds))

    def extend(self, name: str, milliseconds: Sequence[float]) -> None:
        for value in milliseconds:
            self.add(name, value)

    def summary(self) -> dict[str, dict[str, float | int] | None]:
        summaries = {}
        for name, values in self.samples.items():
            if not values:
                summaries[name] = None
                continue
            summaries[name] = {
                "count": len(values),
                "mean_ms": mean(values),
                "median_ms": median(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "p95_ms": float(np.percentile(values, 95)),
            }
        return summaries
