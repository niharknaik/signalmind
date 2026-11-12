"""Fault injector, shared by both systems.

- apply(frame, ...)        System 1: breaks camera FRAMES (demo path)
- CountFaults              System 2: breaks COUNTS the same way (training path)

Fault types: blackout, noise, freeze, occlusion. A fault lasts 10-60 s.
"""
from __future__ import annotations

import random

import numpy as np

TYPES = ["blackout", "noise", "freeze", "occlusion"]
# what the Sentinel reports for each fault (health 0..1)
HEALTH = {"none": 1.0, "blackout": 0.0, "freeze": 0.15, "noise": 0.5, "occlusion": 0.45}


# ------------------------------------------------------------ frames (S1)
def apply(frame: np.ndarray, fault: str, prev: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    if fault == "blackout":
        return np.zeros_like(frame)
    if fault == "noise":
        n = rng.normal(0, 40, frame.shape)
        return np.clip(frame.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if fault == "freeze" and prev is not None:
        return prev
    if fault == "occlusion":
        h, w = frame.shape[:2]
        out = frame.copy()
        y, x = rng.integers(0, h // 2), rng.integers(0, w // 3)
        out[y:y + int(h * 0.6), x:x + int(w * 0.65)] = 0
        return out
    return frame


# ------------------------------------------------------------ counts (S2)
class CountFaults:
    """Per-junction fault process on count vectors. Call .step(j, queues, dt) every decision."""

    def __init__(self, junctions, rate: float = 0.0, types=None, seed: int = 0):
        self.rate, self.types = rate, list(types or TYPES)
        self.rng = random.Random(seed)
        self.state = {j: {"fault": "none", "left": 0.0, "frozen": None} for j in junctions}

    def set_rate(self, rate: float, types=None):
        self.rate = rate
        if types:
            self.types = list(types)

    def step(self, j: str, queues: list[float], dt: float = 5.0):
        s = self.state[j]
        if s["fault"] == "none":
            # rate = long-run fraction of time a camera is faulty (mean fault 35 s)
            p_start = 0.0 if self.rate <= 0 else min(1.0, dt * self.rate / (35.0 * (1 - min(self.rate, 0.95))))
            if self.types and self.rng.random() < p_start:
                s["fault"], s["left"], s["frozen"] = self.rng.choice(self.types), self.rng.uniform(10, 60), list(queues)
        else:
            s["left"] -= dt
            if s["left"] <= 0:
                s["fault"] = "none"
        f = s["fault"]
        if f == "blackout":
            seen = [0.0] * len(queues)
        elif f == "freeze":
            seen = list(s["frozen"])
        elif f == "noise":
            seen = [max(0.0, q + self.rng.gauss(0, 3 + 0.5 * q)) for q in queues]
        elif f == "occlusion":
            seen = [q * self.rng.uniform(0.2, 0.6) for q in queues]
        else:
            seen = list(queues)
        return seen, f, HEALTH[f]
