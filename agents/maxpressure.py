"""Max-Pressure: pick the green whose approaches have the most (queued in - queued out).

Used as a baseline and as the Fallback mode when a junction's camera is blind.
Works on any queue estimate, so in Fallback it runs on Recovery's numbers.
"""
from twin.sim import SERVES


def choose(queues_in: list[float], queues_out: list[float] | None = None) -> int:
    """queues_in / queues_out ordered N, E, S, W. Returns 0 (main) or 1 (side)."""
    out = queues_out or [0, 0, 0, 0]
    p = [0.0, 0.0]
    for a in range(4):
        p[SERVES[a]] += queues_in[a] - 0.5 * out[a]
    return 0 if p[0] >= p[1] else 1
