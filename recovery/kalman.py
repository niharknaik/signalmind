"""Recovery agent: estimates queues when a camera is unreliable.

One 1-D Kalman filter per approach (state = queued vehicles).
  predict:  q += arrivals - departures            (traffic flow model)
            arrivals come from the upstream junction's outflow when known,
            otherwise from the learned arrival rate
  update:   camera reading, trusted by health     (R grows as health falls)
            loop detector reading, trusted when the camera isn't
Output source: camera | kalman+upstream | loop
"""
from __future__ import annotations

SAT_FLOW = [0.45, 0.9, 0.45, 0.9]     # veh/s that can leave per approach on green (N,E,S,W)


class KalmanQueue:
    def __init__(self, q0: float = 0.0):
        self.x, self.P = q0, 4.0
        self.rate = 0.1                   # learned arrivals veh/s (EMA)

    def predict(self, dt: float, green: bool, sat: float, arrivals: float | None):
        a = arrivals if arrivals is not None else self.rate * dt
        d = sat * dt if green else 0.0
        self.x = max(0.0, self.x + a - min(d, self.x + a))
        self.P += 1.0 + 0.2 * dt

    def update(self, z: float, R: float):
        K = self.P / (self.P + R)
        self.x = max(0.0, self.x + K * (z - self.x))
        self.P = (1 - K) * self.P

    def learn_rate(self, prev_x: float, z: float, dt: float, green: bool, sat: float):
        served = sat * dt if green else 0.0
        arr = max(0.0, z - prev_x + min(served, prev_x))
        self.rate = 0.9 * self.rate + 0.1 * arr / dt


class Recovery:
    def __init__(self, junctions):
        self.f = {j: [KalmanQueue() for _ in range(4)] for j in junctions}

    def estimate(self, j: str, seen: list[float], health: float, green_idx: int,
                 loop: list[float] | None = None, upstream_arrivals: list[float | None] | None = None,
                 dt: float = 5.0):
        """seen: camera counts (possibly faulty). Returns (queues, source)."""
        from twin.sim import SERVES
        src = "camera"
        out = []
        for a, kf in enumerate(self.f[j]):
            green = SERVES[a] == green_idx
            prev = kf.x
            arr = upstream_arrivals[a] if upstream_arrivals else None
            kf.predict(dt, green, SAT_FLOW[a], arr)
            if health >= 0.7:
                kf.update(seen[a], R=0.5)
                kf.learn_rate(prev, seen[a], dt, green, SAT_FLOW[a])
            else:
                if health > 0.05:
                    kf.update(seen[a], R=2.0 / max(health, 0.05) ** 2)
                if loop is not None:
                    kf.update(loop[a], R=6.0)
                    src = "loop" if health < 0.3 else "kalman+upstream"
                else:
                    src = "kalman+upstream"
            out.append(round(kf.x, 2))
        return out, src
