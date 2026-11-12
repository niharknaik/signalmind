"""Builds the brain's per-junction observations from a running CorridorEnv."""
from __future__ import annotations

from twin.sim import J


def brain_obs(env) -> dict:
    c = env.c
    out = {}
    for j in J:
        pw = c.peds(j)
        d = c.drivers[j]
        out[j] = {
            "est": env.est[j],
            "health": env.info[j]["health"],
            "fault": env.info[j]["fault"],
            "source": env.info[j]["source"],
            "green": d.green,
            "stage": d.stage,
            "timer": d.timer,
            "emergency": c.emergency(j),
            "ped_wait": [max(pw[0]) if pw[0] else 0.0, max(pw[1]) if pw[1] else 0.0],
            "vec": env.obs(j),
        }
    return out
