"""LangGraph supervisor: one pass per 5 s control cycle.

    sense -> recover -> decide -> act -> publish

sense    latest gt.* (SUMO truth, road sensors) and perc.* (camera + Sentinel) from System 1
recover  Recovery agent: Kalman + upstream + loop fusion -> cleaned queues per junction
decide   Brain: mode routing, specialist agents, priority arbiter, PPO junction agents
act      act.* to System 1 (System 1's shield still enforces amber / all-red / min green)
publish  obs.* for the dashboard, events -> narrator
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

import bus
from agents.junction.env import NEIGHBOURS, obs_vector
from recovery.kalman import Recovery
from orchestrator.brain import Brain
from twin.sim import J


class Cycle(TypedDict, total=False):
    t: float
    gt: dict
    perc: dict
    est: dict
    obs: dict
    decisions: dict
    events: list


def build(brain: Brain, narrator=None, publish: bool = True, send_acts: bool = True):
    """send_acts=False is observe-only mode: decide and publish obs/events, but never write act.*
    (used until the act format is agreed with System 1's bridge)."""
    rec = Recovery(J)

    def sense(s: Cycle) -> dict:
        gt = {j: bus.latest(bus.gt(j), bus.GroundTruth) for j in J}
        pc = {j: bus.latest(bus.perc(j), bus.Perc) for j in J}
        t = max((g.t for g in gt.values() if g), default=0.0)
        return {"t": t, "gt": gt, "perc": pc}

    def recover(s: Cycle) -> dict:
        est = {}
        for j in J:
            g, p = s["gt"][j], s["perc"][j]
            if g is None:
                est[j] = {"q": [0.0] * 4, "health": 0.0, "fault": "none", "source": "loop"}
                continue
            if p is None:   # no camera feed at all: run on road sensors
                seen, health, fault = [0.0] * 4, 0.0, "blackout"
            else:
                seen, health, fault = [float(x) for x in p.queues], p.health, p.fault
            # road sensors = SUMO induction loops on System 1, coarse (multiples of ~2 vehicles)
            loop = [round(q / 2.0) * 2.0 for q in g.queues]
            q, src = rec.estimate(j, seen, health, g.phase, loop=loop)
            est[j] = {"q": q, "health": health, "fault": fault, "source": src}
        return {"est": est}

    def decide(s: Cycle) -> dict:
        obs = {}
        for j in J:
            g, e = s["gt"][j], s["est"][j]
            if g is None:
                continue
            pw = g.ped_wait
            if not any(pw) and g.ped_max_wait_s > 0:
                # System 1's bridge may not send waits per crossing group: derive them.
                # peds_waiting is N,E,S,W; people on N/S kerbs cross the main road (served by green 1),
                # people on E/W kerbs cross the side street (served by green 0).
                ns, ew = g.peds_waiting[0] + g.peds_waiting[2], g.peds_waiting[1] + g.peds_waiting[3]
                pw = [g.ped_max_wait_s if ew else 0.0, g.ped_max_wait_s if ns else 0.0]
                if not any(pw):
                    pw = [g.ped_max_wait_s, g.ped_max_wait_s]
            ped = (g.peds_waiting[1], g.peds_waiting[0], g.ped_max_wait_s)
            nb = [s["est"][n]["q"] if n else None for n in NEIGHBOURS[j]]
            obs[j] = {"est": e["q"], "health": e["health"], "fault": e["fault"], "source": e["source"],
                      "green": g.phase, "stage": g.stage, "timer": g.green_s, "emergency": g.emergency.model_dump(),
                      "ped_wait": pw,
                      "vec": obs_vector(e["q"], g.phase, g.stage, g.green_s, nb, ped, e["health"])}
        if len(obs) < len(J):
            return {"obs": obs, "decisions": {}, "events": []}
        dec = brain.decide(s["t"], obs)
        return {"obs": obs, "decisions": dec, "events": brain.pop_events()}

    def act(s: Cycle) -> dict:
        if publish and send_acts:
            for j, d in s["decisions"].items():
                if d.decided_by not in bus.DECIDERS:      # System 1's bridge drops unknown deciders
                    print(f"[act] unknown decided_by '{d.decided_by}', sending as system2_agent", flush=True)
                    d.decided_by = "system2_agent"
                bus.publish(bus.act(j), bus.Act(junction=j, t=s["t"], phase=d.target, decided_by=d.decided_by,
                                                reason=d.reason, force=d.force))
        return {}

    def pub(s: Cycle) -> dict:
        if publish:
            for j, o in s["obs"].items():
                g, d = s["gt"][j], s["decisions"].get(j)
                bus.publish(bus.obs(j), bus.Obs(
                    junction=j, t=s["t"], queues=o["est"], queue_m=g.queue_m, peds_waiting=g.peds_waiting,
                    ped_max_wait_s=g.ped_max_wait_s, emergency=g.emergency, health=o["health"],
                    fault=o["fault"], source=o["source"], co2_g_s=g.co2_g_s, fuel_ml_s=g.fuel_ml_s,
                    phase=g.phase, stage=g.stage, mode=d.mode if d else "ai", decided_by=d.decided_by if d else "none"))
            for e in s["events"]:
                bus.publish(bus.EVENT, bus.Event(**e))
                if narrator:
                    narrator.push(e)
        return {}

    g = StateGraph(Cycle)
    for name, fn in (("sense", sense), ("recover", recover), ("decide", decide), ("act", act), ("publish", pub)):
        g.add_node(name, fn)
    g.add_edge(START, "sense")
    g.add_edge("sense", "recover")
    g.add_edge("recover", "decide")
    g.add_edge("decide", "act")
    g.add_edge("act", "publish")
    g.add_edge("publish", END)
    return g.compile()
