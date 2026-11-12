"""System 2 live loop: runs the LangGraph supervisor every time System 1 publishes a new cycle.

    python orchestrator/live.py                  # SignalMind (PPO + recovery + specialists)
    python orchestrator/live.py --no-policy      # Max-Pressure junctions + specialists only

Works the same whether System 1 is Unity + YOLO or twin/system1_standin.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bus  # noqa: E402
from orchestrator.brain import Brain  # noqa: E402
from orchestrator.graph import build  # noqa: E402
from orchestrator.narrator import Narrator  # noqa: E402
from twin.sim import J  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--no-policy", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--observe", action="store_true",
                    help="observe-only: decide and show on the dashboard, but never write act.* "
                         "(also set by SIGNALMIND_OBSERVE=1 in .env)")
    a = ap.parse_args()
    observe = a.observe or os.getenv("SIGNALMIND_OBSERVE", "0") == "1"
    pol = None
    if not a.no_policy:
        from agents.junction.policy import load
        pol = load(a.ckpt or None)
        print("policy:", pol.path, flush=True)
    brain = Brain(policy=pol)
    narrator = Narrator(use_llm=False if a.no_llm else None)
    graph = build(brain, narrator, send_acts=not observe)
    print("narrator:", f"Claude ({narrator.model})" if narrator.client else "templates", flush=True)
    print("mode:", "OBSERVE ONLY (no act.* written)" if observe else "CONTROL (writing act.*)", flush=True)
    bus.r().set("sm.mode", "observe" if observe else "control")   # shown on the dashboard
    bus.publish(bus.SAY, bus.Say(t=0, text=("Signo here, in observe-only mode: I'm watching the corridor and showing my decisions, "
                                            "but not changing System 1's lights yet.") if observe else
                                 "Hi, I'm Signo. All four junctions are online and I'm watching the corridor.",
                                 mood="amber" if observe else "green"))
    # Start a cycle once every junction has published for the same sim time (or 1.5 s after the
    # first one, so one silent junction can't stall the corridor). Publish order doesn't matter.
    streams = {bus.gt(j): bus.GroundTruth for j in J}
    streams[bus.EVENT] = bus.Event          # System 1's guard events, so Signo can explain them
    sub = bus.Subscriber(streams)
    print("waiting for System 1 (gt.* on Redis)...", flush=True)
    n, t_last, sim_t = 0, time.time(), -1.0
    pending: dict[str, float] = {}
    first_seen = 0.0
    while True:
        msgs = sub.poll(block_ms=500)
        for _stream, m in msgs:
            if _stream == bus.EVENT:
                if m.type in bus.SYSTEM1_EVENTS:   # ours are narrated where they're raised
                    narrator.push(m.model_dump())
                continue
            if not pending:
                first_seen = time.time()
            pending[m.junction] = m.t
        if not pending:
            if time.time() - t_last > 10:
                print("no data from System 1 for 10 s. Is the bridge or stand-in running?", flush=True)
                t_last = time.time()
            continue
        if len(pending) < len(J) and time.time() - first_seen < 1.5:
            continue
        t_last = time.time()
        new_t = max(pending.values())
        pending.clear()
        if new_t < sim_t - 30:
            # System 1 restarted its simulation: start from a clean brain and filters,
            # otherwise timers like "recover until t=85" would freeze junctions
            print(f"simulation reset detected (t {sim_t:.0f} -> {new_t:.0f}), resetting agents", flush=True)
            brain = Brain(policy=pol)
            graph = build(brain, narrator, send_acts=not observe)
            bus.publish(bus.SAY, bus.Say(t=new_t, text="Simulation restarted. All junctions back on AI control.",
                                         mood="green"))
        sim_t = new_t
        t0 = time.perf_counter()
        out = graph.invoke({})
        n += 1
        if n % 12 == 0:
            modes = {j: d.mode for j, d in out["decisions"].items()}
            who = {j: d.decided_by for j, d in out["decisions"].items()}
            print(f"t={out['t']:.0f}s  cycle {1000 * (time.perf_counter() - t0):.0f} ms  modes {modes}  by {who}", flush=True)


if __name__ == "__main__":
    main()
