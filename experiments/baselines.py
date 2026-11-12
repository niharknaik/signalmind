"""Baseline controllers + a single-episode runner.

    python experiments/baselines.py --controller fixed --seconds 3600
    python experiments/baselines.py --all

Controllers
- fixed        SUMO's static 90 s program (what most Indian junctions still run)
- batcs        BATCS-style: actuated signals that extend green while vehicles keep
               arriving, plus green-wave offsets along the corridor. CoSiCoSt itself
               is not public, so this is our stand-in, not the real BATCS.
- maxpressure  classical throughput controller (also our Fallback mode)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twin.sim import Corridor, J, TOPO, NET_DIR  # noqa: E402
from agents import maxpressure  # noqa: E402

GREENWAVE_SPEED = 11.0   # m/s (~40 km/h)
SPACING = 250


def write_batcs_programs() -> str:
    """Actuated programs with green-wave offsets, as a SUMO additional file."""
    import sumolib
    net = sumolib.net.readNet(os.path.join(NET_DIR, "corridor.net.xml"), withPrograms=True)
    rows = []
    for i, j in enumerate(J):
        prog = list(net.getTLS(j).getPrograms().values())[0]
        st = [p.state for p in prog.getPhases()]
        off = round(i * SPACING / GREENWAVE_SPEED)
        rows.append(f'<tlLogic id="{j}" type="actuated" programID="batcs" offset="{off}">')
        rows.append('  <param key="max-gap" value="3.0"/><param key="detector-gap" value="2.0"/>')
        rows.append(f'  <phase duration="30" minDuration="12" maxDuration="55" state="{st[0]}"/>')
        rows.append(f'  <phase duration="4" state="{st[1]}"/>')
        rows.append(f'  <phase duration="3" state="{st[2]}"/>')
        rows.append(f'  <phase duration="2" state="{"r" * len(st[0])}"/>')
        rows.append(f'  <phase duration="20" minDuration="10" maxDuration="40" state="{st[3]}"/>')
        rows.append(f'  <phase duration="4" state="{st[4]}"/>')
        rows.append(f'  <phase duration="3" state="{st[5]}"/>')
        rows.append(f'  <phase duration="2" state="{"r" * len(st[0])}"/>')
        rows.append('</tlLogic>')
    path = os.path.join(NET_DIR, "batcs.add.xml")
    with open(path, "w", encoding="utf8") as f:
        f.write("<additional>\n" + "\n".join(rows) + "\n</additional>\n")
    return path


def run(controller: str, seconds: int = 3600, scale: float = 1.0, ambulances: bool = False,
        seed: int = 0, policy=None, decide_every: int = 5) -> dict:
    extra = [write_batcs_programs()] if controller == "batcs" else []
    native = controller in ("fixed", "batcs")
    c = Corridor(scale=scale, ambulances=ambulances, seed=seed, native=native,
                 extra_files=extra, end=seconds).start()
    if controller == "batcs":
        for j in J:
            c.T.trafficlight.setProgram(j, "batcs")
    t0 = time.time()
    try:
        while not c.done():
            targets = None
            if not native and int(c.t) % decide_every == 0:
                if controller == "maxpressure":
                    targets = {}
                    for j in J:
                        qin = [c.T.edge.getLastStepHaltingNumber(e) for e in TOPO[j]["in"]]
                        qout = [c.T.edge.getLastStepHaltingNumber(e) for e in TOPO[j]["out"]]
                        targets[j] = maxpressure.choose(qin, qout)
                elif policy is not None:
                    targets = policy(c)
            c.step(targets)
        out = c.k.summary()
    finally:
        c.close()
    out.update(controller=controller, scale=scale, seed=seed, ambulances=ambulances,
               sim_s=seconds, wall_s=round(time.time() - t0, 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="fixed", choices=["fixed", "batcs", "maxpressure"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seconds", type=int, default=3600)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--ambulances", action="store_true")
    a = ap.parse_args()
    for ctl in (["fixed", "batcs", "maxpressure"] if a.all else [a.controller]):
        print(json.dumps(run(ctl, a.seconds, a.scale, a.ambulances)))
