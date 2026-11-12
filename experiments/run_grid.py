"""Experiment grid: controllers x camera-fault rates x scenarios x seeds.

    python experiments/run_grid.py --controllers fixed,batcs,maxpressure,ppo,signalmind \
        --faults 0,0.1,0.2,0.3 --scenarios normal,surge --seeds 2 --seconds 3600

Controllers
  fixed, batcs   SUMO-native programs (don't use our cameras, so faults don't affect them)
  maxpressure    Max-Pressure on raw (possibly faulty) camera counts, no recovery
  ppo            trained PPO on raw camera counts: no recovery, no modes, no specialists
  signalmind     PPO + Recovery + modes + Emergency/Pedestrian/Coordinator agents
Scenarios: normal (1.0x demand) and surge (1.3x). Ambulances run in every scenario.
Results: experiments/results/grid.csv (one row per episode)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "experiments", "results")
SCALES = {"normal": 1.0, "surge": 1.3}


def episode(job):
    ctl, fault, scen, seed, seconds, ckpt = job
    sys.path.insert(0, ROOT)
    t0 = time.time()
    if ctl in ("fixed", "batcs"):
        from experiments.baselines import run
        k = run(ctl, seconds, SCALES[scen], ambulances=True, seed=seed)
    else:
        import torch
        torch.set_num_threads(1)
        from agents.junction.env import CorridorEnv
        from orchestrator.adapter import brain_obs
        from orchestrator.brain import Brain
        pol = None
        if ctl in ("ppo", "signalmind"):
            from agents.junction.policy import load
            pol = load(ckpt or None)
        full = ctl == "signalmind"
        env = CorridorEnv(seconds=seconds, fault_rate=fault, scale=SCALES[scen], seed=seed,
                          ambulances=True, recovery=full)
        brain = Brain(policy=pol, use_specialists=full, use_modes=full)
        env.reset()
        done = False
        modes = {"ai": 0, "degraded": 0, "fallback": 0}
        while not done:
            o = brain_obs(env)
            dec = brain.decide(env.c.t, o)
            for d in dec.values():
                modes[d.mode] += 1
            _, _, done, _ = env.step_targets({j: d.target for j, d in dec.items()},
                                             {j: d.force for j, d in dec.items()})
        k = env.c.k.summary()
        env.close()
        tot = max(sum(modes.values()), 1)
        k.update({f"share_{m}": round(v / tot, 3) for m, v in modes.items()})
    k.update(controller=ctl, fault_rate=fault, scenario=scen, seed=seed, wall_s=round(time.time() - t0, 1))
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", default="fixed,batcs,maxpressure,ppo,signalmind")
    ap.add_argument("--faults", default="0,0.1,0.2,0.3")
    ap.add_argument("--scenarios", default="normal,surge")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seconds", type=int, default=3600)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="grid.csv")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for ctl, scen, seed in itertools.product(a.controllers.split(","), a.scenarios.split(","), range(a.seeds)):
        # camera faults can't affect controllers that don't use our cameras: run them once
        fl = [0.0] if ctl in ("fixed", "batcs") else [float(f) for f in a.faults.split(",")]
        for f in fl:
            jobs.append((ctl, f, scen, 100 + seed, a.seconds, a.ckpt))
    print(f"{len(jobs)} episodes on {a.workers} workers", flush=True)
    path = os.path.join(OUT, a.out)
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k in pool.imap_unordered(episode, jobs):
            rows.append(k)
            print(f"{k['controller']:12s} fault {k['fault_rate']:.1f} {k['scenario']:6s} seed {k['seed']}  "
                  f"wait {k['avg_wait_s']:6.1f}s  amb {k['ambulance_time_s']}  ped_max {k['ped_max_wait_s']}  "
                  f"co2 {k['co2_kg_h']}  ({k['wall_s']}s)", flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
