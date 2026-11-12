"""Ablation of brain rules on the hardest episodes. python experiments/ablate.py"""
import itertools, multiprocessing as mp, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)

CONFIGS = {
    "wave_east":   {"WAVE_OFFSET_S": 23},
    "wave_west":   {"WAVE_OFFSET_S": -23},
    "wave_sync":   {"WAVE_OFFSET_S": 0},
    "wave_w_90":   {"WAVE_OFFSET_S": -23, "WAVE_CYCLE_S": 90, "WAVE_MAIN_S": 60},
    "no_wave":     {"WAVE_ON_Q": 10**6},
}
CASES = [("normal", 1.0, 0.0, 101), ("surge", 1.3, 0.0, 100), ("surge", 1.3, 0.0, 101)]

def run(job):
    name, (scen, scale, fault, seed) = job
    sys.path.insert(0, ROOT)
    import torch; torch.set_num_threads(1)
    import orchestrator.brain as B
    for k, v in CONFIGS[name].items(): setattr(B, k, v)
    from agents.junction.env import CorridorEnv
    from orchestrator.adapter import brain_obs
    from agents.junction.policy import load
    env = CorridorEnv(seconds=3600, fault_rate=fault, scale=scale, seed=seed, ambulances=True)
    brain = B.Brain(policy=load()); env.reset(); done = False
    while not done:
        dec = brain.decide(env.c.t, brain_obs(env))
        _, _, done, _ = env.step_targets({j: d.target for j, d in dec.items()}, {j: d.force for j, d in dec.items()})
    k = env.c.k.summary(); env.close()
    return name, scen, fault, seed, k

if __name__ == "__main__":
    jobs = list(itertools.product(CONFIGS, CASES))
    with mp.get_context("spawn").Pool(min(len(jobs), 11)) as p:
        for name, scen, fault, seed, k in sorted(p.map(run, jobs), key=lambda x: (x[1], x[2], x[3], x[0])):
            print(f"{scen:6s} fault {fault:.1f} seed {seed}  {name:14s} wait {k['avg_wait_s']:6.1f}  spill {k['spillbacks']:5d}  amb {k['ambulance_time_s']}  ped_max {k['ped_max_wait_s']}")
