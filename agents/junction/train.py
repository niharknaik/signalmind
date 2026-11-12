"""PPO training for the 4 junction agents (shared policy), PyTorch + multiprocessing.

    python agents/junction/train.py --iters 60 --workers 8

Curriculum (from the plan):
  Train A  iterations [0, a_iters):   no camera faults
  Train B  iterations [a_iters, ...): fault rate 10-30 % (all 4 fault types)
Writes checkpoints/ppo_A.pt, ppo_B.pt, ppo_latest.pt and checkpoints/train_log.csv
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from agents.junction.env import OBS_DIM  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.v = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def dist(self, x):
        return torch.distributions.Categorical(logits=self.pi(x))


def rollout(args):
    """Worker: one episode with the given weights. Returns per-agent trajectories + KPIs."""
    state, seed, rate, seconds = args
    torch.set_num_threads(1)
    from agents.junction.env import CorridorEnv
    from twin.sim import J
    net = ActorCritic()
    net.load_state_dict(state)
    env = CorridorEnv(seconds=seconds, fault_rate=rate, seed=seed)
    obs = env.reset()
    traj = {j: {"o": [], "a": [], "lp": [], "r": [], "v": []} for j in J}
    done = False
    with torch.no_grad():
        while not done:
            x = torch.as_tensor(np.stack([obs[j] for j in J]))
            d = net.dist(x)
            a = d.sample()
            lp, v = d.log_prob(a), net.v(x).squeeze(-1)
            nobs, rew, done, _ = env.step({j: int(a[i]) for i, j in enumerate(J)})
            for i, j in enumerate(J):
                t = traj[j]
                t["o"].append(obs[j]); t["a"].append(int(a[i])); t["lp"].append(float(lp[i]))
                t["v"].append(float(v[i])); t["r"].append(rew[j])
            obs = nobs
    kpi = env.c.k.summary()
    env.close()
    return traj, kpi, rate


def gae(r, v, gamma=0.98, lam=0.95):
    adv, last = np.zeros(len(r), np.float32), 0.0
    for t in reversed(range(len(r))):
        nv = v[t + 1] if t + 1 < len(v) else 0.0
        delta = r[t] + gamma * nv - v[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.asarray(v, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--a-iters", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=1800)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    torch.manual_seed(0)
    net = ActorCritic()
    if a.resume:
        net.load_state_dict(torch.load(a.resume))
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    log_path = os.path.join(CKPT, "train_log.csv")
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="")
    w = csv.writer(logf)
    if new_log:
        w.writerow(["iter", "phase", "fault_rate", "mean_reward", "avg_wait_s", "ped_max_wait_s",
                    "throughput_vph", "co2_kg_h", "amber_violations", "wall_s"])
    pool = mp.get_context("spawn").Pool(a.workers)
    rates_b = [0.1, 0.2, 0.3]
    for it in range(a.iters):
        t0 = time.time()
        phase = "A" if it < a.a_iters else "B"
        state = {k: v.clone() for k, v in net.state_dict().items()}
        jobs = [(state, it * 100 + k, 0.0 if phase == "A" else rates_b[k % 3], a.seconds) for k in range(a.workers)]
        results = pool.map(rollout, jobs)
        O, A_, LP, ADV, RET, rews = [], [], [], [], [], []
        for traj, kpi, rate in results:
            for t in traj.values():
                adv, ret = gae(t["r"], t["v"])
                O += t["o"]; A_ += t["a"]; LP += t["lp"]; ADV += list(adv); RET += list(ret)
                rews.append(np.mean(t["r"]))
        O = torch.as_tensor(np.stack(O)); A_ = torch.as_tensor(A_); LP = torch.as_tensor(LP)
        ADV = torch.as_tensor(ADV); RET = torch.as_tensor(RET)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(O)
        for _ in range(4):
            perm = torch.randperm(n)
            for s in range(0, n, 1024):
                idx = perm[s:s + 1024]
                d = net.dist(O[idx])
                ratio = torch.exp(d.log_prob(A_[idx]) - LP[idx])
                pg = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 0.8, 1.2) * ADV[idx]).mean()
                vl = ((net.v(O[idx]).squeeze(-1) - RET[idx]) ** 2).mean()
                loss = pg + 0.5 * vl - 0.01 * d.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        kpis = [k for _, k, _ in results]
        row = [it, phase, np.mean([r for _, _, r in results]), round(float(np.mean(rews)), 4),
               round(np.mean([k["avg_wait_s"] for k in kpis]), 2), round(np.mean([k["ped_max_wait_s"] for k in kpis]), 1),
               round(np.mean([k["throughput_vph"] for k in kpis]), 1), round(np.mean([k["co2_kg_h"] for k in kpis]), 2),
               sum(k["amber_violations"] for k in kpis), round(time.time() - t0, 1)]
        w.writerow(row); logf.flush()
        print("iter %d [%s] reward %.3f  avg_wait %.1fs  ped_max %.0fs  (%.0fs)" % (it, phase, row[3], row[4], row[5], row[9]), flush=True)
        torch.save(net.state_dict(), os.path.join(CKPT, "ppo_latest.pt"))
        if it == a.a_iters - 1:
            torch.save(net.state_dict(), os.path.join(CKPT, "ppo_A.pt"))
    torch.save(net.state_dict(), os.path.join(CKPT, "ppo_B.pt"))
    pool.close()
    logf.close()


if __name__ == "__main__":
    main()
