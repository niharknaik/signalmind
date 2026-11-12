"""Multi-agent corridor environment (train path).

SUMO -> true counts -> fault injector -> Recovery (Kalman) -> observation -> PPO.
One agent per junction, shared policy. Every decision = 5 s of simulation.

Action:  0 = keep current green, 1 = switch to the other green
         (the SignalDriver shield still enforces ped-clear, amber, all-red, min/max green)
Reward:  -(halting vehicles + 1.5 x waiting pedestrians)/10  - small switch penalty
         + 0.5 x mean(neighbours' reward)      <- cooperation
"""
from __future__ import annotations

import numpy as np

from perception.faults import CountFaults
from recovery.kalman import Recovery
from twin.sim import Corridor, J, TOPO, SERVES

OBS_DIM = 20
DECISION_S = 5
NEIGHBOURS = {j: [n for n in (TOPO[j]["up"]["W"], TOPO[j]["up"]["E"])] for j in J}


class CorridorEnv:
    def __init__(self, seconds: int = 1800, fault_rate: float = 0.0, scale: float = 1.0,
                 seed: int = 0, ambulances: bool = False, gui: bool = False, use_libsumo: bool = True,
                 recovery: bool = True):
        self.seconds, self.fault_rate, self.scale, self.seed = seconds, fault_rate, scale, seed
        self.use_recovery = recovery
        self.ambulances, self.gui, self.use_libsumo = ambulances, gui, use_libsumo
        self.c: Corridor | None = None

    # ------------------------------------------------------------------ api
    def reset(self):
        if self.c:
            self.c.close()
        self.c = Corridor(scale=self.scale, ambulances=self.ambulances, seed=self.seed,
                          gui=self.gui, use_libsumo=self.use_libsumo, end=self.seconds).start()
        self.faults = CountFaults(J, rate=self.fault_rate, seed=self.seed)
        self.rec = Recovery(J)
        self.info = {j: {} for j in J}
        self._sense()
        return {j: self.obs(j) for j in J}

    def step(self, actions: dict[str, int]):
        c = self.c
        targets = {j: (1 - c.drivers[j].green if a == 1 else c.drivers[j].green) for j, a in actions.items()}
        return self.step_targets(targets)

    def step_targets(self, targets: dict[str, int | None], force: dict[str, bool] | None = None):
        """Advance one decision with explicit target greens (used by the SignalMind brain)."""
        c = self.c
        switched = {j: int(targets.get(j) is not None and targets[j] != c.drivers[j].green
                           and c.drivers[j].stage == "green") for j in J}
        for _ in range(DECISION_S):
            c.step(targets, force)
            if c.done():
                break
        self._sense()
        own = {j: -(sum(self.true[j]) + 1.5 * self.peds_total[j]) / 10.0 - 0.05 * switched[j] for j in J}
        rew = {}
        for j in J:
            nb = [own[n] for n in NEIGHBOURS[j] if n]
            rew[j] = own[j] + 0.5 * (sum(nb) / len(nb) if nb else 0.0)
        done = c.done()
        return {j: self.obs(j) for j in J}, rew, done, self.info

    def close(self):
        if self.c:
            self.c.close()
            self.c = None

    # -------------------------------------------------------------- sensing
    def _sense(self):
        c, T = self.c, self.c.T
        self.true, self.est, self.peds_total, self.pedinfo = {}, {}, {}, {}
        for j in J:
            ins = TOPO[j]["in"]
            q = [float(T.edge.getLastStepHaltingNumber(e)) for e in ins]
            self.true[j] = q
            seen, fault, health = self.faults.step(j, q, DECISION_S)
            loop = []
            for e in ins:
                n_l = T.edge.getLaneNumber(e)
                n_l -= 1  # lane 0 is the sidewalk
                occ = sum(T.inductionloop.getLastStepOccupancy(f"loop_{e}_{k}") for k in range(1, n_l + 1)) / n_l
                loop.append(round(n_l * occ / 100.0 * 8.0, 2))
            ups = []
            for e in ins:
                moving = T.edge.getLastStepVehicleNumber(e) - T.edge.getLastStepHaltingNumber(e)
                ups.append(moving * DECISION_S * 12.0 / max(T.lane.getLength(e + "_0"), 1.0))
            if self.use_recovery:
                est, src = self.rec.estimate(j, seen, health, c.drivers[j].green, loop=loop,
                                             upstream_arrivals=ups)
            else:   # no recovery: agents see whatever the (possibly broken) camera says
                est, src = [round(x, 2) for x in seen], "camera"
            self.est[j] = est
            pw = c.peds(j)
            self.pedinfo[j] = (len(pw[0]), len(pw[1]), max(pw[0] + pw[1]) if (pw[0] or pw[1]) else 0.0)
            self.peds_total[j] = len(pw[0]) + len(pw[1])
            self.info[j] = {"fault": fault, "health": health, "source": src, "est": est, "true": q}

    def obs(self, j: str) -> np.ndarray:
        d = self.c.drivers[j]
        return obs_vector(self.est[j], d.green, d.stage, d.timer,
                          [self.est[n] if n else None for n in NEIGHBOURS[j]],
                          self.pedinfo[j], self.info[j]["health"])


def obs_vector(est, green, stage, timer, nb_est, ped, health) -> np.ndarray:
    """The 20 features the junction policy sees. Shared by training and the live loop.
    ped = (people waiting for green 0, people waiting for green 1, longest wait s)."""
    o = [x / 20.0 for x in est]
    o += [1.0 if green == 0 else 0.0, 1.0 if green == 1 else 0.0]
    o += [min(timer, 60.0) / 60.0 if stage == "green" else 0.0, 0.0 if stage == "green" else 1.0]
    for n in nb_est:
        o += ([x / 20.0 for x in n] if n else [0.0] * 4)
    p0, p1, pmax = ped
    o += [p0 / 5.0, p1 / 5.0, min(pmax, 90.0) / 60.0]
    o += [health]
    return np.asarray(o, dtype=np.float32)
