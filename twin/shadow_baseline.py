"""Shadow twin: the BATCS-style controller on the same traffic, in lockstep with the live run.

    python twin/shadow_baseline.py

Same network, same demand, same SUMO seed as the System 1 stand-in. It repeats every
scenario button (ambulance, pedestrians, rush hour, restart) and steps forward until it
reaches the live run's sim time each time a live KPI arrives. Publishes `kpi.batcs`.

The camera-failure slider does not affect it: BATCS-style runs on road sensors, not on
our cameras. The dashboard says this next to the comparison.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bus  # noqa: E402
from experiments.baselines import write_batcs_programs  # noqa: E402
from twin.sim import Corridor, J  # noqa: E402
from twin.system1_standin import MAIN_ROUTES, Standin  # noqa: E402

KPI_B = "kpi.batcs"


class Shadow(Standin):
    """Reuses the stand-in's scenario code (ambulance, surge, pedestrians) with a native BATCS-style controller."""

    def start(self):
        self.c = Corridor(seed=self.seed, end=10 ** 7, native=True,
                          extra_files=[write_batcs_programs()]).start()
        for j in J:
            self.c.T.trafficlight.setProgram(j, "batcs")
        for rid, edges in MAIN_ROUTES.items():
            self.c.T.route.add(rid, edges)
        self.rng = random.Random(self.seed)
        self.n_amb = self.n_extra = self.n_ped = 0

    def handle(self, m: bus.Ctl):
        if m.cmd == "spawn_ambulance":
            self.spawn_ambulance()
        elif m.cmd == "surge":
            self.surge = m.factor or 1.3
        elif m.cmd == "ped_surge":
            self.ped_surge(m.n or 6)
        elif m.cmd == "reset":
            self.c.close(); self.surge = 1.0; self.start()
        print("ctl:", m.cmd, flush=True)

    def run(self):
        sub = bus.Subscriber({bus.KPI: bus.Kpi, bus.CTL: bus.Ctl})
        print("Shadow BATCS-style twin waiting for the live run (kpi on Redis)...", flush=True)
        while True:
            for stream, m in sub.poll(block_ms=2000):
                if stream == bus.CTL:
                    self.handle(m)
                    continue
                target = m.t
                if target < self.c.t - 30:          # live run restarted without a ctl message
                    self.c.close(); self.surge = 1.0; self.start()
                if target - self.c.t > 600:
                    print(f"catching up from t={self.c.t:.0f} to t={target:.0f}...", flush=True)
                while self.c.t < target:
                    self.c.step()
                    if self.surge > 1.0:
                        self.add_surge()
                k = self.c.k.summary()
                bus.publish(KPI_B, bus.Kpi(t=self.c.t, controller="batcs", avg_wait_s=k["avg_wait_s"],
                                           max_queue_m=k["max_queue_m"], throughput_vph=k["throughput_vph"],
                                           fuel_l_h=k["fuel_l_h"], co2_kg_h=k["co2_kg_h"],
                                           ped_max_wait_s=k["ped_max_wait_s"], vehicles=k["vehicles"],
                                           ambulance_time_s=k["ambulance_time_s"], spillbacks=k["spillbacks"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7, help="must match the stand-in / System 1 SUMO seed")
    a = ap.parse_args()
    s = Shadow(speed=1.0, seed=a.seed, gui=False)
    try:
        s.run()
    except KeyboardInterrupt:
        s.c.close()
