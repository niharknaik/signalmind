"""System 1 stand-in: headless SUMO that speaks the same Redis protocol as the Unity side.

    python twin/system1_standin.py --speed 4

Use it to develop and test System 2 before Unity + YOLO are ready, and as the
fallback if System 1 fails during the demo. The real System 1 bridge does the
same thing, but takes perc.* from YOLO instead of simulated camera faults, and
also streams vehicle positions to Unity.

Publishes every 5 sim-seconds:  gt.J*  perc.J*  kpi
Reads:                           act.J* (applied through the safety shield)
                                 fault.cfg (camera fault rate)
                                 ctl (spawn_ambulance, surge, ped_surge, reset)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bus  # noqa: E402
from perception.faults import CountFaults  # noqa: E402
from twin.sim import Corridor, J  # noqa: E402

MAIN_ROUTES = {"r_we": ["W0_J1", "J1_J2", "J2_J3", "J3_J4", "J4_E0"],
               "r_ew": ["E0_J4", "J4_J3", "J3_J2", "J2_J1", "J1_W0"]}


class Standin:
    def __init__(self, speed: float, seed: int, gui: bool):
        self.speed, self.seed, self.gui = speed, seed, gui
        self.surge = 1.0
        self.n_amb = self.n_extra = self.n_ped = 0
        self.rng = random.Random(seed)
        self.start()

    def start(self):
        self.c = Corridor(seed=self.seed, end=10 ** 7, gui=self.gui, use_libsumo=not self.gui).start()
        T = self.c.T
        for rid, edges in MAIN_ROUTES.items():
            T.route.add(rid, edges)
        self.faults = CountFaults(J, rate=0.0, seed=self.seed)
        self.targets, self.force = {}, {}

    # --------------------------------------------------------------- controls
    def spawn_ambulance(self):
        self.n_amb += 1
        vid = f"amb_live{self.n_amb:02d}"
        self.c.T.vehicle.add(vid, "r_we", typeID="DEFAULT_VEHTYPE", departLane="best", departSpeed="max")
        self.c.T.vehicle.setVehicleClass(vid, "emergency")
        self.c.T.vehicle.setColor(vid, (255, 0, 0, 255))
        self.c.T.vehicle.setSpeedFactor(vid, 1.3)
        return vid

    def add_surge(self):
        # extra main-road vehicles on top of the base demand (1000 + 1200 veh/h)
        lam = (self.surge - 1.0) * 2200 / 3600.0
        for _ in range(sum(1 for _ in range(10) if self.rng.random() < lam / 10)):
            self.n_extra += 1
            self.c.T.vehicle.add(f"surge{self.n_extra}", self.rng.choice(list(MAIN_ROUTES)),
                                 typeID="mix", departLane="best")

    def ped_surge(self, n=6):
        T = self.c.T
        for _ in range(n):
            self.n_ped += 1
            i = self.rng.randint(1, 4)
            j = f"J{i}"
            pid = f"pedlive{self.n_ped}"
            T.person.add(pid, f"N{i}_{j}", pos=150)
            T.person.appendWalkingStage(pid, [f"N{i}_{j}", f"{j}_S{i}"], arrivalPos=30)

    def poll_bus(self, sub):
        for stream, m in sub.poll(block_ms=1):
            if stream.startswith("act."):
                self.targets[m.junction], self.force[m.junction] = m.phase, m.force
            elif stream == bus.FAULT_CFG:
                self.faults.set_rate(m.rate, [t for t in m.types if t != "none"])
                print(f"fault rate -> {m.rate:.0%}", flush=True)
            elif stream == bus.CTL:
                print("ctl:", m.cmd, flush=True)
                if m.cmd == "spawn_ambulance":
                    self.spawn_ambulance()
                elif m.cmd == "surge":
                    self.surge = m.factor or 1.3
                elif m.cmd == "ped_surge":
                    self.ped_surge(m.n or 6)
                elif m.cmd == "reset":
                    rate, types = self.faults.rate, self.faults.types   # keep the operator's fault setting
                    self.c.close(); self.surge = 1.0; self.start()
                    self.faults.set_rate(rate, types)

    # ------------------------------------------------------------------ loop
    def run(self):
        streams = {bus.act(j): bus.Act for j in J}
        streams.update({bus.FAULT_CFG: bus.FaultCfg, bus.CTL: bus.Ctl})
        sub = bus.Subscriber(streams)
        print(f"System 1 stand-in running at {self.speed}x real time. Ctrl+C to stop.", flush=True)
        tick = time.time()
        while True:
            self.poll_bus(sub)
            self.c.step(self.targets or None, self.force or None)
            self.targets, self.force = {}, {}
            if self.surge > 1.0:
                self.add_surge()
            if int(self.c.t) % 5 == 0:
                self.publish()
            tick += 1.0 / self.speed
            time.sleep(max(0.0, tick - time.time()))

    def publish(self):
        c = self.c
        for j in J:
            g = c.ground_truth(j)
            bus.publish(bus.gt(j), bus.GroundTruth(**g))
            seen, fault, health = self.faults.step(j, [float(q) for q in g["queues"]], 5.0)
            ratio = [(s / q) if q else 1.0 for s, q in zip(seen, g["queues"])]
            bus.publish(bus.perc(j), bus.Perc(
                junction=j, t=c.t, queues=[int(round(s)) for s in seen],
                queue_m=[round(m * r, 1) for m, r in zip(g["queue_m"], ratio)],
                peds_waiting=g["peds_waiting"] if health > 0.3 else [0, 0, 0, 0],
                ambulance=g["emergency"]["active"] and health > 0.3,
                health=health, fault=fault))
        k = c.k.summary()
        bus.publish(bus.KPI, bus.Kpi(t=c.t, avg_wait_s=k["avg_wait_s"], max_queue_m=k["max_queue_m"],
                                     throughput_vph=k["throughput_vph"], fuel_l_h=k["fuel_l_h"],
                                     co2_kg_h=k["co2_kg_h"], ped_max_wait_s=k["ped_max_wait_s"],
                                     vehicles=k["vehicles"], ambulance_time_s=k["ambulance_time_s"],
                                     spillbacks=k["spillbacks"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=2.0, help="sim seconds per wall second")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gui", action="store_true", help="open sumo-gui to watch")
    a = ap.parse_args()
    s = Standin(a.speed, a.seed, a.gui)
    try:
        s.run()
    except KeyboardInterrupt:
        s.c.close()
