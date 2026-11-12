"""Corridor simulation core used by every controller, the experiments and the live loop.

- Corridor:      starts SUMO (libsumo for speed, traci for the live bridge), reads
                 ground truth per junction, keeps the KPI meter.
- SignalDriver:  the SAFETY SHIELD. Controllers only choose a target green
                 (0 = main road, 1 = side streets). The driver always plays
                 pedestrian clearance -> amber 3 s -> all-red 2 s, and enforces
                 min green 10 s / max green 60 s. Nothing can skip amber.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import sumo

os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NET_DIR = os.path.join(ROOT, "twin", "net")
CFG = os.path.join(NET_DIR, "corridor.sumocfg")
J = ["J1", "J2", "J3", "J4"]
APPROACH = ["N", "E", "S", "W"]

# Shield timings (seconds)
MIN_GREEN, MAX_GREEN, PED_CLEAR, AMBER, ALL_RED = 10, 60, 4, 3, 2


def topology():
    """Incoming / outgoing edges per junction, ordered N, E, S, W."""
    main = ["W0"] + J + ["E0"]
    topo = {}
    for i, j in enumerate(J, 1):
        w, e = main[i - 1], main[i + 1]
        topo[j] = {
            "in": [f"N{i}_{j}", f"{e}_{j}", f"S{i}_{j}", f"{w}_{j}"],
            "out": [f"{j}_N{i}", f"{j}_{e}", f"{j}_S{i}", f"{j}_{w}"],
            "up": {"E": e if e in J else None, "W": w if w in J else None},
        }
    return topo


TOPO = topology()
# approach index -> which green serves it: N,S = side (1), E,W = main (0)
SERVES = [1, 0, 1, 0]


@dataclass
class SignalDriver:
    """Per-junction shield + phase sequencer."""
    tl: str
    greens: list[str]
    clears: list[str]
    ambers: list[str]
    green: int = 0                 # current (or last) green index
    stage: str = "green"           # green | clear | amber | allred
    timer: float = 0.0             # seconds in the current stage
    target: int = 0
    blocked: int = 0               # requests the shield refused (too early)
    history: list = field(default_factory=list)

    def request(self, target: int, force: bool = False) -> bool:
        """Ask for a green. Returns False if the shield blocks it for now."""
        if target == self.green or self.stage != "green":
            self.target = self.green if self.stage == "green" else self.target
            return target == self.green
        if self.timer < MIN_GREEN and not force:
            self.blocked += 1
            return False
        self.target = target
        self.stage, self.timer = "clear", 0.0
        return True

    def state(self) -> str:
        if self.stage == "green":
            return self.greens[self.green]
        if self.stage == "clear":
            return self.clears[self.green]
        if self.stage == "amber":
            return self.ambers[self.green]
        return "r" * len(self.greens[0])

    def tick(self, dt: float, other_waiting: bool) -> None:
        self.timer += dt
        if self.stage == "green" and self.timer >= MAX_GREEN and other_waiting:
            self.target = 1 - self.green
            self.stage, self.timer = "clear", 0.0
        elif self.stage == "clear" and self.timer >= PED_CLEAR:
            self.stage, self.timer = "amber", 0.0
        elif self.stage == "amber" and self.timer >= AMBER:
            self.stage, self.timer = "allred", 0.0
        elif self.stage == "allred" and self.timer >= ALL_RED:
            self.green, self.stage, self.timer = self.target, "green", 0.0


class Corridor:
    def __init__(self, scale: float = 1.0, ambulances: bool = False, seed: int = 0,
                 gui: bool = False, use_libsumo: bool = True, native: bool = False,
                 extra_files: list[str] | None = None, end: int = 3600):
        self.scale, self.seed, self.gui, self.end = scale, seed, gui, end
        self.native = native          # True = let SUMO run its own programs (baselines)
        routes = ["corridor.rou.xml"] + (["ambulance.rou.xml"] if ambulances else [])
        self.routes = ",".join(os.path.join(NET_DIR, f) for f in routes)
        self.extra = extra_files or []
        if use_libsumo and not gui:
            import libsumo as T
        else:
            import traci as T
        self.T = T
        self.t = 0.0
        self.drivers: dict[str, SignalDriver] = {}
        self.k = KpiMeter()

    # ------------------------------------------------------------ lifecycle
    def start(self):
        binary = os.path.join(sumo.SUMO_HOME, "bin", "sumo-gui" if self.gui else "sumo")
        adds = [os.path.join(NET_DIR, "detectors.add.xml")] + self.extra
        cmd = [binary, "-c", CFG, "-r", self.routes, "-a", ",".join(adds),
               "--scale", str(self.scale), "--seed", str(self.seed), "--end", str(self.end),
               "--no-step-log", "true", "--no-warnings", "true", "--duration-log.disable", "true"]
        self.T.start(cmd)
        T = self.T
        for j in J:
            logic = T.trafficlight.getAllProgramLogics(j)[0]
            st = [p.state for p in logic.phases]
            # netconvert program: 0 main green, 1 main ped-clear, 2 main amber, 3 side green, 4, 5
            self.drivers[j] = SignalDriver(j, greens=[st[0], st[3]], clears=[st[1], st[4]],
                                           ambers=[st[2], st[5]])
        self._prev_state = {j: T.trafficlight.getRedYellowGreenState(j) for j in J}
        return self

    def close(self):
        try:
            self.T.close()
        except Exception:
            pass

    def done(self) -> bool:
        return self.t >= self.end or self.T.simulation.getMinExpectedNumber() <= 0

    # ---------------------------------------------------------------- step
    def step(self, targets: dict[str, int] | None = None, force: dict[str, bool] | None = None):
        """Advance 1 s. targets: junction -> wanted green (0 main / 1 side)."""
        T = self.T
        if not self.native:
            for j, d in self.drivers.items():
                if targets and j in targets and targets[j] is not None:
                    d.request(targets[j], force=bool(force and force.get(j)))
                other = self._demand(j, 1 - d.green) > 0
                d.tick(1.0, other)
                T.trafficlight.setRedYellowGreenState(j, d.state())
        T.simulationStep()
        self.t = T.simulation.getTime()
        self.k.update(self)

    # ----------------------------------------------------------- sensing
    def _demand(self, j, green_idx):
        n = 0
        for a, e in enumerate(TOPO[j]["in"]):
            if SERVES[a] == green_idx:
                n += self.T.edge.getLastStepHaltingNumber(e)
        return n

    def queue_m(self, edge: str) -> float:
        T = self.T
        L = T.lane.getLength(edge + "_0")
        far = 0.0
        for v in T.edge.getLastStepVehicleIDs(edge):
            if T.vehicle.getSpeed(v) < 0.5:
                far = max(far, L - T.vehicle.getLanePosition(v))
        return round(far, 1)

    def peds(self, j):
        """People waiting at this junction, per crossing group: main-road crossings
        (served by side green, idx 1) and side-street crossings (served by main green, idx 0)."""
        T = self.T
        wait = {0: [], 1: []}
        for p in T.person.getIDList():
            road = T.person.getRoadID(p)
            if road.startswith(f":{j}_w") and T.person.getWaitingTime(p) > 0:
                nxt = T.person.getNextEdge(p) if hasattr(T.person, "getNextEdge") else ""
                # crossing edges are :Jk_c<n>. Main-road crossings run north-south.
                grp = 1 if self._crosses_main(j, nxt) else 0
                wait[grp].append(T.person.getWaitingTime(p))
        return wait

    def _crosses_main(self, j, crossing_edge):
        if not crossing_edge or "_c" not in crossing_edge:
            return True
        try:
            shape = self.T.lane.getShape(crossing_edge + "_0")
            (x0, y0), (x1, y1) = shape[0], shape[-1]
            return abs(y1 - y0) > abs(x1 - x0)
        except Exception:
            return True

    def peds_waiting_for(self, j, green_idx):
        return len(self.peds(j)[green_idx])

    def emergency(self, j):
        T = self.T
        for a, e in enumerate(TOPO[j]["in"]):
            for v in T.edge.getLastStepVehicleIDs(e):
                if T.vehicle.getVehicleClass(v) == "emergency":
                    dist = T.lane.getLength(e + "_0") - T.vehicle.getLanePosition(v)
                    spd = max(T.vehicle.getSpeed(v), 3.0)
                    return {"active": True, "id": v, "eta_s": round(dist / spd, 1), "approach": APPROACH[a]}
        # one junction further upstream on the main road
        for side, up in TOPO[j]["up"].items():
            if not up:
                continue
            e = f"{TOPO[up]['in'][3 if side == 'W' else 1].split('_')[0]}_{up}"
            for v in T.edge.getLastStepVehicleIDs(e):
                if T.vehicle.getVehicleClass(v) == "emergency":
                    dist = (T.lane.getLength(e + "_0") - T.vehicle.getLanePosition(v)
                            + T.lane.getLength(f"{up}_{j}_0"))
                    spd = max(T.vehicle.getSpeed(v), 3.0)
                    return {"active": True, "id": v, "eta_s": round(dist / spd, 1), "approach": side}
        return {"active": False}

    def ground_truth(self, j) -> dict:
        T = self.T
        ins = TOPO[j]["in"]
        pw = self.peds(j)
        allw = pw[0] + pw[1]
        co2 = sum(T.edge.getCO2Emission(e) for e in ins) / 1000.0          # mg/s -> g/s
        fuel = sum(T.edge.getFuelConsumption(e) for e in ins) / 745.0        # mg/s -> ml/s (petrol)
        d = self.drivers.get(j)
        return {
            "junction": j, "t": self.t,
            "queues": [T.edge.getLastStepHaltingNumber(e) for e in ins],
            "queue_m": [self.queue_m(e) for e in ins],
            "outflow": [T.edge.getLastStepVehicleNumber(e) for e in TOPO[j]["out"]],
            "peds_waiting": [len(pw[1]), len(pw[0]), len(pw[1]), len(pw[0])],
            "ped_max_wait_s": round(max(allw), 1) if allw else 0.0,
            "emergency": self.emergency(j),
            "co2_g_s": round(co2, 2), "fuel_ml_s": round(fuel, 2),
            "phase": d.green if d else 0,
            "stage": d.stage if d else "green",
            "green_s": round(d.timer, 1) if d else 0.0,
            "ped_wait": [round(max(pw[0]), 1) if pw[0] else 0.0, round(max(pw[1]), 1) if pw[1] else 0.0],
        }


class KpiMeter:
    """Accumulates corridor-wide KPIs. Call .summary() at the end."""

    def __init__(self):
        self.steps = 0
        self.halt_veh_s = 0.0
        self.arrived = 0
        self.departed = 0   # vehicles that entered the corridor
        self.fuel_mg = 0.0
        self.co2_mg = 0.0
        self.max_queue_m = 0.0
        self.ped_wait_max = 0.0
        self.ped_wait_sum = 0.0
        self.ped_seen = set()
        self.amb_depart, self.amb_arrive = {}, {}
        self.amber_violations = 0
        self.spillbacks = 0
        self._edges = None
        self._prev = {}

    def update(self, c: Corridor):
        T = c.T
        if self._edges is None:
            self._edges = [e for e in T.edge.getIDList() if not e.startswith(":")]
        self.steps += 1
        for e in self._edges:
            self.halt_veh_s += T.edge.getLastStepHaltingNumber(e)
            self.fuel_mg += T.edge.getFuelConsumption(e)
            self.co2_mg += T.edge.getCO2Emission(e)
        self.arrived += T.simulation.getArrivedNumber()
        self.departed += T.simulation.getDepartedNumber()
        for v in T.simulation.getDepartedIDList():
            if v.startswith("amb_"):
                self.amb_depart[v] = c.t
        for v in T.simulation.getArrivedIDList():
            if v.startswith("amb_"):
                self.amb_arrive[v] = c.t
        if self.steps % 5 == 0:
            for j in J:
                for e in TOPO[j]["in"]:
                    q = c.queue_m(e)
                    self.max_queue_m = max(self.max_queue_m, q)
                    if q > 0.9 * T.lane.getLength(e + "_0"):
                        self.spillbacks += 1
            for p in T.person.getIDList():
                w = T.person.getWaitingTime(p)
                self.ped_wait_max = max(self.ped_wait_max, w)
        # amber check: any VEHICLE link going straight from green to red is a violation
        # (pedestrian crossing signals have no amber, so they are excluded)
        if not hasattr(self, "_veh_links"):
            self._veh_links = {}
            for j in J:
                links = T.trafficlight.getControlledLinks(j)
                self._veh_links[j] = [i for i, l in enumerate(links)
                                      if l and "_w" not in l[0][0] and "_c" not in l[0][1]]
        for j in J:
            s = T.trafficlight.getRedYellowGreenState(j)
            p = self._prev.get(j)
            if p:
                self.amber_violations += sum(1 for i in self._veh_links[j]
                                             if p[i] in "Gg" and s[i] == "r")
            self._prev[j] = s

    def summary(self) -> dict:
        hours = max(self.steps, 1) / 3600.0
        amb = [self.amb_arrive[v] - self.amb_depart[v] for v in self.amb_arrive if v in self.amb_depart]
        return {
            # average delay (stopped time) per vehicle that entered the corridor
            "avg_wait_s": round(self.halt_veh_s / max(self.departed, 1), 2),
            "max_queue_m": round(self.max_queue_m, 1),
            "throughput_vph": round(self.arrived / hours, 1),
            "fuel_l_h": round(self.fuel_mg / 745000.0 / hours, 2),
            "co2_kg_h": round(self.co2_mg / 1e6 / hours, 2),
            "ped_max_wait_s": round(self.ped_wait_max, 1),
            "ambulance_time_s": round(sum(amb) / len(amb), 1) if amb else None,
            "amber_violations": self.amber_violations,
            "spillbacks": self.spillbacks,
            "vehicles": self.arrived,
        }
