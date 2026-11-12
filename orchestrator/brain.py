"""SignalMind brain: mode routing + specialist agents + priority arbiter.

Pure Python, no Redis, so the same logic runs in experiments and in the live loop.
Called every 5 s with one observation per junction; returns one decision each.

Priority (highest first):
  1 Safety shield        enforced downstream by SignalDriver, can't be overridden
  2 Emergency agent      green corridor from ETA, then recovery green
  3 Pedestrian guardian  60 s wait cap
  4 Fault mode           Max-Pressure on Recovery estimates when the camera is blind
  5 Coordinator          spillback guard (tighter when a neighbour is degraded)
  6 Junction agent       PPO policy
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agents import maxpressure
from twin.sim import J, SERVES, TOPO

PED_CAP_S = 45            # request walk at 45 s so it starts before 60 s (clear+amber+all-red = 9 s)
EMERGENCY_ETA_S = 15
RECOVER_S = 0             # ablation: a fixed side-street green after ambulances caused gridlock (seed 101)
WAVE_CYCLE_S, WAVE_MAIN_S = 100, 66        # green-wave plan: 100 s cycle, main road green window incl. clearance
WAVE_OFFSET_S = -23                         # favour the busier westbound (inbound) direction; 250 m at ~11 m/s
WAVE_ON_Q, WAVE_OFF_Q, WAVE_DWELL = 18, 8, 12   # avg main-road queue per junction; 12 cycles = 60 s
SPILLBACK_VEH = 55          # ~80% of a 250 m, 2-lane main-road approach
SPILLBACK_VEH_TIGHT = 40    # used when the downstream junction is degraded
HEALTHY, BLIND = 0.7, 0.3
GUARD_EMPTY, GUARD_WAITING = 1, 6          # green road (nearly) empty while others wait -> switch
GUARD_HOLD_MAX = 40                         # don't cut a green that is still draining, up to 40 s
SATURATION_Q, SATURATION_HOLD = 12, 30      # heavy traffic: hold green at least 30 s
FALLBACK_AFTER_S, BACK_TO_AI_CYCLES = 30, 3


@dataclass
class JState:
    mode: str = "ai"
    blind_s: float = 0.0
    healthy_cycles: int = 0
    emergency: str | None = None
    recover_until: float = -1.0
    recover_green: int = 0
    ped_flag: bool = False
    spill_flag: bool = False


@dataclass
class Decision:
    junction: str
    target: int | None
    force: bool
    decided_by: str
    reason: str
    mode: str


@dataclass
class Brain:
    policy: object | None = None           # callable(obs_vec) -> 0 keep / 1 switch
    use_specialists: bool = True
    use_modes: bool = True                 # False = never switch to fallback (plain PPO baseline)
    st: dict = field(default_factory=lambda: {j: JState() for j in J})
    events: list = field(default_factory=list)
    wave: bool = False
    wave_t0: float = 0.0
    _heavy: int = 0
    _light: int = 0

    def _event(self, t, typ, j, **detail):
        self.events.append({"type": typ, "junction": j, "t": t, "detail": detail})

    # ------------------------------------------------------------ routing
    def route_mode(self, t, j, health, dt=5.0):
        s = self.st[j]
        old = s.mode
        if health < BLIND:
            s.blind_s += dt
            s.healthy_cycles = 0
        else:
            s.blind_s = 0.0
        if health >= HEALTHY:
            s.healthy_cycles += 1
        if s.blind_s > FALLBACK_AFTER_S:
            s.mode = "fallback"
        elif s.mode == "fallback":
            if s.healthy_cycles >= BACK_TO_AI_CYCLES:
                s.mode = "ai"
        elif health < HEALTHY:
            s.mode = "degraded"
        else:
            s.mode = "ai"
        if s.mode != old:
            self._event(t, "mode_change", j, frm=old, to=s.mode, health=round(health, 2))
            if old == "ai" and s.mode != "ai":
                self._event(t, "camera_down", j, health=round(health, 2))
            if s.mode == "ai":
                self._event(t, "camera_up", j)
        return s.mode

    # ------------------------------------------------------------ decide
    def decide(self, t: float, obs: dict) -> dict[str, Decision]:
        """obs[j] = {"est": [4 queues], "health": h, "green": 0/1, "stage": str,
                     "emergency": {...}, "ped_wait": [w_main_side_group0, w_group1], "vec": np.ndarray}"""
        out = {}
        if self.use_specialists:
            self._update_wave(t, obs)
        for j in J:
            o = obs[j]
            s = self.st[j]
            mode = self.route_mode(t, j, o["health"]) if self.use_modes else "ai"
            green = o["green"]

            if self.use_specialists:
                # 2 Emergency
                em = o.get("emergency") or {}
                if em.get("active") and em.get("eta_s", 99) <= EMERGENCY_ETA_S:
                    want = SERVES["NESW".index(em["approach"])]
                    if s.emergency != em["id"]:
                        s.emergency = em["id"]
                        self._event(t, "emergency_start", j, id=em["id"], eta_s=em["eta_s"], approach=em["approach"])
                    out[j] = Decision(j, want, True, "emergency_agent", f"ambulance {em['id']} ETA {em['eta_s']} s", mode)
                    continue
                if s.emergency and not em.get("active"):
                    self._event(t, "emergency_end", j, id=s.emergency)
                    s.emergency = None
                    s.recover_until, s.recover_green = t + RECOVER_S, 1 - green
                if t < s.recover_until:
                    out[j] = Decision(j, s.recover_green, False, "emergency_agent", "draining queues after ambulance", mode)
                    continue
                # 3 Pedestrian cap
                pw = o.get("ped_wait", [0.0, 0.0])
                need = [g for g in (0, 1) if pw[g] >= PED_CAP_S and g != green]
                if need:
                    if not s.ped_flag:
                        self._event(t, "ped_cap", j, waited_s=round(max(pw), 0))
                        s.ped_flag = True
                    out[j] = Decision(j, need[0], False, "pedestrian_agent", f"pedestrian waited {max(pw):.0f} s", mode)
                    continue
                s.ped_flag = False

            # 3b Coordinator: corridor green wave in heavy traffic (needs no cameras)
            if self.use_specialists and self.wave:
                off = J.index(j) * WAVE_OFFSET_S
                tgt = 0 if ((t - self.wave_t0 - off) % WAVE_CYCLE_S) < WAVE_MAIN_S else 1
                out[j] = Decision(j, tgt, False, "coordinator", "green wave (heavy traffic)", mode)
                continue

            # 4 Fault mode
            if mode == "fallback":
                tgt = maxpressure.choose(o["est"])
                out[j] = Decision(j, tgt, False, "maxpressure_fallback", "camera blind, using estimates", mode)
                continue

            # 5 Coordinator: spillback guard on the main road
            if self.use_specialists:
                hold = False
                for side, idx in (("E", 3), ("W", 1)):   # our outflow to E arrives on neighbour's W approach
                    nb = TOPO[j]["up"][side]
                    if nb and nb in obs:
                        lim = SPILLBACK_VEH_TIGHT if self.st[nb].mode != "ai" else SPILLBACK_VEH
                        if obs[nb]["est"][idx] >= lim:
                            hold = True
                if hold and green == 0:
                    if not s.spill_flag:
                        self._event(t, "spillback", j)
                        s.spill_flag = True
                    out[j] = Decision(j, 1, False, "coordinator", "downstream road nearly full", mode)
                    continue
                s.spill_flag = False

            # 6 Junction agent (PPO), or Max-Pressure if no policy is loaded
            if self.policy is not None:
                a = self.policy(o["vec"])
                tgt = (1 - green) if a == 1 else green
                who, why = "junction_agent", ""
                if self.use_specialists:
                    tgt, who, why = self.guard(o, green, tgt)
            else:
                tgt, who, why = maxpressure.choose(o["est"]), "maxpressure_fallback", ""   # a name System 1 accepts
            out[j] = Decision(j, tgt, False, who, why, mode)
        return out

    @staticmethod
    def guard(o, green, tgt):
        """Coordinator's sanity check on the learned policy (safe-RL style override).
        Keeps the AI in charge in normal cases and bounds its worst-case behaviour."""
        q = o["est"]
        serve = sum(q[a] for a in range(4) if SERVES[a] == green)
        other = sum(q[a] for a in range(4) if SERVES[a] != green)
        timer = o.get("timer", 0.0) if o.get("stage") == "green" else 0.0
        if tgt == green:
            if serve <= GUARD_EMPTY and other >= GUARD_WAITING:
                return 1 - green, "coordinator", f"green road empty, {other:.0f} waiting on the other"
        else:
            if timer < GUARD_HOLD_MAX and serve >= 2 * other + 8:
                return green, "coordinator", f"still draining {serve:.0f} vehicles"
            if timer < SATURATION_HOLD and serve >= SATURATION_Q:
                return green, "coordinator", "heavy traffic: longer green, fewer lost seconds"
        return tgt, "junction_agent", ""

    def _update_wave(self, t, obs):
        """Switch the corridor into / out of green-wave mode with hysteresis."""
        mainq = sum(obs[j]["est"][1] + obs[j]["est"][3] for j in J) / len(J)
        if not self.wave:
            self._heavy = self._heavy + 1 if mainq >= WAVE_ON_Q else 0
            if self._heavy >= WAVE_DWELL:
                self.wave, self.wave_t0, self._light = True, t, 0
                self._event(t, "info", None, text=f"Heavy traffic on the main road (about {mainq:.0f} queued per junction). "
                                                  "Switching the corridor to a coordinated green wave.")
        else:
            self._light = self._light + 1 if mainq <= WAVE_OFF_Q else 0
            if self._light >= WAVE_DWELL:
                self.wave, self._heavy = False, 0
                self._event(t, "info", None, text="Main road has cleared. Junction agents are back in charge.")

    def pop_events(self):
        ev, self.events = self.events, []
        return ev
