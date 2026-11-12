"""SignalMind message bus: Redis Streams helpers + the shared message schemas.

Both systems import this file. If a message doesn't match its schema, it
fails here, loudly, instead of silently breaking the demo.
"""
from __future__ import annotations

import json
import os
import time
from typing import Literal, Optional, Type, TypeVar

import redis
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()

JUNCTIONS = ["J1", "J2", "J3", "J4"]
APPROACHES = ["N", "E", "S", "W"]
MAXLEN = 1000

Fault = Literal["none", "blackout", "noise", "freeze", "occlusion",
                "mismatch"]   # mismatch: System 2's trust check (camera disagrees with road sensors)
Mode = Literal["ai", "degraded", "fallback"]


# ---------------------------------------------------------------- schemas
class Emergency(BaseModel):
    active: bool = False
    id: Optional[str] = None
    eta_s: Optional[float] = None
    approach: Optional[Literal["N", "E", "S", "W"]] = None


class Perc(BaseModel):
    """perc.J* : raw perception from System 1 (camera + YOLO + Sentinel)."""
    junction: str
    t: float
    queues: list[int] = Field(min_length=4, max_length=4)
    queue_m: list[float] = Field(min_length=4, max_length=4)
    peds_waiting: list[int] = Field(default=[0, 0, 0, 0], min_length=4, max_length=4)
    ambulance: bool = False
    health: float = Field(ge=0, le=1)
    fault: Fault = "none"


class GroundTruth(BaseModel):
    """gt.J* : SUMO truth + loop detectors (backup sensor), every 5 s."""
    junction: str
    t: float
    queues: list[int] = Field(min_length=4, max_length=4)
    queue_m: list[float] = Field(min_length=4, max_length=4)
    outflow: list[int] = Field(default=[0, 0, 0, 0], min_length=4, max_length=4)
    peds_waiting: list[int] = Field(default=[0, 0, 0, 0], min_length=4, max_length=4)
    ped_max_wait_s: float = 0
    emergency: Emergency = Emergency()
    co2_g_s: float = 0
    fuel_ml_s: float = 0
    phase: int = 0                 # current (or last) green: 0 main road, 1 side streets
    stage: str = "green"           # green | clear | amber | allred  (from the shield)
    green_s: float = 0             # seconds in the current stage
    ped_wait: list[float] = [0.0, 0.0]   # longest wait per crossing group (served by green 0 / 1)
    # extras System 1's bridge sends (optional)
    sumo_phase: Optional[int] = None   # raw SUMO program index, debugging only (phase is logical)
    nox_mg_s: float = 0
    pm_mg_s: float = 0
    ped_on_crossing: int = 0
    loop_counts: list[float] = []

    @model_validator(mode="before")
    @classmethod
    def _from_system1(cls, d):
        """Accept System 1's field names (their bridge has its own message definitions)."""
        if isinstance(d, dict):
            d = dict(d)
            if "green_s" not in d and "phase_time_s" in d:
                d["green_s"] = d["phase_time_s"]
            if "stage" not in d and isinstance(d.get("phase_state"), str):
                s = d["phase_state"]
                d["stage"] = "amber" if "y" in s.lower() else ("allred" if s and set(s.lower()) <= {"r"} else "green")
        return d


class Obs(BaseModel):
    """obs.J* : cleaned state after Recovery. What every agent reads."""
    junction: str
    t: float
    queues: list[float] = Field(min_length=4, max_length=4)
    queue_m: list[float] = Field(min_length=4, max_length=4)
    peds_waiting: list[int] = Field(default=[0, 0, 0, 0], min_length=4, max_length=4)
    ped_max_wait_s: float = 0
    emergency: Emergency = Emergency()
    health: float = Field(1.0, ge=0, le=1)
    fault: Fault = "none"
    source: Literal["camera", "kalman+upstream", "loop", "ground_truth"] = "ground_truth"
    co2_g_s: float = 0
    fuel_ml_s: float = 0
    phase: int = 0
    stage: str = "green"           # green | clear | amber | allred
    mode: Mode = "ai"
    decided_by: str = "none"


class Act(BaseModel):
    """act.J* : the arbiter's final, shield-approved decision."""
    junction: str
    t: float = 0.0
    phase: int                     # wanted green: 0 main road, 1 side streets
    decided_by: str
    reason: str = ""
    force: bool = False            # True only for emergencies: may cut min-green (never amber)


class Event(BaseModel):
    # Ours: camera_down, camera_up, mode_change, emergency_start, emergency_end, ped_cap,
    #       spillback, shield_block, info
    # System 1's bridge: max_green_forced, max_red_forced, act_deferred_by_guard,
    #       act_rejected_unknown_phase, ctl_reset_ignored (and possibly more), so this is open.
    type: str
    junction: Optional[str] = None
    t: float = 0.0
    detail: dict = {}


SYSTEM1_EVENTS = {"max_green_forced", "max_red_forced", "act_deferred_by_guard",
                  "act_rejected_unknown_phase", "ctl_reset_ignored"}

# decided_by values System 1's bridge accepts on act.* (anything else is dropped there)
DECIDERS = {"junction_agent", "emergency_agent", "pedestrian_agent", "maxpressure_fallback",
            "coordinator", "system2_agent", "emergency_preempt", "manual_override"}


class FaultCfg(BaseModel):
    rate: float = Field(0.0, ge=0, le=1)
    types: list[Fault] = ["blackout", "noise", "freeze", "occlusion"]


class Ctl(BaseModel):
    cmd: Literal["spawn_ambulance", "surge", "ped_surge", "weather", "reset"]
    factor: Optional[float] = None
    v: Optional[str] = None
    n: Optional[int] = None        # ped_surge: how many people (System 1's bridge adds 20 if missing)


# System 1's bridge KPI field name -> our name
KPI_ALIASES = {"throughput_veh_h": "throughput_vph", "ped_wait_max_s": "ped_max_wait_s",
               "n_vehicles": "vehicles", "ambulance_travel_s": "ambulance_time_s",
               "transitions_without_amber": "amber_violations", "queue_m_total": "max_queue_m"}


class Kpi(BaseModel):
    t: float
    controller: str = "signalmind"
    avg_wait_s: float = 0
    max_queue_m: float = 0
    throughput_vph: float = 0
    fuel_l_h: float = 0
    co2_kg_h: float = 0
    ped_max_wait_s: float = 0
    vehicles: int = 0
    ambulance_time_s: Optional[float] = None
    spillbacks: int = 0
    amber_violations: int = 0
    weather: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _from_system1(cls, d):
        if isinstance(d, dict):
            d = dict(d)
            for theirs, ours in KPI_ALIASES.items():
                if theirs in d and ours not in d:
                    d[ours] = d[theirs]
            if d.get("vehicles") is not None:
                d["vehicles"] = int(float(d["vehicles"]))
        return d


class Say(BaseModel):
    t: float
    text: str
    mood: Literal["green", "amber", "red"] = "green"
    junction: Optional[str] = None


# ------------------------------------------------------------- stream names
def perc(j): return f"perc.{j}"
def gt(j): return f"gt.{j}"
def obs(j): return f"obs.{j}"
def act(j): return f"act.{j}"
def frame(j): return f"frame.{j}"


EVENT, FAULT_CFG, CTL, KPI, SAY = "event", "fault.cfg", "ctl", "kpi", "say"

M = TypeVar("M", bound=BaseModel)


# ----------------------------------------------------------------- helpers
_r: Optional[redis.Redis] = None


def r() -> redis.Redis:
    global _r
    if _r is None:
        _r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
            socket_timeout=10,
        )
    return _r


def publish(stream: str, msg: BaseModel, maxlen: int = MAXLEN) -> str:
    return r().xadd(stream, {"d": msg.model_dump_json()}, maxlen=maxlen, approximate=True)


def latest(stream: str, model: Type[M]) -> Optional[M]:
    """Newest message on a stream, or None. Accepts our {"d": json} form and flat redis fields
    (System 1's bridge has its own message definitions). A bad message returns None, never raises."""
    got = r().xrevrange(stream, count=1)
    if not got:
        return None
    fields = got[0][1]
    try:
        if "d" in fields:
            return model.model_validate_json(fields["d"])
        return model.model_validate(_coerce(fields))
    except Exception as e:
        print(f"[bus] unreadable latest message on {stream}: {str(e).splitlines()[0]}", flush=True)
        return None


class Subscriber:
    """Read new messages from several streams. Starts at 'now' unless told otherwise."""

    def __init__(self, streams: dict[str, Type[BaseModel]], from_start: bool = False):
        self.models = streams
        # Pin each stream's position NOW. (Passing "$" on every poll would only see
        # messages that arrive during the block window and silently drop the rest.)
        self.ids = {}
        for s in streams:
            last = None if from_start else r().xrevrange(s, count=1)
            self.ids[s] = last[0][0] if last else "0-0"

    def poll(self, block_ms: int = 1000):
        res = r().xread(self.ids, block=block_ms, count=100) or []
        out = []
        for stream, entries in res:
            for mid, fields in entries:
                self.ids[stream] = mid
                try:
                    if "d" in fields:
                        msg = self.models[stream].model_validate_json(fields["d"])
                    else:   # plain redis-cli XADD, e.g. checkpoint C2's manual act test
                        msg = self.models[stream].model_validate(_coerce(fields))
                except Exception as e:  # one bad message must never stop a system
                    print(f"[bus] skipped bad message on {stream}: {str(e).splitlines()[0]}", flush=True)
                    continue
                out.append((stream, msg))
        return out


def _coerce(fields: dict) -> dict:
    out = {}
    for k, v in fields.items():
        try:
            out[k] = json.loads(v)
        except (ValueError, TypeError):
            out[k] = v
    return out


def now() -> float:
    return time.time()


if __name__ == "__main__":
    print("ping:", r().ping())
    print("example obs:", json.dumps(Obs(junction="J2", t=0, queues=[5, 2, 7, 1],
                                         queue_m=[34, 12, 48, 6]).model_dump(), indent=1))
