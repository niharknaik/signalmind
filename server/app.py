"""Dashboard backend: Redis -> WebSocket, plus control endpoints.

    python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
    open http://localhost:8000

GET  /            dashboard
WS   /ws          state snapshot every 0.5 s (junctions, kpi, kpi history, feed, frames)
POST /api/fault   {"rate": 0.3}
POST /api/ctl     {"cmd": "spawn_ambulance" | "surge" | "ped_surge" | "reset", "factor": 1.3}
GET  /api/results experiment summary from experiments/results/grid.csv (if present)
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import threading
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import bus  # noqa: E402

J = bus.JUNCTIONS
KPI_B = "kpi.batcs"   # shadow twin: BATCS-style on the same traffic (twin/shadow_baseline.py)
STATE = {"junctions": {}, "kpi": None, "kpi_hist": deque(maxlen=240), "kpi_b": None, "kpi_b_hist": deque(maxlen=240),
         "feed": deque(maxlen=30),
         "events": deque(maxlen=50), "frames": {}, "fault_rate": 0.0, "surge": 1.0, "updated": 0.0}
LOCK = threading.Lock()


def _add_kpi(d):
    """Append a KPI sample. If sim time jumped backwards, the simulation was reset: clear history."""
    last = STATE["kpi"]
    if last and d["t"] < last["t"] - 30:
        STATE["kpi_hist"].clear()
        STATE["kpi_b_hist"].clear()
        STATE["kpi_b"] = None
        STATE["feed"].clear()
        STATE["events"].clear()
        STATE["junctions"].clear()
    STATE["kpi"] = d
    STATE["kpi_hist"].append(d)


def reader():
    streams = {}
    for j in J:
        streams[bus.obs(j)] = bus.Obs
        streams[bus.frame(j)] = None
    streams.update({bus.KPI: bus.Kpi, KPI_B: bus.Kpi, bus.SAY: bus.Say, bus.EVENT: bus.Event, bus.FAULT_CFG: bus.FaultCfg})
    ids = {}
    for s in streams:   # pin positions now; "$" on every read would drop messages between reads
        last = bus.r().xrevrange(s, count=1)
        ids[s] = last[0][0] if last else "0-0"
    # start with the last 3 minutes of history so a fresh page isn't empty (older runs are ignored)
    since = f"{int((time.time() - 180) * 1000)}-0"
    for _id, m in bus.r().xrange(bus.SAY, min=since):
        STATE["feed"].appendleft(json.loads(m["d"]))
    for _id, m in bus.r().xrange(bus.KPI, min=since):
        _add_kpi(json.loads(m["d"]))
    while True:
        try:
            res = bus.r().xread(ids, block=1000, count=200) or []
        except Exception as e:
            print("redis read failed:", e, flush=True)
            time.sleep(2)
            continue
        with LOCK:
            for stream, entries in res:
                for mid, f in entries:
                    ids[stream] = mid
                    if stream.startswith("frame."):
                        STATE["frames"][stream.split(".")[1]] = f.get("jpg", "")
                        continue
                    d = json.loads(f["d"]) if "d" in f else bus._coerce(f)
                    if stream in (bus.KPI, KPI_B):
                        try:   # normalise System 1's KPI field names to ours
                            d = bus.Kpi.model_validate(d).model_dump()
                        except Exception as e:
                            print("bad kpi message skipped:", str(e).splitlines()[0], flush=True)
                            continue
                    if stream.startswith("obs."):
                        STATE["junctions"][d["junction"]] = d
                    elif stream == bus.KPI:
                        _add_kpi(d)
                    elif stream == KPI_B:
                        STATE["kpi_b"] = d
                        STATE["kpi_b_hist"].append(d)
                    elif stream == bus.SAY:
                        STATE["feed"].appendleft(d)
                    elif stream == bus.EVENT:
                        STATE["events"].appendleft(d)
                    elif stream == bus.FAULT_CFG:
                        STATE["fault_rate"] = d["rate"]
            STATE["updated"] = time.time()
        try:
            STATE["mode"] = bus.r().get("sm.mode")   # "observe" | "control", set by orchestrator/live.py
        except Exception:
            pass


app = FastAPI(title="SignalMind")


@app.on_event("startup")
def _start():
    threading.Thread(target=reader, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "dashboard", "index.html"))


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            with LOCK:
                snap = {"junctions": STATE["junctions"], "kpi": STATE["kpi"], "kpi_hist": list(STATE["kpi_hist"]),
                        "kpi_b": STATE["kpi_b"], "kpi_b_hist": list(STATE["kpi_b_hist"]),
                        "feed": list(STATE["feed"]), "events": list(STATE["events"])[:10], "frames": STATE["frames"],
                        "fault_rate": STATE["fault_rate"], "surge": STATE["surge"],
                        "mode": STATE.get("mode"),
                        "age_s": round(time.time() - STATE["updated"], 1) if STATE["updated"] else None}
            await sock.send_text(json.dumps(snap))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


class FaultReq(BaseModel):
    rate: float


class CtlReq(BaseModel):
    cmd: str
    factor: float | None = None
    n: int | None = None


@app.post("/api/fault")
def set_fault(req: FaultReq):
    rate = max(0.0, min(0.5, req.rate))
    bus.publish(bus.FAULT_CFG, bus.FaultCfg(rate=rate))
    STATE["fault_rate"] = rate
    return {"ok": True, "rate": rate}


@app.post("/api/ctl")
def ctl(req: CtlReq):
    n = req.n if req.n is not None else (6 if req.cmd == "ped_surge" else None)   # System 1 adds 20 if n is missing
    m = bus.Ctl(cmd=req.cmd, factor=req.factor, n=n)
    bus.publish(bus.CTL, m)
    if req.cmd == "surge":
        STATE["surge"] = req.factor or 1.3
    if req.cmd == "reset":
        STATE["surge"] = 1.0
    return {"ok": True}


@app.get("/api/results")
def results():
    path = os.path.join(ROOT, "experiments", "results", "grid.csv")
    if not os.path.exists(path):
        return JSONResponse({"available": False})
    rows = list(csv.DictReader(open(path)))
    agg = {}
    for r in rows:
        key = (r["controller"], r["scenario"], float(r["fault_rate"]))
        agg.setdefault(key, []).append(r)
    out = []
    for (ctl, scen, fr), rs in sorted(agg.items()):
        def m(k):
            v = [float(x[k]) for x in rs if x.get(k) not in ("", "None", None)]
            return round(sum(v) / len(v), 2) if v else None
        out.append({"controller": ctl, "scenario": scen, "fault_rate": fr, "avg_wait_s": m("avg_wait_s"),
                    "fuel_l_h": m("fuel_l_h"), "co2_kg_h": m("co2_kg_h"), "ambulance_time_s": m("ambulance_time_s"),
                    "ped_max_wait_s": m("ped_max_wait_s"), "throughput_vph": m("throughput_vph"), "n": len(rs)})
    return {"available": True, "rows": out}
