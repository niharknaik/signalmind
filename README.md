# SignalMind

Fault-tolerant multi-agent traffic signal control on a 4-junction Bengaluru-style corridor, in a SUMO + 3D digital twin.
CodeSpark 2026 · Domain 3 · PS1.

Plans: [`docs/SIGNALMIND_PLAN.md`](docs/SIGNALMIND_PLAN.md) · [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md)

## What runs where

```
SYSTEM 1 (high GPU)                         REDIS (team password)          SYSTEM 2 (low GPU)
Unity 3D + live SUMO + YOLO  ── gt.* perc.* kpi frame.* ──►   ──►  orchestrator/live.py (LangGraph)
  (or twin/system1_standin.py)  ◄── act.* fault.cfg ctl ──     ◄──  server/app.py + dashboard
```

| Part | File | Status |
|---|---|---|
| Message bus + schemas (shared by both systems) | `bus.py` | done |
| Corridor network, Bengaluru traffic mix (40% two-wheelers, autos, buses), pedestrians, ambulances | `twin/build_net.py` → `twin/net/` | done |
| Simulation core, safety shield, KPI meter | `twin/sim.py` | done |
| System 1 stand-in (headless SUMO, same protocol as Unity side) | `twin/system1_standin.py` | done |
| Fault injector (frames + counts) | `perception/faults.py` | done |
| Recovery agent (Kalman + upstream + loop fusion) | `recovery/kalman.py` | done |
| Junction agents: env, PPO trainer, policy loader | `agents/junction/` | done, training |
| Max-Pressure fallback | `agents/maxpressure.py` | done |
| Brain: modes, Emergency, Pedestrian, Coordinator, arbiter | `orchestrator/brain.py` | done |
| LangGraph supervisor + live loop + Signo narrator | `orchestrator/graph.py`, `live.py`, `narrator.py` | done |
| Dashboard (FastAPI + WebSocket, no build step) | `server/app.py`, `dashboard/index.html` | done |
| Baselines (fixed-time, BATCS-style), experiment grid, charts | `experiments/` | done |
| Unity scene, SUMO↔Unity bridge, cameras, YOLO | System 1 team | to do |

## Setup (System 2)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe redis python-dotenv pydantic eclipse-sumo libsumo traci sumolib numpy pandas matplotlib filterpy fastapi "uvicorn[standard]" langgraph anthropic sumo-rl pettingzoo gymnasium
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
copy .env.example .env      # set REDIS_HOST, REDIS_PASSWORD (and ANTHROPIC_API_KEY for Signo's LLM voice)
.venv\Scripts\python.exe twin\build_net.py
.venv\Scripts\python.exe scripts\check_link.py
```

SUMO comes from pip (`eclipse-sumo`), so no separate install is needed.

## Run

| What | Command |
|---|---|
| Everything on System 2, with the stand-in for System 1 | `powershell -ExecutionPolicy Bypass -File scripts\start_system2.ps1 -Standin` |
| System 2 only (System 1 = Unity) | `powershell -ExecutionPolicy Bypass -File scripts\start_system2.ps1` |
| Dashboard | http://localhost:8000 (other laptops: `http://<system2-ip>:8000`) |
| Watch the stand-in in sumo-gui | `scripts\start_standin.ps1 -Gui` |
| Train the junction agents | `.venv\Scripts\python.exe agents\junction\train.py --iters 120 --a-iters 40 --workers 10` |
| Baselines only | `.venv\Scripts\python.exe experiments\baselines.py --all` |
| Experiment grid | `.venv\Scripts\python.exe experiments\run_grid.py` |
| Charts + summary table | `.venv\Scripts\python.exe experiments\charts.py` |

The live loop takes about 30–90 s to start, because PyTorch and LangGraph are slow to import.

## For the System 1 (Unity) team: what to publish and read

Copy the logic in `twin/system1_standin.py`. It already does everything below except Unity and YOLO.

1. Run SUMO with `twin/net/corridor.sumocfg` through `twin.sim.Corridor(use_libsumo=False)`. Its `SignalDriver` **is the safety shield**, so keep it.
2. Every 5 sim-seconds publish, for J1..J4 in that order (System 2 starts a cycle when `gt.J4` arrives):
   - `gt.J*` = `bus.GroundTruth(**corridor.ground_truth(j))`
   - `perc.J*` = `bus.Perc(...)` from YOLO + Sentinel (camera counts per approach N, E, S, W, plus health 0–1 and fault name)
   - `kpi` = `corridor.k.summary()` mapped into `bus.Kpi`
   - optionally `frame.J*` = `{"jpg": <base64 320x180 JPEG>}` with `maxlen=1`, so camera tiles show on the dashboard
3. Read `act.J*` and call `corridor.step(targets={j: act.phase}, force={j: act.force})`.
4. Read `fault.cfg`: apply `perception/faults.py:apply()` to camera frames at that rate.
5. Read `ctl`: `spawn_ambulance`, `surge`, `ped_surge`, `reset` (see `Standin.poll_bus`).
6. Junction IDs must be exactly `J1`–`J4`, and approach order is always N, E, S, W.

## Honest notes for the pitch

- The BATCS-style baseline is our stand-in (actuated signals + green-wave offsets). CoSiCoSt isn't public, so we never claim to beat the real BATCS.
- The agents train on SUMO counts with simulated camera faults (fast path). The demo path uses camera input from System 1.
- The corridor is a generated 4-junction network with Bengaluru-like traffic. It can be swapped for an OSM export by replacing `twin/net/corridor.net.xml` and keeping the TLS IDs J1–J4.
- All numbers come from simulation.
- "Average wait" = total stopped time of all vehicles ÷ vehicles that entered the corridor (average delay per vehicle). All controllers are scored the same way.
