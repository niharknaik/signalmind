# SignalMind: Development Plan

**Two systems, 6 hours.**
- **System 1 (high GPU):** Unity 3D twin, cameras, YOLO perception, everything that renders or runs on the GPU.
- **System 2 (low GPU):** SUMO training, multi-agent RL, recovery, specialist agents, LangGraph, backend, dashboard, experiments. All of it runs on the CPU.

Both systems talk only through **Redis Streams** running on System 2. Either side can be built and tested alone against mock messages, but the two are connected **one link at a time from 0:30 onward**, not all at once at the end (see §7).

> ⚠️ **Check with the organisers first.** The problem statement names **CARLA** for the 3D co-simulation. This plan uses **Unity** instead, with SUMO kept as the traffic engine. Ask the organisers if Unity is acceptable before the event. If CARLA is mandatory, swap §4.1 for the CARLA co-sim bridge (`run_synchronization.py`). Everything else stays the same.

---

## 1. The split at a glance

```
┌──────────────────────── SYSTEM 1 · HIGH GPU ────────────────────────┐
│                                                                     │
│  SUMO (live demo instance) ──TraCI──► sumo_unity_bridge.py          │
│        ▲                               │  vehicle + signal state     │
│        │ phases                        ▼  (TCP / localhost, 10 Hz)  │
│        │                        UNITY 3D TWIN (URP)                 │
│        │                        · corridor roads from SUMO net.xml  │
│        │                        · vehicles, ambulance, pedestrians  │
│        │                        · 4 junction pole cameras           │
│        │                        · weather FX (rain / fog / night)   │
│        │                               │ JPEG frames 5 fps × 4      │
│        │                               ▼  (TCP / localhost)         │
│        │                        perception_server.py (GPU)          │
│        │                        · fault injector (on frames)        │
│        │                        · YOLOv10n + ByteTrack              │
│        │                        · lane zones → queues, people, amb  │
│        │                        · Sentinel health score             │
│        │                               │                            │
└────────┼───────────────────────────────┼────────────────────────────┘
         │ act.*                         │ perc.*  frame.*(thumbnails)
═════════╪════════════ REDIS STREAMS (on System 2, LAN) ══════════════
         │                               ▼
┌────────┼──────────────────── SYSTEM 2 · LOW GPU ────────────────────┐
│        │                                                            │
│  Recovery agent (Kalman + upstream + loop fusion) → obs.*           │
│  Junction agents ×4 (PPO inference) · Coordinator                   │
│  Emergency · Pedestrian · Eco agents · Safety shield                │
│  LangGraph supervisor → priority arbiter → act.*                    │
│  Narrator (Claude Haiku 5.5, async) → say                           │
│  FastAPI + WebSocket → React dashboard (Signo + Pip)                │
│  TRAINING: libsumo × N workers + RLlib PPO (CPU)                    │
│  EXPERIMENTS: baseline + grid runner → results/*.csv → charts       │
└─────────────────────────────────────────────────────────────────────┘
```

**Why the live SUMO runs on System 1:** Unity has to mirror SUMO every step. Keeping them on one machine avoids network jitter in the lockstep. Training uses separate, headless SUMO instances on System 2.

**Demo layout:** System 1's screen shows the Unity 3D view. System 2's screen shows the dashboard. Put them side by side for the judges.

---

## 2. Who works where (4 people)

| Person | System | Role | Owns |
|---|---|---|---|
| **A · 3D engineer** | System 1 | Unity twin | Unity scene, SUMO↔Unity bridge, cameras, weather FX, ambulance and pedestrian models |
| **B · Vision engineer** | System 1 | Perception | Frame transport, YOLO + ByteTrack, lane zones, queue lines, fault injector, Sentinel |
| **C · RL engineer** | System 2 | Intelligence | SUMO network and demand, sumo-rl env, reward, PPO training, Coordinator, shield, Max-Pressure, baselines, experiments |
| **D · App engineer** | System 2 | Orchestration + UI | Redis, bus helpers, Recovery, Emergency / Pedestrian / Eco agents, LangGraph, FastAPI, dashboard, Signo + Pip |

With 2 people, merge A+B on System 1 and C+D on System 2. Drop the stretch items in §9 first.

---

## 3. Before the event (setup only, no project code)

Check the rules. Installs and environment setup are usually allowed. Pre-written project code usually isn't.

### System 1 (high GPU)
| Item | Version / command | Check |
|---|---|---|
| NVIDIA driver + CUDA | CUDA 12.x | `nvidia-smi` |
| Unity Hub + Editor | Unity 2022.3 LTS, URP template | New URP project opens |
| Unity packages | Newtonsoft Json (`com.unity.nuget.newtonsoft-json`) | Package Manager |
| Vehicle / pedestrian assets | Free low-poly cars, an ambulance, a few humanoids (Unity Asset Store) | Imported into a test scene |
| Python | 3.10 | `python --version` |
| SUMO | 1.20+, `SUMO_HOME` set | `sumo --version` |
| Python packages | `pip install ultralytics supervision opencv-python numpy redis python-dotenv traci sumolib` | `yolo predict model=yolov10n.pt source=test.jpg` |
| YOLO weights | `yolov10n.pt` downloaded | Runs on GPU (`device=0`) |
| (Optional) Sumo2Unity | An open-source SUMO–Unity co-sim tool | If it works on your setup, it can replace the custom bridge in §4.1 |

### System 2 (low GPU)
| Item | Version / command | Check |
|---|---|---|
| Python | 3.10 | |
| SUMO | 1.20+, `SUMO_HOME` set | `sumo-gui` opens |
| Python packages | `pip install "ray[rllib]" sumo-rl pettingzoo torch filterpy langgraph anthropic fastapi "uvicorn[standard]" redis python-dotenv pandas plotly matplotlib` | `python -c "import ray, sumo_rl, langgraph"` |
| Redis | Docker: `docker run -d -p 6379:6379 --name redis redis:7` | `redis-cli ping` → `PONG` |
| Node.js | 20 LTS, `npm create vite@latest` works | |
| Front-end packages | `react deck.gl @deck.gl/react recharts` (installed in a throwaway test) | |
| API key | `ANTHROPIC_API_KEY` in `.env` | One test call |
| Map | OSM corridor exported with `osmWebWizard.py` (4 signalised junctions) | Opens in `sumo-gui` |

### Both
- Same Wi-Fi/LAN. Note System 2's IP and put it in `.env` as `REDIS_HOST`.
- Open the Windows firewall for port 6379 on System 2.
- An empty Git repo with the folder layout from §5, cloned on both systems.

---

## 4. Interfaces (agree in the first 30 minutes)

### 4.1 SUMO → Unity bridge (System 1, localhost TCP port 5555)

`sumo_unity_bridge.py` steps SUMO, then sends one newline-delimited JSON message per step (step length 0.1 s, sent at 10 Hz):

```jsonc
{
  "t": 1240.3,
  "vehicles": [
    {"id": "veh12", "x": 532.1, "y": 210.7, "angle": 87.0, "speed": 8.2, "type": "car"},
    {"id": "amb_07", "x": 401.0, "y": 212.3, "angle": 90.0, "speed": 14.1, "type": "ambulance"}
  ],
  "persons": [{"id": "ped3", "x": 520.0, "y": 198.4, "angle": 0.0}],
  "tls": {"J1": "GGrrGGrr", "J2": "yyrryyrr", "J3": "rrGGrrGG", "J4": "GGrrGGrr"},
  "weather": "rain"
}
```

- **Coordinates:** SUMO (x, y) → Unity (x, 0, y). Subtract the `netOffset` from `net.xml` so the corridor sits near the origin. SUMO's angle is clockwise from north, so Unity yaw = `angle`.
- **Unity side (`SumoReceiver.cs`):** read lines on a background thread and queue them to the main thread. Spawn or move prefabs by `id` with interpolation (lerp between the last 2 messages). Destroy IDs that disappear. Set traffic-light materials from the `tls` strings.
- **Road geometry (`NetBuilder.cs`):** at startup, read lane shapes from a JSON file exported by `export_lanes.py` (sumolib) and build a flat mesh strip per lane. No OSM import into Unity is needed.

### 4.2 Unity → Perception frames (System 1, localhost TCP port 5556)

- 4 cameras on junction poles, each rendering to a 640×360 `RenderTexture`, captured with `AsyncGPUReadback` at 5 fps per camera.
- Encode JPEG (quality 80). Send a 4-byte length header, then a JSON header line, then the JPEG bytes:
  `{"junction":"J2","t":1240.3,"w":640,"h":360}`

### 4.3 Redis streams (on System 2)

| Stream | Producer | Consumer | Content |
|---|---|---|---|
| `perc.J1..J4` | S1 perception | S2 recovery | Raw counts, queue_m, people, ambulance flag, health, fault |
| `gt.J1..J4` | S1 bridge | S2 recovery, eval | TraCI ground truth + loop detectors (backup sensor) |
| `obs.J1..J4` | S2 recovery | S2 agents, dashboard | Cleaned state (full contract below) |
| `act.J1..J4` | S2 arbiter | S1 bridge | `{"junction":"J2","phase":2,"decided_by":"junction_agent"}` |
| `event` | Any agent | Supervisor, narrator, dashboard | `{"type":"camera_down","junction":"J2",...}` |
| `fault.cfg` | Dashboard slider | S1 perception | `{"rate":0.2,"types":["blackout","noise","freeze","occlusion"]}` |
| `ctl` | Dashboard buttons | S1 bridge | `{"cmd":"spawn_ambulance"}` / `{"cmd":"surge","factor":1.3}` / `{"cmd":"weather","v":"fog"}` |
| `kpi` | S1 bridge | Dashboard | Wait, queue, throughput, fuel, CO₂, ped wait, every 5 s |
| `frame.J1..J4` | S1 perception | Dashboard | 320×180 JPEG thumbnail (base64), 1 fps, `MAXLEN 1` |
| `say` | Narrator | Dashboard | `{"text":"J2 camera frozen. Using J1 counts.","mood":"amber"}` |

All streams use `XADD ... MAXLEN ~ 1000` so memory stays bounded.

### 4.4 Observation contract (`obs.J2`)

```jsonc
{
  "junction": "J2", "t": 1240,
  "queues":   [5, 2, 7, 1],      // vehicles N,E,S,W
  "queue_m":  [34, 12, 48, 6],
  "peds_waiting": [0, 3, 0, 1],
  "ped_max_wait_s": 41,
  "emergency": {"active": true, "id": "amb_07", "eta_s": 18, "approach": "W"},
  "health": 0.42, "fault": "freeze",
  "source": "kalman+upstream",   // camera | kalman+upstream | loop
  "co2_g_s": 212.4, "fuel_ml_s": 91.3,
  "phase": 2, "mode": "degraded",  // ai | degraded | fallback
  "decided_by": "emergency_agent"
}
```

`bus.py` holds pydantic models for every message. **Both systems import the same file**, so a schema change shows up as a validation error straight away.

---

## 5. Repository layout

```
signalmind/
├─ .env.example                REDIS_HOST, ANTHROPIC_API_KEY, SUMO_CFG
├─ bus.py                      redis helpers + pydantic schemas        (D)  [both]
├─ twin/
│  ├─ net/                     corridor.net.xml, routes, peds, sumocfg (C)  [both]
│  ├─ export_lanes.py          net.xml → lanes.json for Unity          (A)  [S1]
│  └─ sumo_unity_bridge.py     TraCI ⇄ Unity TCP ⇄ Redis act/kpi/gt    (A)  [S1]
├─ unity/SignalMindTwin/       Unity project                           (A)  [S1]
│  └─ Assets/Scripts/          SumoReceiver.cs, NetBuilder.cs,
│                              CameraStreamer.cs, WeatherFX.cs
├─ perception/
│  ├─ frame_server.py          TCP frames in → pipeline                (B)  [S1]
│  ├─ detector.py              YOLOv10n + ByteTrack + zones            (B)  [S1]
│  ├─ zones/J1..J4.json        lane polygons per camera                (B)  [S1]
│  ├─ faults.py                blackout / noise / freeze / occlusion   (B)  [both]
│  └─ sentinel.py              health score 0–1                        (B)  [S1]
├─ recovery/kalman.py          Kalman + upstream + loop fusion         (D)  [S2]
├─ agents/
│  ├─ junction/env.py          sumo-rl wrapper + reward + fault wrap   (C)  [S2]
│  ├─ junction/train.py        RLlib PPO, shared policy                (C)  [S2]
│  ├─ junction/policy.py       load checkpoint, act(obs)               (C)  [S2]
│  ├─ coordinator.py           offsets + spillback guard               (C)  [S2]
│  ├─ shield.py                amber / all-red / min-max green         (C)  [S2]
│  ├─ maxpressure.py           fallback controller                     (C)  [S2]
│  ├─ emergency.py             ETA + green corridor                    (D)  [S2]
│  ├─ pedestrian.py            60 s cap, walk extension                (D)  [S2]
│  └─ eco.py                   HBEFA aggregation, hotspots             (D)  [S2]
├─ orchestrator/
│  ├─ graph.py                 LangGraph supervisor                    (D)  [S2]
│  ├─ arbiter.py               priority order                          (D)  [S2]
│  └─ narrator.py              Claude Haiku → `say`                    (D)  [S2]
├─ server/app.py               FastAPI + WebSocket bridge              (D)  [S2]
├─ dashboard/                  Vite + React + deck.gl + Recharts       (D)  [S2]
├─ experiments/
│  ├─ baselines.py             fixed-time, BATCS-style, actuated       (C)  [S2]
│  ├─ run_grid.py              controllers × faults × scenarios × seeds(C)  [S2]
│  └─ charts.py                results/*.csv → PNG charts              (C)  [S2]
├─ mocks/                      fake publishers for every stream        (D)  [both]
└─ scripts/
   ├─ start_system1.ps1        bridge + perception (+ open Unity)
   └─ start_system2.ps1        redis + recovery + agents + server + dashboard
```

---

## 6. Six-hour timeline

| Time | System 1 · A (Unity) | System 1 · B (Vision) | System 2 · C (RL) | System 2 · D (App) |
|---|---|---|---|---|
| **0:00–0:30** | All four: agree on §4, create repo, `bus.py` schemas, start Redis, write `mocks/`. **C0: both systems reach Redis (§7.2)** | | | |
| **0:30–1:30** | Unity scene. `export_lanes.py` + `NetBuilder.cs` (roads). `SumoReceiver.cs` spawns cars from the bridge | `frame_server.py` with test images. YOLOv10n + ByteTrack on GPU. Draw lane zones on a Unity screenshot | Corridor net + demand + pedestrians + ambulance vType. sumo-rl env, reward, fault wrapper. **Start train A by 1:30** | `server/app.py` WebSocket. Dashboard shell on mocks: map, KPI strip, Signo feed |
| **1:30–2:30** | Traffic lights from `tls`. 4 pole cameras + `CameraStreamer.cs`. Ambulance + pedestrian prefabs | Queue lines (speed < 0.5 m/s), people count, ambulance flag. Publish `perc.*` | Baselines: fixed-time, BATCS-style, Max-Pressure. KPI + HBEFA logger | `recovery/kalman.py`. Emergency, Pedestrian, Eco agents on `gt.*` data |
| **2:30–3:30** | `WeatherFX.cs` (rain, fog, night). `ctl` commands (spawn ambulance, surge, weather). Polish visuals | `faults.py` on frames + `sentinel.py`. Thumbnails to `frame.*` | **Train B** (fault curriculum 0→30%). `coordinator.py`, `shield.py`, `policy.py` | LangGraph supervisor + arbiter + narrator. Fault slider, ambulance button, heatmap, Pip |
| **checkpoints** | **C1 at 1:30** live SUMO → dashboard · **C2 at 2:30** agents → Unity lights · **C3 at 3:15** YOLO → agents (§7.3) | | | |
| **3:30–4:30** | **C4** controls across systems · **C5** 5-minute soak test · polish (§7.3) | | | |
| **4:30–5:00** | Record Unity demo clip (backup) | Record perception overlay clip | Run experiment grid, `charts.py` | Charts into dashboard, slides |
| **5:00–6:00** | **Code freeze at 5:15.** Backup video, rehearse the demo twice | | | |

### Gates

| Gate | Time | Must be true | If not |
|---|---|---|---|
| 1 | 1:30 | PPO training is running on System 2 | Use sumo-rl's default reward and observation unchanged |
| 2 | 2:30 | Unity shows moving SUMO cars and changing lights | Use `sumo-gui` as the visual and stream its screenshots (`gui.screenshot`) to YOLO |
| 3 | 3:30 | Each system works alone against mocks | Fall back to the train path (ground truth + simulated faults) for the live loop |
| 4 | 4:30 | Full loop runs 5 minutes without crashing | Lead with Max-Pressure + recovery + specialists, and show PPO in the charts |
| 5 | 5:15 | Code freeze | Only charts, slides and video after this |

---

## 7. Connecting the two systems

### 7.1 The idea in one line

System 2 runs a "mailbox" (Redis). Both systems post messages to it and read messages from it. They never talk to each other directly, so connecting them just means **System 1 can reach Redis on System 2**.

```
 SYSTEM 1  ──── Wi-Fi / cable ────►  SYSTEM 2
 (Unity, YOLO)                       Redis mailbox on port 6379
   posts:  gt.*  perc.*  kpi  frame.*        ◄── agents + dashboard read these
   reads:  act.*  fault.cfg  ctl             ◄── agents + dashboard post these
```

### 7.2 Step by step (about 15 minutes, do it before the event too)

**Step 1. Put both laptops on the same network.**
Venue Wi-Fi often blocks laptops from talking to each other. The safest choices, best first:
1. An Ethernet cable straight between the two laptops, or a small travel router.
2. A phone hotspot that both laptops join.
3. Venue Wi-Fi (test it first with step 5).

**Step 2. Find System 2's address.** On System 2, run `ipconfig` and note the **IPv4 Address**, for example `192.168.1.20`.

**Step 3. Start the mailbox on System 2, with a password.**
```powershell
docker run -d --name redis -p 6379:6379 redis:7 redis-server --requirepass <team-password>
```
Redis refuses connections from other machines unless a password is set. Use any team password.

**Step 4. Open the door in System 2's firewall** (PowerShell as Administrator):
```powershell
New-NetFirewallRule -DisplayName "SignalMind Redis" -Direction Inbound -Protocol TCP -LocalPort 6379 -Action Allow
```

**Step 5. Check System 1 can reach it.** On System 1:
```powershell
Test-NetConnection 192.168.1.20 -Port 6379
```
`TcpTestSucceeded : True` means the systems are connected. If it says `False`, check step 1 (same network?) and step 4 (firewall). Plain `ping` can fail even when this works, so trust this test.

**Step 6. Put the same settings on both systems** in `.env`:
```
REDIS_HOST=192.168.1.20
REDIS_PORT=6379
REDIS_PASSWORD=<team-password>
```
On System 2 you can use `REDIS_HOST=localhost`.

**Step 7. Send a hello message across.** Run `scripts/check_link.py` on both systems:
```python
# scripts/check_link.py  ·  run on both: python scripts/check_link.py
import os, socket, time, redis
from dotenv import load_dotenv
load_dotenv()
r = redis.Redis(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD"), decode_responses=True)
print("ping:", r.ping())
me = socket.gethostname()
r.xadd("hello", {"from": me, "at": time.time()}, maxlen=50)
for _id, msg in r.xrevrange("hello", count=5):
    print("seen:", msg["from"])
```
Pass when each system's output lists **both** computer names.

**Step 8. Use simulation time, not clock time.** Every message carries `t` (SUMO seconds), so the two laptops' clocks don't need to match.

**If the network fails during the event:** switch both laptops to the phone hotspot, run `ipconfig` again, update `REDIS_HOST` in `.env` on System 1, and rerun step 5. That takes 2 minutes.

### 7.3 Connection checkpoints (connect early, one link at a time)

Develop separately against `mocks/`, but replace one mock with the real link at each checkpoint below. Keep every mock runnable until its real link passes, so a link that breaks late can be swapped back to its mock in seconds.

| # | Time | Link | How to test | Pass when | If it fails |
|---|---|---|---|---|---|
| **C0** | 0:30 | Both systems reach Redis | §7.2 steps 5 and 7 | Both names appear in `hello` | Use the phone hotspot. Fix before doing anything else |
| **C1** | 1:30 | Live SUMO (S1) → dashboard (S2) | Start the bridge on S1. On S2: `redis-cli -a <team-password> XLEN gt.J1` twice, 5 s apart | Number grows, and the dashboard KPI strip shows real values | Dashboard stays on `mocks/`. A fixes the bridge in parallel |
| **C2** | 2:30 | Agents (S2) → traffic lights in Unity (S1) | On S2: `redis-cli -a <team-password> XADD act.J1 '*' junction J1 phase 2 decided_by manual` | J1 changes in both SUMO and Unity within 1 s | Check junction IDs match (`J1`–`J4`) in net.xml, Unity and `bus.py` |
| **C3** | 3:15 | YOLO counts (S1) → Recovery → agents (S2) | Look at `obs.J*` on the dashboard | `source: camera`, counts within ±2 of `gt.*` | Recovery reads `gt.*` instead. Demo still runs with simulated faults |
| **C4** | 3:45 | Controls (S2) → System 1 | Fault slider to 30%, then "send ambulance" | `health` drops and the mode changes. The ambulance appears in Unity and lights ahead turn green | Trigger faults and the ambulance from a script on S1 |
| **C5** | 4:15 | Soak test | Leave everything running 5 minutes, no touching | No crash, no growing lag, Redis memory stable | Lower camera rate to 3 fps, cap stream lengths |
| **C6** | 4:30 | Backup | Record the full demo once | Video saved on both laptops | — |

By 3:30 the full loop already works, so 3:30–4:30 is for controls, polish and the soak test rather than first-time wiring.

### 7.4 Things that usually break, and the fix

| Symptom | Likely cause | Fix |
|---|---|---|
| `Test-NetConnection` false | Different networks, or venue Wi-Fi isolates devices | Cable or phone hotspot |
| Works, then stops | Laptop changed IP after reconnecting | Rerun `ipconfig`, update `.env` |
| `NOAUTH` or `DENIED` error | Password missing on one side | Same `REDIS_PASSWORD` in both `.env` files |
| Cars float beside the roads in Unity | `netOffset` not subtracted | Read `netOffset` from net.xml in `export_lanes.py` and the bridge |
| Lights don't react to `act.*` | Junction IDs differ | Use the SUMO TLS IDs everywhere, mapped to `J1`–`J4` in `bus.py` |
| Dashboard lags more and more | Streams growing without limit | `maxlen=1000` on every `xadd`, `maxlen=1` on `frame.*` |
| Validation error on a message | Schemas out of date on one side | `git pull` on both, since both import `bus.py` |

---

## 8. Module specs

### System 1

**`sumo_unity_bridge.py` (A)**
- Starts SUMO with `traci.start(["sumo", "-c", cfg, "--step-length", "0.1"])`.
- Each step: read vehicles, persons and TLS states. Send to Unity. Every 50 steps (5 s), publish `gt.*` and `kpi`.
- Reads `act.*` and applies phases with `traci.trafficlight.setPhase`. The shield has already approved the action on System 2.
- Reads `ctl`: `spawn_ambulance` adds a vehicle of type `ambulance` (vClass `emergency`) on the corridor route. `surge` scales demand with `traci.simulation` route insertions.
- KPIs: wait (`getWaitingTime`), queue (`lane.getLastStepHaltingNumber`), fuel and CO₂ (`vehicle.getFuelConsumption`, `getCO2Emission`), pedestrian wait (`person.getWaitingTime`).

**Unity scripts (A)**
- `NetBuilder.cs`: lanes.json → mesh strips + white stop lines.
- `SumoReceiver.cs`: TCP client, a `ConcurrentQueue` to the main thread, a prefab pool keyed by id, lerp positions, light materials.
- `CameraStreamer.cs`: 4 cameras → RenderTexture → `AsyncGPUReadback` → `ImageConversion.EncodeToJPG` → TCP.
- `WeatherFX.cs`: particle rain, fog density, night lighting. Makes the faults look real on screen.

**`detector.py` (B)**
- `model = YOLO("yolov10n.pt")`, `device=0`, `imgsz=640`. COCO classes car, motorcycle, bus, truck, person.
- Ambulance: fastest path is a colour + shape rule on the ambulance prefab, or trusting the `type` from `gt.*`. Fine-tuning a class is a stretch.
- `sv.ByteTrack()` gives IDs. `sv.PolygonZone` per lane gives counts. Speed comes from track displacement between frames, scaled by metres per pixel calibrated once per camera.
- Queue: tracked vehicles inside a lane zone at < 0.5 m/s. Length = farthest queued vehicle from the stop line.

**`faults.py` (B, shared with System 2)**
- `apply(frame, rng, rate, types)`: blackout (zeros), noise (Gaussian σ = 40), freeze (return the previous frame), occlusion (a random black rectangle over 40% of the frame).
- `apply_counts(obs, rng, rate)`: the same faults applied to numbers, for the training path (dropout, noise, stale values).

**`sentinel.py` (B)**
- Entropy < 1.0 → blackout. Frame diff = 0 for 3 frames → freeze. Mean confidence < 0.35 → noise. |camera − loop| / loop > 0.5 → occlusion.
- `health = 1 − weighted sum of flags`, smoothed with an EMA (α = 0.3).

### System 2

**`env.py` + `train.py` (C)**
- `sumo_rl.parallel_env(net_file, route_file, use_gui=False, num_seconds=3600, delta_time=5, reward_fn=custom)`. Set `LIBSUMO_AS_TRACI=1` for speed.
- Observation = own lane densities + queues + phase one-hot + neighbours' queues + health flag + pedestrian waiting.
- Reward from the main plan, §6. Fault wrapper calls `faults.apply_counts` at rate p, with a curriculum: 0% (train A), then 10→30% (train B).
- RLlib PPO, one shared policy across the 4 agents, `num_env_runners = cores − 2`, checkpoint every 10 iterations.

**`shield.py` (C):** a phase change always goes through yellow 3 s, then all-red 2 s. Min green 10 s, max red 90 s. Returns the safe action plus a `blocked` flag that gets logged.

**`maxpressure.py` (C):** choose the phase with the largest (incoming queue − outgoing queue).

**`baselines.py` (C)**
- Fixed-time: the SUMO default program with 60 s green.
- **BATCS-style:** SUMO `actuated` type with detectors, plus green-wave offsets of (distance / 11 m/s) between junctions.

**`kalman.py` (D):** one filter per approach. State = [queue, inflow rate]. Prediction uses the upstream junction's outflow. The update weights the camera by `health` and the loop detector by `1 − health`.

**`emergency.py` (D):** on an ambulance in `gt.*`, read `traci.vehicle.getRoute`, compute ETA = distance / speed for each junction ahead. Request green on its approach when ETA < 15 s. Release when it passes, then 2 cycles of recovery green for the other directions.

**`pedestrian.py` (D):** if `ped_max_wait_s ≥ 60`, request the walk phase. If a person is still on the crossing at the end of walk, extend all-red by up to 4 s. Walkers under 0.8 m/s get +5 s.

**`graph.py` + `arbiter.py` (D)**
- LangGraph nodes: `sense → assess_health → route_mode → propose (agents) → arbitrate → act → log`. Runs every 5 s.
- Arbiter order: shield > emergency > pedestrian > fault mode > junction agent.
- `narrator.py` runs as its own async task, triggered only by `event` messages. Claude Haiku 5.5, max 30 words. Rate limit: 1 call per 3 s.

**Dashboard (D)**
- `useWebSocket` → a store keyed by stream.
- Panels: KPI strip, deck.gl map (junction markers coloured by mode, queue bars), camera thumbnails, Signo feed (SVG mascot + bubble, colour follows mood), controls, charts.
- Pip: an SVG car that moves along the map path to the junction of the latest event with a CSS transition.

---

## 9. Stretch goals (only after gate 4 passes)

1. Graph Attention layer for neighbour messages.
2. YOLO11n vs. YOLOv10n ablation (one-line swap).
3. Traffic-R1 checkpoint as an extra baseline.
4. Signo voice through the browser's `speechSynthesis`.
5. Fine-tuned ambulance class from 200 auto-labelled Unity frames.

## 10. Experiment run (System 2, 4:30–5:00)

`python experiments/run_grid.py --controllers fixed,batcs,maxpressure,ppo,signalmind --faults 0,0.1,0.2,0.3 --scenarios normal,surge --seeds 2`

- 5 × 4 × 2 × 2 = 80 headless 1-hour episodes on the train path. Run them in parallel with `multiprocessing` (cores − 1).
- If it runs long, drop `surge` first, then cut seeds to 1.
- `charts.py` produces: (1) wait + fuel bars per controller, (2) wait vs. failure %, with and without recovery, (3) a time series of one camera failure, (4) ambulance time and max pedestrian wait.

## 11. Demo script (3 minutes)

1. **0:00** Signo intro on the dashboard. The Unity view shows normal traffic.
2. **0:30** KPI strip vs. the BATCS-style baseline.
3. **1:00** Judge drags the fault slider to 30%. Unity shows rain. A J2 camera thumbnail freezes. Signo: *"J2 camera frozen. Using J1 counts and road sensor."* Traffic keeps moving.
4. **1:45** "Send ambulance". Lights turn green ahead of it in Unity and Pip races to the corridor.
5. **2:20** Charts: the degradation curve with and without recovery, plus fuel and CO₂ saved.
6. **2:45** Pitch line (main plan, §16).

## 12. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Organisers require CARLA | Swap Unity for the CARLA co-sim bridge. Interfaces in §4.3 are unchanged |
| Unity bridge unstable | `sumo-gui` screenshots → YOLO (gate 2) |
| YOLO too slow | Drop to 3 fps per camera, or `imgsz=480` |
| PPO doesn't converge | Max-Pressure + recovery + specialists still shows every feature |
| LAN or Redis problems | Phone hotspot. Have a static IP ready. Run everything on System 1 as a last resort |
| LLM API down | Template sentences for each event type |
| Live demo crashes | Backup video recorded at 4:30 |

## 13. Definition of done

- [ ] `start_system1.ps1` and `start_system2.ps1` bring everything up from a clean boot.
- [ ] The full loop runs 5 minutes with faults, an ambulance and a pedestrian surge.
- [ ] 4 charts exist in `experiments/results/`.
- [ ] README: architecture, how to run, results table, the honest note on the BATCS-style baseline.
- [ ] Backup demo video recorded.
