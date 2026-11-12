# SignalMind: Project Plan

**CodeSpark 2026 · Domain 3 · PS1 · 6-hour build · 2 systems**

A team of AI agents that runs the traffic signals of a Bengaluru corridor inside a 3D digital twin. They coordinate with each other, clear the way for ambulances, protect pedestrians, track fuel and CO₂, and keep working when their cameras fail.

| | |
|---|---|
| Corridor | 4 junctions (OpenStreetMap, Bengaluru) |
| 3D twin | Unity (rendering) + SUMO (traffic engine) |
| Systems | System 1: high GPU (Unity, YOLO). System 2: low GPU (RL, agents, dashboard) |
| Agents | 9 types, 4 decision speeds |
| Camera fault rates tested | 0 / 10 / 20 / 30 % |
| Bar to beat | Bengaluru's BATCS |

The step-by-step build is in [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md).

> ⚠️ The problem statement names CARLA for the 3D co-simulation. Confirm with the organisers that Unity is acceptable. If not, swap in the CARLA co-sim bridge. The rest of the design is unchanged.

---

## 1. The bar to beat: Bengaluru's AI signals today

Bengaluru Traffic Police run **BATCS** (Bengaluru Adaptive Traffic Control System) on about 165–169 junctions. It uses junction cameras and C-DAC's **CoSiCoSt** algorithm, controlled from one central room.

**What works**
- Travel time at Hudson Circle fell 33%, and on the four-signal KR Road corridor it fell 20% (traffic police figures).
- Camera-based vehicle counts adjust timings in real time.
- Corridor green waves and emergency-vehicle priority are listed as features.
- The commissioner reports over 95% automation, with under 5% manual overrides.

**Reported problems**
- Peak hours: police sources said the AI signals "failed" to ease peak traffic. An IISc expert says such systems help only in unsaturated traffic.
- Safety: in September 2025 some signals skipped amber and went straight from green to red, causing near-misses.
- No published behaviour for when a camera fails.
- Most results come from the police. No independent city-wide before/after study exists.

### BATCS vs. SignalMind

| Problem | BATCS today | What SignalMind does | How we prove it |
|---|---|---|---|
| Peak-hour saturation | Adjusts each junction from local counts | Neighbour-aware agents, spillback guard, Max-Pressure fallback built for heavy load | Peak surge ×1.3: spillbacks and throughput |
| Amber skipping | Green to red without amber reported | Hard safety shield: amber 3 s, all-red 2 s, min green 10 s, max red 90 s | Target: 0 transitions without amber |
| Camera failure | Not documented | Sentinel health score → Recovery (Kalman + upstream + road sensors) → AI / Degraded / Fallback per junction | Wait vs. failure %, with and without recovery |
| Emergency vehicles | Listed as a feature, with no public detail | ETA-based green corridor + recovery phase | Ambulance time, and the delay to other traffic |
| Pedestrians | Not mentioned | Camera-counted demand, 60 s max walk wait, extra time for slow walkers | Max and average pedestrian wait |
| Pollution and fuel | Not measured | HBEFA fuel, CO₂, NOx, PM per junction. Stop-and-go penalised | Fuel L/h, CO₂ kg/h |
| Testing | Measured on live roads | Digital twin under faults, surges and emergencies | Full experiment grid |
| Explaining decisions | Operators watch from a control room | Signo explains every mode change | Live dashboard feed |

> **Comparing fairly:** CoSiCoSt isn't public. We build a **BATCS-style baseline** in SUMO: actuated signals plus green-wave offsets. We claim to beat this baseline in simulation, not BATCS on real roads.

---

## 2. What the research says

All results below are the authors' own claims.

| Paper | What already works | What we take or add |
|---|---|---|
| Azfar & Ke, arXiv 2412.03925 | 3D pole cameras → YOLO → MARL in SUMO | Closed-loop pipeline. **Add** faults + recovery |
| VLMLight, NeurIPS 2025 | Fast RL + slow LLM reasoning. Up to 65% less ambulance waiting | Fast/slow split, but the critical path is deterministic rules. The LLM only explains |
| Traffic-R1, arXiv 2508.02344 | 3B LLM controller, async messages between junctions | Async bus messages. *Stretch:* checkpoint as a baseline |
| MalLight, arXiv 2408.09768 | One failed junction costs ~30% throughput. Neighbour coordination limits it | Neighbours tighten spillback limits when a junction degrades |
| Wiering et al. 2019; Berkeley ITS | Training with failures gives robustness. Twin DQN beats Max-Pressure under sensor faults | Fault curriculum 0→30%, extended to 4 agents |
| EMVLight; GNN-MAPPO (T-ITS 2026) | Coordinated corridors beat plain preemption | ETA-planned corridor. Measure the delay to other traffic |
| Pedestrian-aware RL studies | Vehicle-only rewards ignore pedestrians | Reward per person + hard 60 s cap |
| YOLO11 / v12 / RT-DETR comparisons | YOLO is faster and generalises better | YOLOv10n (required by the PS). *Stretch:* YOLO11n ablation |

---

## 3. Social impact

| Impact | Metric |
|---|---|
| Emergency green corridor | Ambulance travel time ↓ |
| Pedestrian-first safety | Max walk wait ≤ 60 s |
| Cleaner air | Fuel L/h, CO₂ kg/h ↓ |
| Shorter commutes | Avg wait s/veh ↓ |
| Works when cameras break | Wait at 20% camera failure |
| Safe by design | 0 transitions without amber |

---

## 4. Architecture: six layers across two systems

```
                      SYSTEM 1 · HIGH GPU                 SYSTEM 2 · LOW GPU
L6 Command center     Unity 3D view (screen 1)            React + deck.gl dashboard, Signo + Pip (screen 2)
L5 Orchestration                                          LangGraph supervisor, priority arbiter, narrator
L4 Multi-agent                                            4 PPO junction agents, Coordinator, Emergency,
                                                          Pedestrian, Eco, Safety shield, Max-Pressure
L3 Fault & recovery   Fault injector (frames), Sentinel   Recovery (Kalman + upstream + loop fusion)
L2 Perception         YOLOv10n + ByteTrack, lane zones,
                      queue lines, people, ambulance
L1 Digital twin       Unity scene ⇄ live SUMO (bridge)    Headless SUMO/libsumo for training + experiments
                      ─────────────── Redis Streams on System 2 (LAN) ───────────────
```

- **Train path (System 2, fast):** `SUMO → TraCI counts → fault injector → recovery → PPO`
- **Demo path (both systems):** `SUMO⇄Unity → camera → faults → YOLO → Sentinel (S1) → Recovery → agents → arbiter (S2) → act → SUMO (S1)`

---

## 5. The agent team

| Tier | Speed | Agents |
|---|---|---|
| Reflex | < 50 ms | Safety shield, Emergency preemption, Pedestrian clearance, Perception, Sentinel |
| Tactical | 5 s | Junction agents ×4 (PPO), Eco, Recovery |
| Strategic | 60 s | Coordinator (offsets, spillback limits) |
| Narrative | async | Signo narrator (Claude Haiku 5.5) |

**Priority arbiter:** Safety shield > Emergency > Pedestrian > Fault mode > Junction agent.

## 6. Cooperation and reward

- Shared eyes (neighbour queues and health in each observation), shared reward (+0.5 × neighbours), spillback guard (downstream > 80% full → upstream holds).

```python
r = - (vehicle_wait + 1.5 * pedestrian_wait) / people
    - 0.3 * pressure
    - 0.1 * stops
    + 0.5 * mean(r_neighbours)
```

## 7. Fault detection and recovery

| Fault | Detection | | Mode | Condition |
|---|---|---|---|---|
| Blackout | entropy ≈ 0 | | AI | health ≥ 0.7 |
| Frozen | frame diff = 0, > 3 frames | | Degraded-AI | 0.3–0.7 |
| Noise / rain | mean conf < 0.35 | | Fallback (Max-Pressure) | < 0.3 for > 30 s |
| Occlusion | camera vs. loop > 50% apart | | | Back to AI after 3 healthy cycles |

## 8. Emergency, pedestrians, pollution

- **Green corridor:** detect → route + ETA → green 15 s ahead → hold → recover → explain.
- **Pedestrian-first:** 60 s cap, all-red extension while someone is crossing, +5 s for walkers under 0.8 m/s.
- **Pollution and fuel:** HBEFA per vehicle → junction heatmap → Eco reward term → "CO₂ saved vs. BATCS-style" on the dashboard.

## 9. Dashboard and characters

- KPI strip, live map with mode chips and queue bars, Unity camera thumbnails, Signo's feed, controls (fault slider, send ambulance, pedestrian surge, demand ×1.3, weather), agents panel, charts.
- **Signo:** traffic-light narrator whose colour follows the mode. **Pip:** a tiny car that drives to whichever junction has an event.

## 10. Stack

| Area | System 1 (high GPU) | System 2 (low GPU) |
|---|---|---|
| Sim | Unity 2022.3 LTS (URP) + live SUMO 1.20 + TCP bridge | Headless SUMO + libsumo |
| AI | Ultralytics YOLOv10n, supervision ByteTrack | Ray RLlib PPO, sumo-rl, filterpy |
| Orchestration | | LangGraph, Claude Haiku 5.5 |
| App | | Redis 7, FastAPI + WebSocket, React (Vite) + deck.gl + Recharts |
| Python | 3.10 | 3.10 |

## 11. Evaluation

| Item | Values |
|---|---|
| Controllers | Fixed-time · BATCS-style · Max-Pressure · PPO · SignalMind |
| Faults | 0, 10, 20, 30 % |
| Scenarios | Normal · surge ×1.3 (plus ambulances and pedestrian surge in the demo) |
| Seeds | 2 |
| Metrics | Avg wait, queue m, spillbacks, throughput, fuel, CO₂, ambulance time, max walk wait, transitions without amber |

## 12. Pitch (Signo's voice)

> Bengaluru's AI signals are smart one junction at a time, but they struggle at peak hour and once skipped the yellow light. SignalMind makes every signal part of a team. It clears the road for ambulances, never makes a pedestrian wait more than a minute, cuts fuel burnt at red lights, and keeps working when a camera dies. Drag the slider and watch.

---

## Sources

1. [Smart Mobility: a case study of BATCS](https://indianinfrastructure.com/2026/04/06/smart-mobility-a-case-study-of-the-bengaluru-adaptive-traffic-control-system/)
2. [AI-powered signals reduce travel time by 33%](https://www.deccanherald.com/india/karnataka/bengaluru/ai-powered-signals-in-bengaluru-reduce-travel-time-by-33-3226394)
3. [City's AI traffic signals cut out amber](https://www.deccanherald.com/amp/story/india%2Fkarnataka%2Fbengaluru%2Fcity-s-ai-traffic-signals-cut-out-amber-push-riders-into-red-alert-3718264)
4. [Why experts feel AI signals are no silver bullet](https://www.thenewsminute.com/article/why-experts-feel-ai-traffic-signals-are-no-silver-bullet-bengaluru-s-traffic-154641)
5. [How Bengaluru rewired its traffic signals](https://www.deccanherald.com/amp/story/opinion/how-bengaluru-rewired-its-traffic-signals-to-beat-gridlock-3832080)
6. [Bengaluru Police unveil AI adaptive traffic control](https://www.theweek.in/wire-updates/national/2024/10/09/mes5-ka-traffic-system.html)
7. [Traffic co-simulation with infrastructure cameras and RL](https://arxiv.org/html/2412.03925v2)
8. [VLMLight](https://arxiv.org/html/2505.19486v2)
9. [Traffic-R1](https://arxiv.org/abs/2508.02344v2)
10. [MalLight](https://arxiv.org/html/2408.09768v3)
11. [Robust deep RL for traffic signal control](https://arxiv.org/pdf/1904.08353)
12. [Digital twin RL for fault-resilient signal control](https://its.berkeley.edu/node/6749)
13. [Hierarchical GNN-MAPPO for emergency vehicles](https://arxiv.org/pdf/2601.04177v1)
14. [EMVLight](https://arxiv.org/pdf/2109.05429)
15. [Pedestrian-friendly RL signal control](https://discovery.ucl.ac.uk/10215752/1/Manuscript.pdf)
16. [YOLO11 vs RT-DETR on driving data](https://isprs-archives.copernicus.org/articles/XLIX-B2-2026/1191/2026/)
