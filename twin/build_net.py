"""Build the 4-junction corridor network and demand files.

    python twin/build_net.py

Main road runs west to east (2 lanes each way, 50 km/h) through J1..J4, 250 m
apart. Each junction has a north and south side street (1 lane each way).
Sidewalks and pedestrian crossings are generated at every junction.
Swap in a real OSM corridor later by replacing corridor.net.xml.
"""
import os
import subprocess
import sys

import sumo

HERE = os.path.dirname(os.path.abspath(__file__))
NET = os.path.join(HERE, "net")
BIN = os.path.join(sumo.SUMO_HOME, "bin")
SPACING, SIDE = 250, 180
J = ["J1", "J2", "J3", "J4"]


def write(name, text):
    with open(os.path.join(NET, name), "w", encoding="utf8") as f:
        f.write(text)


def nodes():
    xs = {j: SPACING * (i + 1) for i, j in enumerate(J)}
    rows = ['<node id="W0" x="0" y="0" type="priority"/>',
            f'<node id="E0" x="{SPACING * 5}" y="0" type="priority"/>']
    for i, j in enumerate(J, 1):
        x = xs[j]
        rows.append(f'<node id="{j}" x="{x}" y="0" type="traffic_light" tl="{j}"/>')
        rows.append(f'<node id="N{i}" x="{x}" y="{SIDE}" type="priority"/>')
        rows.append(f'<node id="S{i}" x="{x}" y="{-SIDE}" type="priority"/>')
    write("corridor.nod.xml", "<nodes>\n  " + "\n  ".join(rows) + "\n</nodes>\n")


def edges():
    main = ["W0"] + J + ["E0"]
    rows = []
    for a, b in zip(main, main[1:]):
        for u, v in ((a, b), (b, a)):
            rows.append(f'<edge id="{u}_{v}" from="{u}" to="{v}" numLanes="2" speed="13.89" sidewalkWidth="2.5"/>')
    for i, j in enumerate(J, 1):
        for s in (f"N{i}", f"S{i}"):
            for u, v in ((s, j), (j, s)):
                rows.append(f'<edge id="{u}_{v}" from="{u}" to="{v}" numLanes="1" speed="11.11" sidewalkWidth="2.0"/>')
    write("corridor.edg.xml", "<edges>\n  " + "\n  ".join(rows) + "\n</edges>\n")


def routes():
    vtypes = """
  <vType id="car" vClass="passenger" emissionClass="HBEFA4/PC_petrol_Euro-4" length="4.5" sigma="0.5"/>
  <vType id="bike2w" vClass="motorcycle" emissionClass="HBEFA4/MC_4S_le250cc_Euro-4" length="2.2" maxSpeed="16" sigma="0.6" latAlignment="arbitrary" minGapLat="0.3"/>
  <vType id="auto3w" vClass="passenger" emissionClass="HBEFA4/PC_petrol_Euro-3" length="3.0" maxSpeed="12" sigma="0.6" guiShape="passenger/hatchback"/>
  <vType id="bus" vClass="bus" emissionClass="HBEFA3/Bus" length="12" personCapacity="60"/>
  <vTypeDistribution id="mix">
    <vType id="car_m" vClass="passenger" emissionClass="HBEFA4/PC_petrol_Euro-4" length="4.5" sigma="0.5" probability="0.40"/>
    <vType id="bike_m" vClass="motorcycle" emissionClass="HBEFA4/MC_4S_le250cc_Euro-4" length="2.2" maxSpeed="16" sigma="0.6" probability="0.40"/>
    <vType id="auto_m" vClass="passenger" emissionClass="HBEFA4/PC_petrol_Euro-3" length="3.0" maxSpeed="12" sigma="0.6" probability="0.15"/>
    <vType id="bus_m" vClass="bus" emissionClass="HBEFA3/Bus" length="12" personCapacity="60" probability="0.05"/>
  </vTypeDistribution>"""
    flows = []
    # Main road, both directions (veh/h). Morning peak on the inbound (E->W) side.
    flows.append('<flow id="main_we" type="mix" begin="0" end="3600" from="W0_J1" to="J4_E0" vehsPerHour="1000" departLane="best" departSpeed="max"/>')
    flows.append('<flow id="main_ew" type="mix" begin="0" end="3600" from="E0_J4" to="J1_W0" vehsPerHour="1200" departLane="best" departSpeed="max"/>')
    for i, j in enumerate(J, 1):
        nxt = J[i] if i < 4 else "E0"
        prv = J[i - 2] if i > 1 else "W0"
        # straight across, and turning onto the main road
        flows.append(f'<flow id="ns{i}" type="mix" begin="0" end="3600" from="N{i}_{j}" to="{j}_S{i}" vehsPerHour="180"/>')
        flows.append(f'<flow id="sn{i}" type="mix" begin="0" end="3600" from="S{i}_{j}" to="{j}_N{i}" vehsPerHour="180"/>')
        flows.append(f'<flow id="nt{i}" type="mix" begin="0" end="3600" from="N{i}_{j}" to="{j}_{nxt}" vehsPerHour="90"/>')
        flows.append(f'<flow id="st{i}" type="mix" begin="0" end="3600" from="S{i}_{j}" to="{j}_{prv}" vehsPerHour="90"/>')
    peds = []
    for i, j in enumerate(J, 1):
        # people crossing the main road at each junction, both ways
        peds.append(f'<personFlow id="pn{i}" begin="0" end="3600" period="exp(0.025)"><walk from="N{i}_{j}" to="{j}_S{i}"/></personFlow>')
        peds.append(f'<personFlow id="ps{i}" begin="0" end="3600" period="exp(0.025)"><walk from="S{i}_{j}" to="{j}_N{i}"/></personFlow>')
    write("corridor.rou.xml", "<routes>" + vtypes + "\n  " + "\n  ".join(flows + peds) + "\n</routes>\n")
    amb = [f'<vehicle id="amb_{k:02d}" type="ambulance" depart="{t}" departLane="best" departSpeed="max"><route edges="W0_J1 J1_J2 J2_J3 J3_J4 J4_E0"/></vehicle>'
           for k, t in enumerate(range(300, 3600, 1200), 1)]
    write("ambulance.rou.xml", '<routes>\n  <vType id="ambulance" vClass="emergency" guiShape="emergency" emissionClass="HBEFA3/LDV_D_EU4" length="6" speedFactor="1.3" color="1,0,0"/>\n  '
          + "\n  ".join(amb) + "\n</routes>\n")


def detectors():
    rows = []
    for i, j in enumerate(J, 1):
        prv = J[i - 2] if i > 1 else "W0"
        nxt = J[i] if i < 4 else "E0"
        for edge in (f"{prv}_{j}", f"{nxt}_{j}", f"N{i}_{j}", f"S{i}_{j}"):
            lanes = 2 if not edge.startswith(("N", "S")) else 1
            for ln in range(1, lanes + 1):  # lane 0 is the sidewalk
                rows.append(f'<inductionLoop id="loop_{edge}_{ln}" lane="{edge}_{ln}" pos="-30" period="5" file="NUL"/>')
    write("detectors.add.xml", "<additional>\n  " + "\n  ".join(rows) + "\n</additional>\n")


def cfg():
    write("corridor.sumocfg", """<configuration>
  <input>
    <net-file value="corridor.net.xml"/>
    <route-files value="corridor.rou.xml"/>
    <additional-files value="detectors.add.xml"/>
  </input>
  <time><begin value="0"/><end value="3600"/><step-length value="1"/></time>
  <processing><time-to-teleport value="300"/><lateral-resolution value="0.8"/></processing>
  <report><no-step-log value="true"/><no-warnings value="true"/></report>
</configuration>
""")


if __name__ == "__main__":
    os.makedirs(NET, exist_ok=True)
    nodes(); edges(); routes(); detectors(); cfg()
    cmd = [os.path.join(BIN, "netconvert"), "-n", "corridor.nod.xml", "-e", "corridor.edg.xml",
           "-o", "corridor.net.xml", "--sidewalks.guess.from-permissions", "false",
           "--crossings.guess", "true", "--walkingareas", "true",
           "--tls.default-type", "static", "--tls.cycle.time", "90",
           "--no-turnarounds", "true", "--junctions.corner-detail", "5", "--no-warnings", "true"]
    p = subprocess.run(cmd, cwd=NET, capture_output=True, text=True)
    print(p.stdout[-800:], p.stderr[-1500:])
    sys.exit(p.returncode)
