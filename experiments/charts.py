"""Charts for the judges, from experiments/results/grid.csv.

    python experiments/charts.py

1_before_after.png     avg wait + fuel per controller (normal demand, no faults)
2_fault_curve.png      avg wait vs camera failure %, with vs without recovery  (hero chart)
3_social_impact.png    ambulance travel time + max pedestrian wait per controller
4_peak_surge.png       avg wait + spillbacks at 1.3x demand
summary.md             the same numbers as a table, for the README / slides
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
NAMES = {"fixed": "Fixed-time", "batcs": "BATCS-style", "maxpressure": "Max-Pressure",
         "ppo": "PPO only", "signalmind": "SignalMind"}
ORDER = ["fixed", "batcs", "maxpressure", "ppo", "signalmind"]
GREY, ACCENT, AMBER, RED = "#8a948d", "#1f9d61", "#d99a1e", "#c9483b"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def load(name="grid.csv"):
    rows = list(csv.DictReader(open(os.path.join(RES, name))))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["controller"], r["scenario"], float(r["fault_rate"]))].append(r)
    return agg


def mean(rs, k):
    v = [float(r[k]) for r in rs if r.get(k) not in ("", "None", None)]
    return sum(v) / len(v) if v else float("nan")


def get(agg, ctl, scen, fault, k):
    # fixed / batcs don't use our cameras, so their 0 % result holds at every fault rate
    key = (ctl, scen, 0.0 if ctl in ("fixed", "batcs") else fault)
    return mean(agg.get(key, []), k)


def bars(ax, labels, vals, title, unit, highlight="SignalMind"):
    cols = [ACCENT if l == highlight else GREY for l in labels]
    b = ax.bar(labels, vals, color=cols)
    ax.bar_label(b, fmt="%.0f", padding=3)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(unit)
    ax.tick_params(axis="x", rotation=15)


def main():
    agg = load()
    ctls = [c for c in ORDER if any(k[0] == c for k in agg)]
    labels = [NAMES[c] for c in ctls]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    bars(ax[0], labels, [get(agg, c, "normal", 0.0, "avg_wait_s") for c in ctls], "Average wait", "seconds per vehicle")
    bars(ax[1], labels, [get(agg, c, "normal", 0.0, "fuel_l_h") for c in ctls], "Fuel burnt", "litres per hour")
    fig.suptitle("Before / after: 4-junction corridor, normal demand, healthy cameras", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "1_before_after.png"), dpi=160); plt.close(fig)

    faults = sorted({k[2] for k in agg})
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for c, col, ls in (("batcs", GREY, ":"), ("maxpressure", AMBER, "--"), ("ppo", RED, "-"), ("signalmind", ACCENT, "-")):
        if c in ctls:
            ys = [get(agg, c, "normal", f, "avg_wait_s") for f in faults]
            ax.plot([f * 100 for f in faults], ys, ls, color=col, marker="o", lw=2.5 if c == "signalmind" else 1.8,
                    label=NAMES[c] + (" (no recovery)" if c in ("ppo", "maxpressure") else ""))
    ax.axvline(20, color=GREY, lw=1, alpha=0.6)
    ax.text(20.5, ax.get_ylim()[1] * 0.95, "PS target: 20 %", color=GREY, fontsize=9, va="top")
    ax.set_xlabel("camera failure rate (%)"); ax.set_ylabel("average wait (s / vehicle)")
    ax.set_title("Fault tolerance: wait vs camera failure", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "2_fault_curve.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    bars(ax[0], labels, [get(agg, c, "normal", 0.0, "ambulance_time_s") for c in ctls], "Ambulance trip, 1.25 km", "seconds")
    bars(ax[1], labels, [get(agg, c, "normal", 0.0, "ped_max_wait_s") for c in ctls], "Longest pedestrian wait", "seconds")
    ax[1].axhline(60, color=RED, lw=1, ls="--"); ax[1].text(-0.4, 61.5, "60 s cap", color=RED, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "3_social_impact.png"), dpi=160); plt.close(fig)

    if any(k[1] == "surge" for k in agg):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        bars(ax[0], labels, [get(agg, c, "surge", 0.0, "avg_wait_s") for c in ctls], "Peak hour (1.3x demand): average wait", "seconds per vehicle")
        bars(ax[1], labels, [get(agg, c, "surge", 0.0, "spillbacks") for c in ctls], "Queue spillbacks", "count per hour")
        fig.tight_layout(); fig.savefig(os.path.join(RES, "4_peak_surge.png"), dpi=160); plt.close(fig)

    lines = ["| Controller | Scenario | Camera faults | Avg wait s | Fuel L/h | CO2 kg/h | Ambulance s | Max walk s | Spillbacks | Amber violations | Runs |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (c, s, f), rs in sorted(agg.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1], kv[0][2])):
        lines.append(f"| {NAMES[c]} | {s} | {f:.0%} | {mean(rs, 'avg_wait_s'):.1f} | {mean(rs, 'fuel_l_h'):.0f} | "
                     f"{mean(rs, 'co2_kg_h'):.0f} | {mean(rs, 'ambulance_time_s'):.0f} | {mean(rs, 'ped_max_wait_s'):.0f} | "
                     f"{mean(rs, 'spillbacks'):.0f} | {mean(rs, 'amber_violations'):.0f} | {len(rs)} |")
    open(os.path.join(RES, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("charts written to", RES)


if __name__ == "__main__":
    main()
