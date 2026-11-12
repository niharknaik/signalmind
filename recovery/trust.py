"""Camera trust check: don't believe a camera just because it says it's healthy.

Each cycle we compare the camera's total queue count with the road sensors' count for the
same junction. A smoothed disagreement score decides how far we trust the camera:

    mismatch  = |camera_total - sensor_total| / max(sensor_total, FLOOR)
    score     = EMA of mismatch (alpha 0.3, so one odd frame doesn't trigger it)
    distrust when score > ON, trust again when score < OFF  (hysteresis: no flicker)
    while distrusted: health used = min(camera health, max(0.35, 1 - score)), fault = "mismatch"

FLOOR stops tiny queues from looking like a total failure: with 2 cars waiting and a
camera saying 0, the mismatch is 2/3 = 0.67, not 100%.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ALPHA = 0.3
FLOOR = 3.0
ON, OFF = 0.45, 0.25
# A miscounting camera lands in "estimating" (health 0.3-0.7), never in "backup" (< 0.3):
# the road sensors still work, so the AI keeps deciding on sensor-based estimates. A camera
# that is really blind still reports its own low health and goes to backup as before.
MIN_TRUST = 0.35


@dataclass
class CameraTrust:
    score: dict = field(default_factory=dict)
    distrusted: dict = field(default_factory=dict)

    def check(self, j: str, t: float, camera: list[float], sensors: list[float], health: float, fault: str):
        """Returns (health_to_use, fault_to_use, event_or_None)."""
        cam, sen = float(sum(camera)), float(sum(sensors))
        mismatch = abs(cam - sen) / max(sen, FLOOR)
        s = ALPHA * mismatch + (1 - ALPHA) * self.score.get(j, 0.0)
        self.score[j] = s
        was = self.distrusted.get(j, False)
        now = s > ON if not was else s > OFF
        self.distrusted[j] = now
        event = None
        if now and not was:
            event = {"type": "camera_distrusted", "junction": j, "t": t,
                     "detail": {"camera": round(cam), "sensors": round(sen), "score": round(s, 2)}}
        elif was and not now:
            event = {"type": "camera_trusted", "junction": j, "t": t, "detail": {"score": round(s, 2)}}
        if now:
            capped = max(MIN_TRUST, 1.0 - s)
            if capped < health:
                return capped, "mismatch", event
        return health, fault, event
