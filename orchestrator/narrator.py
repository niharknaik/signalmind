"""Signo, the narrator: turns events into one short, friendly sentence.

Runs in its own thread and never blocks the control loop. Uses Claude when
credentials are available (model from NARRATOR_MODEL, default claude-opus-5-5
at low effort), and falls back to templates otherwise, so the demo never
depends on the network.
"""
from __future__ import annotations

import os
import queue
import threading
import time

import bus

MOOD = {"camera_down": "amber", "camera_up": "green", "mode_change": "amber", "emergency_start": "red",
        "emergency_end": "green", "ped_cap": "amber", "spillback": "amber", "shield_block": "amber", "info": "green",
        # System 1 bridge events
        "max_green_forced": "amber", "max_red_forced": "amber", "act_rejected_unknown_phase": "red",
        "ctl_reset_ignored": "amber"}

NAMES = {"J1": "J1", "J2": "J2", "J3": "J3", "J4": "J4"}


def template(e: dict) -> str:
    j, d, typ = e.get("junction") or "", e.get("detail", {}), e["type"]
    if typ == "camera_down":
        return f"{j} camera is unreliable. I'm estimating its queues from the road sensors and the junction upstream."
    if typ == "camera_up":
        return f"{j} camera is back. Switching that junction back to full AI control."
    if typ == "mode_change":
        to = d.get("to")
        if to == "fallback":
            return f"{j} has been blind for 30 seconds, so it's running the safe backup rule."
        if to == "degraded":
            return f"{j} camera looks shaky (health {d.get('health', 0):.0%}). The AI is using estimates there."
        return f"{j} is back on full AI control."
    if typ == "emergency_start":
        return f"Ambulance {d.get('id', '')} is {d.get('eta_s', '?')} s from {j}. Turning it green before it arrives."
    if typ == "emergency_end":
        return f"Ambulance cleared {j}. Now giving the waiting side extra green to drain the queue."
    if typ == "ped_cap":
        return f"Someone at {j} has waited {d.get('waited_s', 60):.0f} s to cross. Walk signal comes next."
    if typ == "max_green_forced":
        return f"{j} kept one green for 60 s, so System 1's safety guard switched it to give the other road a turn."
    if typ == "max_red_forced":
        return f"A road at {j} had waited 90 s, so System 1's safety guard gave it a green."
    if typ == "act_rejected_unknown_phase":
        return f"System 1 rejected a command for {j} because the phase wasn't 0 or 1. Please check the control loop."
    if typ == "ctl_reset_ignored":
        return "Restart isn't available from the dashboard while System 1 is running. Restart the demo on System 1."
    if typ == "spillback":
        return f"The road after {j} is nearly full, so {j} is holding traffic back to stop a jam spreading."
    return d.get("text", "All signals running normally.")


class Narrator:
    def __init__(self, use_llm: bool | None = None, min_gap_s: float = 3.0):
        self.q: queue.Queue = queue.Queue(maxsize=50)
        self.min_gap = min_gap_s
        self.model = os.getenv("NARRATOR_MODEL", "claude-opus-5-5")
        self.client = None
        if use_llm is not False:
            try:
                import anthropic
                self.client = anthropic.Anthropic()
                if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
                    self.client = None if use_llm is None else self.client
            except Exception:
                self.client = None
        threading.Thread(target=self._run, daemon=True).start()

    def push(self, event: dict):
        # "back to AI" is already announced by camera_up; deferred acts are routine bridge housekeeping
        if event["type"] == "mode_change" and event.get("detail", {}).get("to") == "ai":
            return
        if event["type"] == "act_deferred_by_guard":
            return
        try:
            self.q.put_nowait(event)
        except queue.Full:
            pass

    def _say_llm(self, base: str) -> str:
        import anthropic
        try:
            r = self.client.messages.create(
                model=self.model, max_tokens=300,
                output_config={"effort": "low"},
                system=("You are Signo, a friendly traffic light who narrates a smart traffic-signal "
                        "system for a city dashboard. Rewrite the update in one plain sentence, "
                        "at most 25 words, keeping every number and junction name."),
                messages=[{"role": "user", "content": base}],
            )
            if r.stop_reason == "refusal":
                return base
            text = next((b.text for b in r.content if b.type == "text"), "").strip()
            return text or base
        except (anthropic.APIStatusError, anthropic.APIConnectionError):
            return base

    def _run(self):
        last = 0.0
        while True:
            e = self.q.get()
            # merge a burst of events: keep the most important one
            batch = [e]
            while not self.q.empty() and len(batch) < 10:
                batch.append(self.q.get_nowait())
            order = ["emergency_start", "act_rejected_unknown_phase", "camera_down", "mode_change", "ped_cap",
                     "max_red_forced", "max_green_forced", "spillback", "emergency_end", "camera_up",
                     "shield_block", "ctl_reset_ignored", "info"]
            batch.sort(key=lambda x: order.index(x["type"]) if x["type"] in order else 99)
            e = batch[0]
            wait = self.min_gap - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            base = template(e)
            text = self._say_llm(base) if self.client else base
            bus.publish(bus.SAY, bus.Say(t=e["t"], text=text, mood=MOOD.get(e["type"], "green"),
                                         junction=e.get("junction")))
            last = time.time()
