"""Checkpoint C0: run on BOTH systems. Passes when each lists both computer names."""
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bus import r  # noqa: E402

print("redis host:", os.getenv("REDIS_HOST"), "| ping:", r().ping())
me = socket.gethostname()
r().xadd("hello", {"from": me, "at": time.time()}, maxlen=50)
names = {msg["from"] for _id, msg in r().xrevrange("hello", count=20)}
print("computers seen on the bus:", ", ".join(sorted(names)))
print("PASS" if len(names) >= 2 else "Only this computer so far. Run this on the other system too.")
