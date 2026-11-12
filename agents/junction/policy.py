"""Load a trained PPO checkpoint and act greedily. ~1 ms per decision on CPU."""
from __future__ import annotations

import os

import numpy as np
import torch

from agents.junction.train import ActorCritic, CKPT


def load(path: str | None = None):
    path = path or os.path.join(CKPT, "ppo_B.pt")
    if not os.path.exists(path):
        path = os.path.join(CKPT, "ppo_latest.pt")
    net = ActorCritic()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()

    def act(vec: np.ndarray) -> int:
        with torch.no_grad():
            return int(net.pi(torch.as_tensor(vec).unsqueeze(0)).argmax(-1)[0])
    act.path = path
    return act
