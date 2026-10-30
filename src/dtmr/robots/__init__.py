"""Robot descriptors: one YAML per robot holding everything that is a fact about
the robot rather than a tuning choice (model file, PD gains, which bodies are
feet, knees and hands, named joint groups).

A run config selects one with `robot: <name>`; the descriptor is merged in as the
base layer and the config overrides whatever it sets itself.
"""
from __future__ import annotations

import os
from typing import Optional

import yaml

ROBOTS_DIR = os.path.dirname(__file__)


def robot_names() -> list[str]:
    return sorted(f[:-5] for f in os.listdir(ROBOTS_DIR) if f.endswith(".yaml"))


def load_robot(name: str) -> Optional[dict]:
    """The descriptor mapping for `name`, or None if there is no such robot."""
    path = os.path.join(ROBOTS_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}
