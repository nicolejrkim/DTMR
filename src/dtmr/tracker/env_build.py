"""Turn a resolved YAML config into an environment.

Every key is read generically: the environment config dataclass defines the accepted keys, their types and their
defaults, so adding a knob means adding one dataclass field and nothing here.
"""
from __future__ import annotations

import dataclasses

from dtmr.utils import ConfigError, check_keys, from_mapping
from dtmr.tracker.config import TrackerConfig
from dtmr.env import (
    TrackingEnvWarp,
    TrackingEnvConfig,
)
from dtmr.env.config import TOP_LEVEL_KEYS

# Top-level keys that configure the run rather than the controller or environment.
META_KEYS = frozenset({"cost_config", "robot", "base_config"})

# Keys older configs carry that nothing on the Warp path reads. Accepted with a warning.
DEPRECATED_TOP_LEVEL = {
    "output_dir": "the output location is the --output_dir flag",
    "env_name": "there is one tracking environment",
    "robot_class": "only humanoids are supported; there is one tracking environment",
    "control_mode": "position PD is the only control law; drop the key",
    "auto_joint_range_clamp_physical":
        "nothing reads it; the Warp environment uses "
        "`clip_joint_range_to_physical`",
}
_BM_TERMS = ("anchor_pos", "anchor_quat", "body_pos", "body_quat", "body_linvel", "body_angvel")
DEPRECATED_COST = {
    f"{prefix}_{term}": "BeyondMimic-style reward term, not implemented in the Warp environment"
    for prefix in ("w_bm", "std") for term in _BM_TERMS
}


def validate_config(cfg: dict, *, where: str = "config", allow_unknown: bool = False) -> list[str]:
    """Check every key of a resolved YAML against the config dataclasses.

    Raises ConfigError for keys nothing reads or keys in the wrong section, and
    returns warning lines for deprecated keys.
    """
    env_cfg_cls = TrackingEnvConfig
    controller = {f.name for f in dataclasses.fields(TrackerConfig)}
    env_all = {f.name for f in dataclasses.fields(env_cfg_cls)}
    env_top = env_all & TOP_LEVEL_KEYS
    env_cost = env_all - TOP_LEVEL_KEYS
    cost = cfg.get("cost_config") or {}
    if not isinstance(cost, dict):
        raise ConfigError(f"{where}: cost_config must be a mapping")
    warnings = check_keys(
        cfg.keys(), controller | env_top | META_KEYS, where=where,
        deprecated=DEPRECATED_TOP_LEVEL,
        elsewhere={k: "under cost_config" for k in env_cost},
        allow_unknown=allow_unknown,
    )
    warnings += check_keys(
        cost.keys(), env_cost, where=f"{where} cost_config",
        deprecated=DEPRECATED_COST,
        elsewhere={k: "at the top level" for k in controller | env_top},
        allow_unknown=allow_unknown,
    )
    return warnings


def build_env_config(cfg: dict):
    """The environment config dataclass for a resolved YAML (no environment built)."""
    cfg_cls = TrackingEnvConfig
    cost = cfg.get("cost_config") or {}
    values = {}
    for f in dataclasses.fields(cfg_cls):
        source = cfg if f.name in TOP_LEVEL_KEYS else cost
        if f.name in source:
            values[f.name] = source[f.name]
    return from_mapping(cfg_cls, values, where="environment config")


def build_env(cfg: dict) -> TrackingEnvWarp:
    env_cls = TrackingEnvWarp
    return env_cls(build_env_config(cfg))
