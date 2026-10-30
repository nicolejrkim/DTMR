"""Shared helpers: packaged resource paths, strict config loading from YAML mappings,
and the reference-file loaders.

For config loading the dataclass is the single source of truth: its fields are the
accepted keys, its annotations give the types, and its defaults are the only defaults.
"""
from __future__ import annotations

import dataclasses
import difflib
import importlib.resources
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


# ---------------------------------------------------------------- packaged resources

def get_model_path(robot_name, model_name):
    with importlib.resources.path(f"dtmr.models.{robot_name}", model_name) as path:
        return path


def get_example_path(example_name):
    with importlib.resources.path("dtmr.examples", example_name) as path:
        return path


# ---------------------------------------------------------------- config loading

class ConfigError(ValueError):
    """A config names a key nothing reads, or gives a value of the wrong shape."""


def _annotation(field: dataclasses.Field) -> str:
    return str(field.type).replace("typing.", "").replace(" ", "")


def _is_optional(field: dataclasses.Field) -> bool:
    return _annotation(field).startswith("Optional[")


def _coerce(name: str, value: Any, annotation: str) -> Any:
    ann = annotation
    if ann.startswith("Optional["):
        ann = ann[len("Optional["):-1]
    if value is None:
        return None
    try:
        if ann == "bool":
            return bool(value)
        if ann == "int":
            return int(value)
        if ann == "float":
            return float(value)
        if ann == "str":
            return str(value)
        if ann.startswith("Sequence[float]"):
            if hasattr(value, "tolist"):
                value = value.tolist()
            return tuple(float(x) for x in value)
        if ann == "tuple" or ann.startswith(("Sequence[", "Tuple[", "tuple[")):
            return tuple(value or ())
        if ann == "dict" or ann.startswith(("Dict[", "dict[")):
            return dict(value or {})
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{name}: cannot read {value!r} as {annotation}") from e
    return value


def from_mapping(cls, values: Mapping[str, Any], where: str = "config"):
    """Instantiate dataclass `cls` from `values`, coercing each value to its field type.

    Keys must be fields of `cls` (run `check_keys` first for friendly errors). A
    null value means "use the default" unless the field is Optional.
    """
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in values.items():
        if key not in fields:
            raise ConfigError(f"{where}: {key!r} is not a field of {cls.__name__}")
        field = fields[key]
        if value is None and not _is_optional(field):
            continue
        kwargs[key] = _coerce(key, value, _annotation(field))
    missing = [
        name for name, f in fields.items()
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        and name not in kwargs
    ]
    if missing:
        raise ConfigError(f"{where}: required key(s) missing: {', '.join(missing)}")
    return cls(**kwargs)


def check_keys(
    present: Iterable[str],
    allowed: Iterable[str],
    *,
    where: str,
    deprecated: Mapping[str, str] | None = None,
    elsewhere: Mapping[str, str] | None = None,
    allow_unknown: bool = False,
) -> list[str]:
    """Reject keys nothing reads. Returns warning lines for deprecated/tolerated keys.

    deprecated: key -> why it is ignored (accepted with a warning, never an error).
    elsewhere:  key -> the section it belongs in (always an error: the value would be lost).
    """
    allowed = set(allowed)
    deprecated = deprecated or {}
    elsewhere = elsewhere or {}
    warnings, problems = [], []
    for key in present:
        if key in allowed:
            continue
        if key in deprecated:
            warnings.append(f"{where}: ignoring {key!r} ({deprecated[key]})")
        elif key in elsewhere:
            problems.append(f"{key!r} belongs {elsewhere[key]}")
        else:
            close = difflib.get_close_matches(key, allowed, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            if allow_unknown:
                warnings.append(f"{where}: ignoring unknown key {key!r}{hint}")
            else:
                problems.append(f"unknown key {key!r}{hint}")
    if problems:
        raise ConfigError(
            f"{where}: " + "; ".join(problems)
            + ". Nothing reads these keys, so they would be silently ignored; fix them or "
              "pass --allow_unknown_keys."
        )
    return warnings


# ---------------------------------------------------------------- reference files

def load_link_poses_npz(npz_path):
    """Load the link world poses of a reference npz.

    Keys: pos (T, B, 3), rot or quat (T, B, 4) wxyz, link_names (B,), link_idx,
    timestamps or time_idx.

    Returns:
      pos, quat, link_names, name_to_idx, time_idx  (NumPy)
    """
    data = np.load(npz_path, allow_pickle=True)
    pos = data["pos"]
    quat = data["quat"] if "quat" in data.files else data["rot"]
    link_names = data["link_names"]
    time_idx = data["time_idx"] if "time_idx" in data.files else data["timestamps"]
    name_to_idx = {name: i for i, name in enumerate(link_names.tolist())}
    return pos, quat, link_names, name_to_idx, time_idx


def load_reference_state(path, nq=None, nv=None):
    """The state rows of a reference: {"qpos": (T, nq), "qvel": (T, nv)}.

    A reference npz carries them as `qpos` / `qvel`. For the older two-file layout
    (`<name>_link_world_poses.npz` with a `<name>.csv` of nq + nv columns next to it, or
    that CSV itself) the rows are split at `nq`; without `nq` the whole rows are returned
    as `qpos` (qpos comes first, so callers that only slice leading columns still work).
    Returns None when the npz has no state and no sibling CSV exists.
    """
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "qpos" in data.files:
            out = {"qpos": np.asarray(data["qpos"], dtype=np.float32)}
            if "qvel" in data.files:
                out["qvel"] = np.asarray(data["qvel"], dtype=np.float32)
            return out
        stem = path.stem
        candidates = []
        if stem.endswith("_link_world_poses"):
            candidates.append(path.with_name(stem[: -len("_link_world_poses")] + ".csv"))
        candidates.append(path.with_suffix(".csv"))
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            return None
    rows = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    if nq is None:
        return {"qpos": rows}
    if nv is not None and rows.shape[1] != nq + nv:
        raise ValueError(f"State CSV {path} width mismatch: expected {nq + nv}, got {rows.shape[1]}")
    return {"qpos": rows[:, :nq], "qvel": rows[:, nq:]}
