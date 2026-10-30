"""CLI + config loading + entry point for dtmr.

_deep_merge/_load_config resolve base_config: inheritance; build_dial_args maps
the config to TrackerConfig; _add_common_cli_args defines the flags; main()
orchestrates env build -> controller.run() -> post-run scoring and visualization."""
from __future__ import annotations

import argparse
import dataclasses
import os
import time

import numpy as np
import yaml

from dtmr.utils import ConfigError, from_mapping
from dtmr.robots import load_robot, robot_names

from .config import TrackerConfig
from .controller import Tracker
from .env_build import build_env, validate_config


def _deep_merge(base: dict, override: dict) -> dict:
    import copy as _copy
    result = _copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = _copy.deepcopy(value)
    return result


def _read_yaml(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("base_config", None)
    if base_ref is None:
        return cfg
    base_path = base_ref
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(config_path), base_path)
    return _deep_merge(_read_yaml(base_path), cfg)


def apply_robot(cfg: dict, where: str = "config") -> dict:
    """Merge the robot descriptor under `cfg` (the config wins where both set a key).

    The robot is named by `robot:`; configs without it fall back to their
    `model_dir` (older configs). Naming neither is an error.
    """
    cfg = dict(cfg)
    explicit = cfg.pop("robot", None)
    if not (explicit or cfg.get("model_dir")):
        raise ConfigError(
            f"{where}: no robot named - set `robot:` (available: {', '.join(robot_names())})"
        )
    name = str(explicit or cfg.get("model_dir"))
    descriptor = load_robot(name)
    if descriptor is None:
        if explicit is not None:
            raise ConfigError(
                f"{where}: unknown robot {name!r} (available: {', '.join(robot_names())})"
            )
        return cfg
    return _deep_merge(descriptor, cfg)


def _resolve_config(config_path: str) -> dict:
    """YAML with `base_config:` inheritance resolved and the robot descriptor merged in."""
    return apply_robot(_read_yaml(config_path), where=os.path.basename(config_path))


def _load_config(config_path: str, allow_unknown: bool = False) -> dict:
    """Load a run config: resolve inheritance and the robot descriptor, then check
    every key against the config dataclasses.

    A key nothing reads is an error (it would otherwise be silently ignored);
    pass allow_unknown=True to downgrade that to a warning.
    """
    cfg = _resolve_config(config_path)
    for line in validate_config(cfg, where=os.path.basename(config_path),
                                allow_unknown=allow_unknown):
        print(f"[config] {line}")
    return cfg


def build_dial_args(cfg: dict, cli_args) -> TrackerConfig:
    """Controller config from the YAML top level; --seed / --n_steps override it."""
    names = {f.name for f in dataclasses.fields(TrackerConfig)}
    values = {k: v for k, v in cfg.items() if k in names}
    if cli_args.seed is not None:
        values["seed"] = cli_args.seed
    if cli_args.n_steps is not None:
        values["n_steps"] = cli_args.n_steps
    return from_mapping(TrackerConfig, values, where="controller config")


def _pad_improvement_hist(rows) -> np.ndarray:
    """(n_steps, max_iters) float32, NaN-padded — rows are variable length when
    early termination fires (see TrackerConfig.improvement_threshold)."""
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    width = max(len(r) for r in rows)
    out = np.full((len(rows), max(width, 1)), np.nan, dtype=np.float32)
    for k, r in enumerate(rows):
        out[k, :len(r)] = r
    return out


def _add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--allow_unknown_keys", action="store_true",
                        help="Warn instead of failing when the config has keys nothing reads.")
    parser.add_argument("--motion_path", type=str, default=None,
                        help="Override motion_path from config: the reference's "
                             "<name>_50Hz.npz written by an importer.")
    parser.add_argument("--n_steps", type=int, default=None,
                        help="Override n_steps from config (an upper bound: the run stops "
                             "when the reference ends).")
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_graph", action="store_true",
                        help="Disable CUDA graph capture (runs eager).")
    parser.add_argument("--output_dir", type=str, default=None, required=True,
                        help="REQUIRED. Directory for all run outputs (log.npz, metrics, "
                             "ghost/visualization HTML, ...). "
                             "e.g. --output_dir ./output/g1_run")
    parser.add_argument("--no_serve", action="store_true",
                        help="Skip Flask server, just write the HTML file.")
    parser.add_argument("--no_viz", action="store_true",
                        help="Skip Brax HTML visualization entirely.")
    parser.add_argument("--viz_stride", type=int, default=1,
                        help="Keep every Nth frame in the HTML viewer (1 = all).")
    parser.add_argument("--no_dtw", action="store_true",
                        help="Skip post-run DTW scoring (default: compute + save YAML).")
    parser.add_argument("--no_phi_tracking", action="store_true",
                        help="Skip post-run phi-targeted MPJPE scoring (default: compute + save YAML).")
    parser.add_argument("--no_ghost", action="store_true",
                        help="Master switch: disable all three ghost overlays.")
    parser.add_argument("--no_ref_ghost", action="store_true",
                        help="Skip the plain (time-aligned) rollout_vs_ref overlay.")
    parser.add_argument("--no_phase_ghost", action="store_true",
                        help="Skip the phase-aligned rollout_vs_phase_ref overlay.")
    parser.add_argument("--no_dtw_ghost", action="store_true",
                        help="Skip the DTW-aligned rollout_vs_dtw_ref overlay.")
    parser.add_argument("--ghost_alpha", type=float, default=0.4,
                        help="Ghost transparency (0-1, lower = more transparent).")


def _check_motion_path(cfg: dict, config_path: str) -> None:
    """Fail with a pointer, not a numpy traceback, when the reference is missing."""
    path = str(cfg.get("motion_path", ""))
    if path and os.path.exists(path):
        return
    hint = (" (the shipped configs leave it as a placeholder)" if "REPLACE_ME" in path else "")
    raise SystemExit(
        f"[config] motion_path {path!r} from {os.path.basename(config_path)} does not exist{hint}.\n"
        f"  Build a reference with one of the importers (docs: 'Reference motions'),\n"
        f"  then pass --motion_path <out-dir>/<name>_50Hz.npz or set motion_path in the config.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="dtmr", description="DTMR: sampling-based MPC motion tracking with temporal motion retargeting (MuJoCo Warp)"
    )
    _add_common_cli_args(parser)
    cli_args = parser.parse_args()

    cfg = _load_config(cli_args.config, allow_unknown=cli_args.allow_unknown_keys)
    if cli_args.motion_path is not None:
        cfg["motion_path"] = cli_args.motion_path
    _check_motion_path(cfg, cli_args.config)
    env = build_env(cfg)

    dial_args = build_dial_args(cfg, cli_args)

    controller = Tracker(env, dial_args, use_graph=not cli_args.no_graph)

    t0 = time.time()
    log = controller.run(print_every=cli_args.print_every)
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.2f}s for {log['steps']} steps")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(cli_args.output_dir, exist_ok=True)
    npz_path = os.path.join(cli_args.output_dir, f"{timestamp}_log.npz")
    np.savez_compressed(
        npz_path,
        qpos=np.asarray(log["qpos"], dtype=np.float32),
        qvel=np.asarray(log["qvel"], dtype=np.float32),
        phases=np.asarray(log["phases"], dtype=np.float32),
        rewards=np.asarray(log["rewards"], dtype=np.float32),
        joint_targets=np.asarray(log["joint_targets"], dtype=np.float32),
        diffuse_iters=np.asarray(log.get("diffuse_iters", []), dtype=np.int32),
        # per-step per-iteration improvement (best sample - incumbent), NaN-padded
        # to the longest step; empty rows when improvement_threshold=0 (check off)
        improvement_hist=_pad_improvement_hist(log.get("improvement_hist", [])),
        elapsed=np.asarray([elapsed], dtype=np.float32),
        steps=np.asarray([log["steps"]], dtype=np.int32),
        # sigma_cert probe/active telemetry: (call, rung_exp, ess,
        # imp_headroom, p_beat, incumbent_reward) per diffusion iteration
        **({"cert_ess": np.asarray(log["cert_ess"], dtype=np.float32)}
           if "cert_ess" in log else {}),
    )
    print(f"[save] wrote {npz_path}")

    want_dtw_ghost = not (cli_args.no_viz or cli_args.no_ghost or cli_args.no_dtw_ghost)
    dtw_metrics = None
    if not cli_args.no_dtw or want_dtw_ghost:
        try:
            from dtmr.post.dtw import compute_run_dtw
            dtw_metrics = compute_run_dtw(
                env,
                np.asarray(log["qpos"], dtype=np.float32),
                np.asarray(log["rewards"], dtype=np.float32),
                cli_args.output_dir,
                timestamp,
                save_path=want_dtw_ghost,
            )
        except Exception as e:
            print(f"[dtw] scoring failed: {e}")

    if not cli_args.no_phi_tracking:
        try:
            from dtmr.post.phi_tracking import compute_run_phi_tracking
            compute_run_phi_tracking(
                env,
                np.asarray(log["qpos"], dtype=np.float32),
                np.asarray(log["phases"], dtype=np.float32),
                cli_args.output_dir,
                timestamp,
            )
        except Exception as e:
            print(f"[phi_tracking] scoring failed: {e}")

    if cli_args.no_viz:
        return

    dtw_path = dtw_metrics.get("dtw_path") if isinstance(dtw_metrics, dict) else None
    from dtmr.post.viz import render_run, serve_html
    webpage = render_run(
        env,
        log["qpos"], log["qvel"], log["phases"],
        cli_args.output_dir, timestamp,
        stride=cli_args.viz_stride,
        visualize_ref_plain=not (cli_args.no_ghost or cli_args.no_ref_ghost),
        visualize_ref_phase=not (cli_args.no_ghost or cli_args.no_phase_ghost),
        visualize_ref_dtw=not (cli_args.no_ghost or cli_args.no_dtw_ghost),
        dtw_path=dtw_path,
        ghost_alpha=cli_args.ghost_alpha,
    )
    if webpage is None or cli_args.no_serve:
        return
    serve_html(webpage)


if __name__ == "__main__":
    main()
