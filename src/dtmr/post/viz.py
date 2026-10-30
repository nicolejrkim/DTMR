"""Post-run visualization for Warp-backend rollouts.

  - Plain Brax HTML replay of the rollout qpos trajectory.
  - Three ghost overlays (rollout + reference in one scene) on a two-robot copy
    of the tracking scene (`<scene>_dual.xml` when shipped, else built at runtime):
      * `_rollout_vs_ref.html`       — ref played at its native fps (no alignment)
      * `_rollout_vs_phase_ref.html` — phase-aligned (frame_idx = floor(phi * T_ref))
      * `_rollout_vs_dtw_ref.html`   — DTW-aligned (uses (rollout_idx, ref_idx) path)

Called by the `dtmr` entry point at the end of a run.
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np


def _load_brax_sys(mjcf_path: str, dt: float):
    from brax.io import mjcf as brax_mjcf
    sys = brax_mjcf.load(str(mjcf_path))
    sys = sys.tree_replace({"opt.timestep": float(dt)})
    return sys


def _build_brax_rollout(sys, qpos_arr: np.ndarray, qvel_arr: np.ndarray,
                        stride: int = 1):
    """Vectorized pipeline.init across frames. Falls back to per-frame."""
    import jax
    import jax.numpy as jnp
    try:
        from brax.mjx import pipeline as brax_pipeline
    except Exception:
        from brax.generalized import pipeline as brax_pipeline

    if stride > 1:
        qpos_arr = qpos_arr[::stride]
        qvel_arr = qvel_arr[::stride]
    n_frames = len(qpos_arr)

    batched_init = jax.jit(jax.vmap(lambda q, v: brax_pipeline.init(sys, q, v)))
    try:
        stacked = batched_init(jnp.asarray(qpos_arr), jnp.asarray(qvel_arr))
        jax.block_until_ready(stacked)
        rollout = [jax.tree_util.tree_map(lambda x, i=i: x[i], stacked)
                   for i in range(n_frames)]
    except Exception as e:
        print(f"[viz] batched pipeline.init failed ({e}); per-frame fallback.")
        rollout = [brax_pipeline.init(sys, jnp.asarray(q), jnp.asarray(v))
                   for q, v in zip(qpos_arr, qvel_arr)]
    return rollout, n_frames


def _render_ghost(
    *, sys, rollout, ref_qpos_arr, env_dt, ghost_alpha, model_path, title,
    dtw_path=None, frame_indices=None, ref_fps=None,
):
    """Thin wrapper around `render_overlay_comparison` with an env-like stand-in
    (only `sys` and `dt` are read)."""
    from dtmr.post.ghost import render_overlay_comparison
    env_like = SimpleNamespace(sys=sys, dt=float(env_dt))
    return render_overlay_comparison(
        env=env_like,
        rollout=rollout,
        ref_qpos=ref_qpos_arr,
        dtw_path=dtw_path,
        frame_indices=frame_indices,
        ref_fps=ref_fps,
        ghost_alpha=float(ghost_alpha),
        height=1080,
        loop=True,
        title=title,
        model_path=model_path,
    )


def render_run(
    env,
    qpos_log,
    qvel_log,
    phases_log,
    output_dir: str,
    timestamp: str,
    *,
    stride: int = 1,
    visualize_ref_plain: bool = True,
    visualize_ref_phase: bool = True,
    visualize_ref_dtw: bool = True,
    dtw_path: Optional[np.ndarray] = None,
    ghost_alpha: float = 0.4,
) -> Optional[str]:
    """Emit plain Brax HTML and up to three ghost HTMLs.

    Ghost variants (each gated by its own bool):
      - plain: reference at its native fps, no alignment (`_rollout_vs_ref.html`)
      - phase: frame_idx = floor(phi * T_ref)           (`_rollout_vs_phase_ref.html`)
      - dtw:   follow DTW (rollout_idx, ref_idx) path   (`_rollout_vs_dtw_ref.html`)

    `env` must expose `mj_model`, `config.model_dir`, `config.model_file`,
    `config.motion_fps`, `dt`; `_state_ref["qpos"]` is used as the ghost
    trajectory (skipped with a warning if absent).

    Returns the plain-HTML page string (or None on failure) so the caller can
    pass it to `serve_html`.
    """
    os.makedirs(output_dir, exist_ok=True)

    qpos_arr = np.asarray(qpos_log, dtype=np.float32)
    qvel_arr = np.asarray(qvel_log, dtype=np.float32)
    phi_arr = np.asarray(phases_log, dtype=np.float32) if phases_log is not None else None

    try:
        from dtmr.utils import get_model_path
        model_path = get_model_path(env.config.model_dir, env.config.model_file)
        sys = _load_brax_sys(model_path, env.dt)
    except Exception as e:
        print(f"[viz] brax.mjcf.load failed ({e}); skipping HTML viz.")
        return None

    t0 = time.time()
    rollout, n_frames = _build_brax_rollout(sys, qpos_arr, qvel_arr, stride=stride)
    print(f"[viz] built {n_frames}-frame rollout in {time.time()-t0:.2f}s")

    from brax.io import html as brax_html
    webpage = None
    plain_html = os.path.join(output_dir, f"{timestamp}_brax_visualization.html")
    try:
        t0 = time.time()
        webpage = brax_html.render(sys, rollout, 1080, True)
        with open(plain_html, "w") as f:
            f.write(webpage)
        print(f"[viz] wrote {plain_html} ({time.time()-t0:.2f}s)")
    except Exception as e:
        err_path = os.path.join(output_dir, f"{timestamp}_render_error.txt")
        with open(err_path, "w") as f:
            f.write(repr(e))
        print(f"[viz] plain html.render failed ({e}); wrote {err_path}")

    state_ref = getattr(env, "_state_ref", None) or {}
    ref_qpos = state_ref.get("qpos") if isinstance(state_ref, dict) else None
    want_any_ghost = visualize_ref_plain or visualize_ref_phase or visualize_ref_dtw
    if want_any_ghost and ref_qpos is None:
        print("[viz] skipping all ghost overlays (env._state_ref['qpos'] missing)")
        return webpage
    if not want_any_ghost:
        return webpage

    ref_qpos_arr = np.asarray(ref_qpos, dtype=np.float32)
    T_ref = int(ref_qpos_arr.shape[0])
    motion_fps = float(getattr(env.config, "motion_fps", 0.0) or 0.0)

    # --- Plain (no alignment, ref at its native fps) ---
    if visualize_ref_plain:
        if motion_fps <= 0:
            print("[viz] skipping plain ghost (motion_fps unknown)")
        else:
            try:
                t0 = time.time()
                page = _render_ghost(
                    sys=sys, rollout=rollout, ref_qpos_arr=ref_qpos_arr,
                    env_dt=env.dt, ghost_alpha=ghost_alpha, model_path=model_path,
                    title="Warp rollout vs Reference (time-aligned @ motion_fps)",
                    ref_fps=motion_fps,
                )
                out = os.path.join(output_dir, f"{timestamp}_rollout_vs_ref.html")
                with open(out, "w") as f:
                    f.write(page)
                print(f"[viz] wrote {out} ({time.time()-t0:.2f}s)")
            except Exception as e:
                print(f"[viz] plain ghost overlay failed: {e}")
                import traceback; traceback.print_exc()

    # --- Phase-aligned (frame_idx = floor(phi * T_ref)) ---
    if visualize_ref_phase:
        if phi_arr is None:
            print("[viz] skipping phase ghost (phases log missing)")
        else:
            try:
                frame_indices = np.clip(
                    np.floor(phi_arr * T_ref).astype(np.int32), 0, T_ref - 1,
                )
                if stride > 1:
                    frame_indices = frame_indices[::stride]
                t0 = time.time()
                page = _render_ghost(
                    sys=sys, rollout=rollout, ref_qpos_arr=ref_qpos_arr,
                    env_dt=env.dt, ghost_alpha=ghost_alpha, model_path=model_path,
                    title="Warp rollout vs Phase-Aligned Reference",
                    frame_indices=frame_indices,
                )
                out = os.path.join(output_dir,
                                   f"{timestamp}_rollout_vs_phase_ref.html")
                with open(out, "w") as f:
                    f.write(page)
                print(f"[viz] wrote {out} ({time.time()-t0:.2f}s)")
            except Exception as e:
                print(f"[viz] phase ghost overlay failed: {e}")
                import traceback; traceback.print_exc()

    # --- DTW-aligned (path is (rollout_idx, ref_idx) pairs) ---
    if visualize_ref_dtw:
        if dtw_path is None:
            print("[viz] skipping dtw ghost (dtw_path not provided)")
        else:
            try:
                path = np.asarray(dtw_path, dtype=np.int32)
                if stride > 1:
                    path = path.copy()
                    path[:, 0] = path[:, 0] // stride
                t0 = time.time()
                page = _render_ghost(
                    sys=sys, rollout=rollout, ref_qpos_arr=ref_qpos_arr,
                    env_dt=env.dt, ghost_alpha=ghost_alpha, model_path=model_path,
                    title="Warp rollout vs DTW-Aligned Reference",
                    dtw_path=path,
                )
                out = os.path.join(output_dir,
                                   f"{timestamp}_rollout_vs_dtw_ref.html")
                with open(out, "w") as f:
                    f.write(page)
                print(f"[viz] wrote {out} ({time.time()-t0:.2f}s)")
            except Exception as e:
                print(f"[viz] dtw ghost overlay failed: {e}")
                import traceback; traceback.print_exc()

    return webpage


def serve_html(webpage: str, port: int = 5000) -> None:
    """Serve one HTML page on http://127.0.0.1:<port> until interrupted."""
    import flask
    app = flask.Flask(__name__)

    @app.route("/")
    def index():
        return webpage

    print(f"[viz] serving on http://127.0.0.1:{port}")
    app.run(port=port)
