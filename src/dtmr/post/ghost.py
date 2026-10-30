"""
Same-scene ghost overlays for Brax HTML: the rollout and the reference drawn
together in one html.render() call, the reference as a semi-transparent copy.

`render_overlay_comparison` loads `<scene>_dual.xml` next to the tracking scene
when one is shipped, and otherwise builds the two-robot model at runtime with
`create_dual_robot_xml`.
"""
import copy
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import jax
import numpy as np
from jax import numpy as jnp
from brax.io import html


def create_dual_robot_xml(
    xml_path: str,
    ghost_prefix: str = "ghost_",
    ghost_rgba: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 0.4),
    ghost_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """
    Create a new XML with two robots: original + ghost copy.
    
    Uses MuJoCo to resolve includes, then manipulates the full model.
    
    Args:
        xml_path: Path to original robot XML
        ghost_prefix: Prefix for ghost robot elements
        ghost_rgba: RGBA color for ghost robot (default: semi-transparent red)
        ghost_offset: Initial position offset for ghost robot
        
    Returns:
        Path to new XML file with dual robots
    """
    import mujoco

    # Use MuJoCo to load and save the fully resolved XML
    # This handles <include> directives properly
    # Convert to string in case it's a Path object
    xml_path_str = str(xml_path)
    model = mujoco.MjModel.from_xml_path(xml_path_str)

    # Save to a temp file to get the resolved XML
    fd, resolved_xml_path = tempfile.mkstemp(suffix=".xml", prefix="resolved_")
    os.close(fd)
    mujoco.mj_saveLastXML(resolved_xml_path, model)

    # Now parse the resolved XML
    tree = ET.parse(resolved_xml_path)
    root = tree.getroot()
    os.remove(resolved_xml_path)

    # The dual XML is written to a temp dir, so asset paths must not stay relative
    # to the original model's folder.
    model_dir = os.path.dirname(os.path.abspath(xml_path_str))
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    for attr in ("meshdir", "texturedir"):
        sub = compiler.get(attr) or compiler.get("assetdir") or ""
        compiler.set(attr, os.path.normpath(os.path.join(model_dir, sub)))
    if "assetdir" in compiler.attrib:
        del compiler.attrib["assetdir"]

    # Find worldbody
    worldbody = root.find(".//worldbody")
    if worldbody is None:
        raise ValueError("No worldbody found in XML")

    # Find the main robot body - look for a body with freejoint (floating base robot)
    # or the first body that has joints and isn't a static object
    robot_body = None

    def find_robot_body(parent):
        """Find the robot root body (has freejoint or is the main articulated body)."""
        for body in parent.findall("body"):
            name = body.get("name", "").lower()
            # Skip floor/ground/light
            if "floor" in name or "ground" in name or "plane" in name or "light" in name:
                continue

            # Check if this body has a freejoint (floating base robot)
            if body.find("freejoint") is not None:
                return body

            # Check if this body has joints (articulated)
            if body.find("joint") is not None:
                return body

            # Check children recursively
            child_robot = find_robot_body(body)
            if child_robot is not None:
                return body  # Return the parent as robot root

        return None

    robot_body = find_robot_body(worldbody)

    if robot_body is None:
        # Fallback: just take the first body that isn't floor/ground/light
        for body in worldbody.findall("body"):
            name = body.get("name", "").lower()
            if not any(skip in name for skip in ["floor", "ground", "plane", "light"]):
                robot_body = body
                break

    if robot_body is None:
        raise ValueError("Could not identify robot body in XML")

    # Deep copy the robot body
    ghost_body = copy.deepcopy(robot_body)

    # Rename all elements in ghost body with prefix
    _rename_elements_recursive(ghost_body, ghost_prefix)

    # Change all geom colors to ghost color
    _set_geom_rgba_recursive(ghost_body, ghost_rgba)

    # Apply position offset to ghost body
    if ghost_offset != (0.0, 0.0, 0.0):
        current_pos = ghost_body.get("pos", "0 0 0")
        pos_vals = [float(x) for x in current_pos.split()]
        if len(pos_vals) < 3:
            pos_vals = [0.0, 0.0, 0.0]
        new_pos = f"{pos_vals[0] + ghost_offset[0]} {pos_vals[1] + ghost_offset[1]} {pos_vals[2] + ghost_offset[2]}"
        ghost_body.set("pos", new_pos)

    # Add ghost body to worldbody
    worldbody.append(ghost_body)

    # Also duplicate actuators if present
    actuator = root.find(".//actuator")
    if actuator is not None:
        ghost_actuators = []
        for act in list(actuator):
            ghost_act = copy.deepcopy(act)
            # Rename actuator
            if ghost_act.get("name"):
                ghost_act.set("name", ghost_prefix + ghost_act.get("name"))
            # Update joint reference
            if ghost_act.get("joint"):
                ghost_act.set("joint", ghost_prefix + ghost_act.get("joint"))
            if ghost_act.get("tendon"):
                ghost_act.set("tendon", ghost_prefix + ghost_act.get("tendon"))
            ghost_actuators.append(ghost_act)

        for ghost_act in ghost_actuators:
            actuator.append(ghost_act)

    # Save to temp file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".xml", prefix="dual_robot_")
    os.close(temp_fd)
    tree.write(temp_path)

    return temp_path


def _rename_elements_recursive(element: ET.Element, prefix: str):
    """Recursively rename all named elements with prefix."""
    # Rename this element
    if element.get("name"):
        element.set("name", prefix + element.get("name"))

    # Update references
    for attr in ["joint", "body1", "body2", "geom1", "geom2", "site", "tendon", "target"]:
        if element.get(attr):
            element.set(attr, prefix + element.get(attr))

    # Recurse to children
    for child in element:
        _rename_elements_recursive(child, prefix)


def _set_geom_rgba_recursive(element: ET.Element, rgba: Tuple[float, float, float, float]):
    """Recursively set rgba on all geoms."""
    if element.tag == "geom":
        element.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")

    for child in element:
        _set_geom_rgba_recursive(child, rgba)


def align_trajectory_to_length(
    trajectory: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Resample trajectory to match target length using nearest neighbor.
    
    Args:
        trajectory: Shape (T, ...)
        target_length: Desired length
        
    Returns:
        Resampled trajectory, shape (target_length, ...)
    """
    src_len = trajectory.shape[0]
    if src_len == target_length:
        return trajectory

    indices = np.linspace(0, src_len - 1, target_length).astype(int)
    return trajectory[indices]


def align_trajectory_by_time(
    trajectory: np.ndarray,
    src_fps: float,
    target_length: int,
    target_dt: float,
) -> np.ndarray:
    """
    Align trajectory by real time, preserving original fps.
    
    Maps each target frame to the corresponding source frame based on real time.
    If the source is shorter than target duration, the last frame is held.
    
    Args:
        trajectory: Shape (T_src, ...) source trajectory
        src_fps: Frame rate of source trajectory (Hz)
        target_length: Number of frames in target
        target_dt: Time step of target (seconds)
        
    Returns:
        Aligned trajectory, shape (target_length, ...)
    """
    src_len = trajectory.shape[0]

    aligned = np.zeros((target_length,) + trajectory.shape[1:], dtype=trajectory.dtype)

    for t in range(target_length):
        # Real time at this target frame
        real_time = t * target_dt
        # Corresponding source frame
        src_frame = int(real_time * src_fps)
        # Clamp to valid range (hold last frame if out of bounds)
        src_frame = min(src_frame, src_len - 1)
        aligned[t] = trajectory[src_frame]

    return aligned


def align_with_dtw_path(
    trajectory: np.ndarray,
    dtw_path: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Align trajectory using DTW warping path.
    
    Args:
        trajectory: Shape (T_src, ...)
        dtw_path: Shape (path_len, 2), each row is (target_idx, src_idx)
        target_length: Length of target trajectory
        
    Returns:
        Aligned trajectory, shape (target_length, ...)
    """
    aligned = np.zeros((target_length,) + trajectory.shape[1:], dtype=trajectory.dtype)

    # Build mapping from target index to source index
    target_to_src = {}
    for target_idx, src_idx in dtw_path:
        target_to_src[int(target_idx)] = int(src_idx)

    last_src_idx = 0
    for t in range(target_length):
        if t in target_to_src:
            src_idx = target_to_src[t]
            last_src_idx = src_idx
        else:
            src_idx = last_src_idx

        src_idx = max(0, min(src_idx, trajectory.shape[0] - 1))
        aligned[t] = trajectory[src_idx]

    return aligned


def align_with_frame_indices(
    trajectory: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """
    Align trajectory using frame indices from the phase variable.
    
    For each rollout step t, frame_indices[t] gives the reference frame index
    that was being tracked at that step.
    
    Args:
        trajectory: Shape (T_ref, ...) reference trajectory
        frame_indices: Shape (T_rollout,) array of reference frame indices
        
    Returns:
        Aligned trajectory, shape (T_rollout, ...)
    """
    target_length = len(frame_indices)
    aligned = np.zeros((target_length,) + trajectory.shape[1:], dtype=trajectory.dtype)

    for t in range(target_length):
        src_idx = int(frame_indices[t])
        # Clamp to valid range
        src_idx = max(0, min(src_idx, trajectory.shape[0] - 1))
        aligned[t] = trajectory[src_idx]

    return aligned


def render_overlay_comparison(
    env,
    rollout: List,
    ref_qpos: np.ndarray,
    dtw_path: Optional[np.ndarray] = None,
    ref_qvel: Optional[np.ndarray] = None,
    ref_fps: Optional[float] = None,
    frame_indices: Optional[np.ndarray] = None,
    ghost_alpha: float = 0.4,
    height: int = 1080,
    loop: bool = True,
    title: str = "Motion Comparison",
    model_path: Optional[str] = None,
) -> str:
    """
    TRUE SAME-SCENE RENDERING: Both robots in ONE html.render() call.
    
    Builds (or loads) a two-robot model with the ghost robot in a semi-transparent
    colour, then renders both trajectories together in a single scene.
    
    Args:
        env: object exposing `sys` (the brax.mjcf system) and `dt`
        rollout: List of pipeline states from actual rollout  
        ref_qpos: Reference qpos trajectory
        dtw_path: Optional DTW path for temporal alignment (takes precedence over all other alignment methods)
        ref_qvel: Optional reference qvel
        ref_fps: Optional reference fps for time-based alignment (preserves original timing).
                 If None, reference is resampled to match rollout length.
                 Ignored if dtw_path or frame_indices is provided.
        frame_indices: Optional array of frame indices from phase variable (takes precedence over ref_fps).
                      Each entry gives the reference frame being tracked at that rollout step.
        ghost_alpha: Transparency for ghost robot (0-1) - only used when the dual model is built at runtime
        height: Render height
        loop: Whether to loop
        title: HTML title
        model_path: The tracking scene the rollout was run on. A `<scene>_dual.xml`
                    beside it is loaded when present (faster); otherwise the
                    two-robot model is built from it at runtime.
        
    Returns:
        HTML string from single html.render() with both robots
    """
    from brax.io import mjcf
    from brax.mjx import pipeline as mjx_pipeline

    rollout_len = len(rollout)

    # Extract qpos/qvel from rollout
    rollout_qpos = []
    rollout_qvel = []
    for state in rollout:
        if hasattr(state, 'qpos'):
            rollout_qpos.append(np.array(state.qpos))
            qvel = np.array(state.qvel) if hasattr(state, 'qvel') else np.zeros(env.sys.nv)
            rollout_qvel.append(qvel)
        elif hasattr(state, 'q'):
            rollout_qpos.append(np.array(state.q))
            qvel = np.array(state.qd) if hasattr(state, 'qd') else np.zeros(env.sys.nv)
            rollout_qvel.append(qvel)

    rollout_qpos = np.array(rollout_qpos)
    rollout_qvel = np.array(rollout_qvel)

    # Align reference to rollout length
    if dtw_path is not None:
        # DTW alignment takes precedence
        aligned_ref_qpos = align_with_dtw_path(ref_qpos, dtw_path, rollout_len)
        aligned_ref_qvel = align_with_dtw_path(ref_qvel, dtw_path, rollout_len) if ref_qvel is not None else None
    elif frame_indices is not None:
        # Phase-based alignment: use frame indices from TMR phase variable
        aligned_ref_qpos = align_with_frame_indices(ref_qpos, frame_indices)
        aligned_ref_qvel = align_with_frame_indices(ref_qvel, frame_indices) if ref_qvel is not None else None
    elif ref_fps is not None:
        # Time-based alignment: reference plays at original fps
        aligned_ref_qpos = align_trajectory_by_time(ref_qpos, ref_fps, rollout_len, env.dt)
        aligned_ref_qvel = align_trajectory_by_time(ref_qvel, ref_fps, rollout_len, env.dt) if ref_qvel is not None else None
    else:
        # Default: resample to match rollout length
        aligned_ref_qpos = align_trajectory_to_length(ref_qpos, rollout_len)
        aligned_ref_qvel = align_trajectory_to_length(ref_qvel, rollout_len) if ref_qvel is not None else None

    if aligned_ref_qvel is None:
        aligned_ref_qvel = np.zeros_like(rollout_qvel)

    # Pre-build concatenated qpos/qvel arrays for vectorized init
    all_qpos = jnp.array([
        np.concatenate([rollout_qpos[t], aligned_ref_qpos[t]])
        for t in range(rollout_len)
    ])
    all_qvel = jnp.array([
        np.concatenate([rollout_qvel[t], aligned_ref_qvel[t]])
        for t in range(rollout_len)
    ])

    # Load dual-robot system
    if model_path is None:
        raise ValueError("render_overlay_comparison needs model_path (the tracking scene)")
    model_path = str(model_path)
    pregenerated = os.path.splitext(model_path)[0] + "_dual.xml"
    if os.path.exists(pregenerated):
        dual_sys = mjcf.load(pregenerated)
    else:
        dual_xml_path = create_dual_robot_xml(
            model_path,
            ghost_prefix="ghost_",
            ghost_rgba=(0.8, 0.2, 0.2, ghost_alpha),
        )
        try:
            dual_sys = mjcf.load(dual_xml_path)
        finally:
            os.remove(dual_xml_path)
    if dual_sys.nq != 2 * rollout_qpos.shape[1]:
        raise ValueError(f"dual model has nq={dual_sys.nq}, expected 2 x {rollout_qpos.shape[1]}")

    # Build dual rollout using vectorized init (much faster than loop)
    # Use vmap to initialize all states in parallel
    def init_single_state(qpos, qvel):
        return mjx_pipeline.init(dual_sys, qpos, qvel)

    batched_init = jax.vmap(init_single_state)
    dual_rollout_batched = batched_init(all_qpos, all_qvel)

    # Convert batched pytree to list of states for html.render()
    dual_rollout = [
        jax.tree.map(lambda x: x[t], dual_rollout_batched)
        for t in range(rollout_len)
    ]

    # Single html.render() call with both robots in same scene
    dual_sys_render = dual_sys.tree_replace({"opt.timestep": env.dt})
    webpage = html.render(dual_sys_render, dual_rollout, height, loop)

    return webpage
