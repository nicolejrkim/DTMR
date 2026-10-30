"""Lowest collision surface of a robot pose, for the env's spawn grounding.

A retargeted reference whose lowest collision surface sits below z=0 makes the
physics engine eject the robot on the first step, and most retargeted clips
start a few millimetres inside the floor. The tracking env therefore lifts its
reset pose (`reset_ground`) until it rests on the floor; the reference itself
is never modified.

Measure from COLLISION GEOMETRY, never from link origins. A link origin sits
above its own contact surface (G1's toe_link rests at +17.1 mm with the foot
flat), and that offset is meaningless for knee / hand / torso contacts.
"""
from __future__ import annotations

def geom_lowpoints(model):
    """Per collidable geom: (kind, id, local points, radius) for lowest-z queries.

    The lowest point of a convex body under any rotation is a hull vertex, so
    meshes reduce to their convex hull; primitives use analytic extents.
    """
    import mujoco
    import numpy as np

    # A geom collides if contype/conaffinity enable broadphase OR it appears in
    # an explicit contact pair. The tracking sim models take the second route
    # (contype=0 on every robot geom, 57 floor<->*_collision pairs), so a
    # contype-only filter finds nothing and grounding silently no-ops.
    collidable = {g for g in range(model.ngeom)
                  if model.geom_contype[g] or model.geom_conaffinity[g]}
    for i in range(int(model.npair)):
        collidable.add(int(model.pair_geom1[i]))
        collidable.add(int(model.pair_geom2[i]))

    out = []
    for g in range(model.ngeom):
        if g not in collidable:
            continue
        t = int(model.geom_type[g])
        sz = model.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            m = model.geom_dataid[g]
            a, n = model.mesh_vertadr[m], model.mesh_vertnum[m]
            v = model.mesh_vert[a:a + n].reshape(-1, 3).astype(float)
            if len(v) > 4:
                try:
                    from scipy.spatial import ConvexHull
                    v = v[np.unique(ConvexHull(v).vertices)]
                except Exception:
                    pass
            out.append(("verts", g, v, 0.0))
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            out.append(("verts", g, np.zeros((1, 3)), float(sz[0])))
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            h = float(sz[1])
            out.append(("verts", g, np.array([[0, 0, h], [0, 0, -h]], float), float(sz[0])))
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            h = float(sz[1])
            out.append(("cyl", g, np.array([[0, 0, h], [0, 0, -h]], float), float(sz[0])))
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            c = np.array([[x, y, z] for x in (-sz[0], sz[0])
                          for y in (-sz[1], sz[1]) for z in (-sz[2], sz[2])])
            out.append(("verts", g, c, 0.0))
    return out


def _lowest_z(data, specs):
    import numpy as np

    lo = np.inf
    for kind, g, loc, r in specs:
        p = data.geom_xpos[g]
        M = data.geom_xmat[g].reshape(3, 3)
        wz = (loc @ M.T)[:, 2] + p[2]
        if kind == "cyl":
            nz = M[2, 2]
            lo = min(lo, float(wz.min() - r * np.sqrt(max(0.0, 1.0 - nz * nz))))
        else:
            lo = min(lo, float(wz.min() - r))
    return lo
