"""Warp @wp.func device helpers: quaternion math + robust-norm loss terms.
Inlined by the cost kernels (see kernels.py)."""
import warp as wp

# Added to a vector norm before dividing by it.
NORM_EPS = wp.constant(1.0e-8)


@wp.func
def _rotate_vec_wxyz(q0: float, q1: float, q2: float, q3: float,
                     v0: float, v1: float, v2: float) -> wp.vec3:
    """Rotate vector v by quaternion q (wxyz). Equivalent to quat_rotate_vec."""
    t0 = 2.0 * (q2 * v2 - q3 * v1)
    t1 = 2.0 * (q3 * v0 - q1 * v2)
    t2 = 2.0 * (q1 * v1 - q2 * v0)
    r0 = v0 + q0 * t0 + (q2 * t2 - q3 * t1)
    r1 = v1 + q0 * t1 + (q3 * t0 - q1 * t2)
    r2 = v2 + q0 * t2 + (q1 * t1 - q2 * t0)
    return wp.vec3(r0, r1, r2)


@wp.func
def _smooth_abs2(x: float, p: float, q: float) -> float:
    """Per-element pseudo-Huber: (|x|^q + p^q)^(1/q) - p."""
    return wp.pow(wp.pow(wp.abs(x), q) + wp.pow(p, q), 1.0 / q) - p


@wp.func
def _smooth_abs(x: float, p: float) -> float:
    """Per-element smooth L1: sqrt(x^2 + p^2) - p."""
    return wp.sqrt(x * x + p * p) - p


@wp.func
def _cosh_term(x: float, p: float) -> float:
    """Per-element cosh norm: p^2 * (cosh(x/p) - 1). Uses `0.5*(exp+exp)` since
    older warp versions don't expose `wp.cosh`."""
    y = x / p
    return p * p * (0.5 * (wp.exp(y) + wp.exp(-y)) - 1.0)


@wp.func
def _slerp_wxyz(q0w: float, q0x: float, q0y: float, q0z: float,
                q1w: float, q1x: float, q1y: float, q1z: float,
                t: float) -> wp.vec4:
    """Spherical linear interpolation between two wxyz quaternions.

    Picks the shortest path (sign-flips q1 if dot < 0) and falls back to
    nlerp when the two quaternions are nearly aligned to avoid sin(0)
    division. The returned quaternion is renormalized.
    """
    n0 = wp.sqrt(q0w * q0w + q0x * q0x + q0y * q0y + q0z * q0z) + NORM_EPS
    n1 = wp.sqrt(q1w * q1w + q1x * q1x + q1y * q1y + q1z * q1z) + NORM_EPS
    a0w = q0w / n0; a0x = q0x / n0; a0y = q0y / n0; a0z = q0z / n0
    a1w = q1w / n1; a1x = q1x / n1; a1y = q1y / n1; a1z = q1z / n1
    cos_theta = a0w * a1w + a0x * a1x + a0y * a1y + a0z * a1z
    if cos_theta < 0.0:
        a1w = -a1w; a1x = -a1x; a1y = -a1y; a1z = -a1z
        cos_theta = -cos_theta
    rw = float(0.0); rx = float(0.0); ry = float(0.0); rz = float(0.0)
    if cos_theta > 0.9995:
        rw = a0w + t * (a1w - a0w)
        rx = a0x + t * (a1x - a0x)
        ry = a0y + t * (a1y - a0y)
        rz = a0z + t * (a1z - a0z)
    else:
        cos_theta = wp.clamp(cos_theta, -1.0, 1.0)
        theta = wp.acos(cos_theta)
        sin_theta = wp.sin(theta)
        w0 = wp.sin((1.0 - t) * theta) / sin_theta
        w1 = wp.sin(t * theta) / sin_theta
        rw = w0 * a0w + w1 * a1w
        rx = w0 * a0x + w1 * a1x
        ry = w0 * a0y + w1 * a1y
        rz = w0 * a0z + w1 * a1z
    rn = wp.sqrt(rw * rw + rx * rx + ry * ry + rz * rz) + NORM_EPS
    return wp.vec4(rw / rn, rx / rn, ry / rn, rz / rn)


@wp.func
def _quat_geodesic_angle(cw: float, cx: float, cy: float, cz: float,
                         rw: float, rx: float, ry: float, rz: float) -> float:
    """Geodesic angle (radians) between two wxyz quaternions.

    Normalizes the current quaternion `c`, forms the error quaternion
    `e = c * conj(r)`, and returns `2·atan2(|e_xyz|, |e_w|)`. `r` is assumed
    unit-norm (it comes from `_slerp_wxyz`, which renormalizes). Arithmetic is
    identical to the inlined body/anchor-quat blocks it replaces.
    """
    cn = wp.sqrt(cw * cw + cx * cx + cy * cy + cz * cz) + NORM_EPS
    nw = cw / cn; nx = cx / cn; ny = cy / cn; nz = cz / cn
    ew = nw * rw + nx * rx + ny * ry + nz * rz
    ex = nx * rw - nw * rx + nz * ry - ny * rz
    ey = ny * rw - nz * rx - nw * ry + nx * rz
    ez = nz * rw + ny * rx - nx * ry - nw * rz
    v_mag = wp.sqrt(ex * ex + ey * ey + ez * ez)
    return 2.0 * wp.atan2(v_mag, wp.abs(ew))
