"""MPPI @wp.kernel set for the warp controller (perturb, spline node->u, copy,
phase extract/broadcast, softmax weights, weighted Ybar update). Used by both the
graph and eager rollout paths. These call no @wp.func — only wp builtins."""
import warp as wp

# Added to the reward variance before the square root (softmax normalisation).
REWARD_VAR_EPS = wp.constant(1.0e-6)


@wp.kernel
def _perturb_kernel(
    Ybar: wp.array2d(dtype=float),
    sigma: wp.array2d(dtype=float),
    Y0s_out: wp.array3d(dtype=float),
    seed_buf: wp.array(dtype=wp.int32),
    hnode_p1: int,
    n_ctrl_dofs: int,
    nsample: int,
    rho: float,
):
    """Generate noise on GPU, add to Ybar, clip; thread per sample.

    sigma has shape (hnode_p1, n_ctrl_dofs) — allows per-dim scale (e.g.
    sigma_scale_phi for the trailing dphi dimension).
    `rho` > 0 draws AR(1)-correlated noise across nodes (per dof, unit
    marginal variance) in a separate k-outer branch; rho <= 0 draws
    independent noise per node.
    Index `nsample` (the last world) is filled with Ybar itself so the
    controller can read `rew_ybar` from the rollout reward buffer directly.
    """
    wid = wp.tid()
    state = wp.rand_init(seed_buf[0], wid)

    if rho <= 0.0:
        for j in range(hnode_p1):
            for k in range(n_ctrl_dofs):
                eps = wp.randn(state)
                val = Ybar[j, k] + eps * sigma[j, k]
                val = wp.clamp(val, -1.0, 1.0)
                if j == 0:
                    Y0s_out[wid, 0, k] = Ybar[0, k]
                else:
                    Y0s_out[wid, j, k] = val
    else:
        c = wp.sqrt(1.0 - rho * rho)
        for k in range(n_ctrl_dofs):
            e = float(0.0)
            for j in range(hnode_p1):
                eps = wp.randn(state)
                if j == 0:
                    e = eps
                else:
                    e = rho * e + c * eps
                val = Ybar[j, k] + e * sigma[j, k]
                val = wp.clamp(val, -1.0, 1.0)
                if j == 0:
                    Y0s_out[wid, 0, k] = Ybar[0, k]
                else:
                    Y0s_out[wid, j, k] = val

    if wid == 0:
        for j in range(hnode_p1):
            for k in range(n_ctrl_dofs):
                Y0s_out[nsample, j, k] = Ybar[j, k]


@wp.kernel
def _node2u_batch_kernel(
    mat: wp.array2d(dtype=float),
    Y0s: wp.array3d(dtype=float),
    uss_out: wp.array3d(dtype=float),
    hsample_p1: int,
    hnode_p1: int,
    n_ctrl_dofs: int,
):
    """Batched matmul uss[b] = mat @ Y0s[b], thread per trajectory."""
    bid = wp.tid()
    for i in range(hsample_p1):
        for k in range(n_ctrl_dofs):
            val = float(0.0)
            for j in range(hnode_p1):
                val = val + mat[i, j] * Y0s[bid, j, k]
            uss_out[bid, i, k] = val


@wp.kernel
def _copy_actions_kernel(
    uss: wp.array3d(dtype=float),
    actions_buf: wp.array3d(dtype=float),
    hsample: int,
    n_action_dim: int,
):
    """Copy uss[:, :Hsample, :] → actions_buf, dropping the trailing timestep."""
    wid = wp.tid()
    for t in range(hsample):
        for k in range(n_action_dim):
            actions_buf[wid, t, k] = uss[wid, t, k]


@wp.kernel
def _extract_rdphi_kernel(
    actions_buf: wp.array3d(dtype=float),
    rdphi_out: wp.array(dtype=float),
    step_t: int,
    rdphi_dim: int,
):
    """Route the last column of actions_buf into rdphi_wp at step_t."""
    wid = wp.tid()
    rdphi_out[wid] = actions_buf[wid, step_t, rdphi_dim]


@wp.kernel
def _broadcast_phi_kernel(
    src_phi: wp.array(dtype=float),
    dst_phi: wp.array(dtype=float),
):
    """Set all worlds' phi to src_phi[0]."""
    wid = wp.tid()
    dst_phi[wid] = src_phi[0]


@wp.kernel
def _softmax_weights_kernel(
    rewards: wp.array(dtype=float),
    weights_out: wp.array(dtype=float),
    temp_sample: float,
    nworld: int,
    n_elite: int,
):
    """Single-thread softmax over rewards. rew_ybar = rewards[nworld-1].

    When `n_elite < nworld`, only the top `n_elite` highest-reward rows survive
    (truncated MPPI): a fixed-iteration bisection finds the reward threshold that
    keeps ~n_elite rows, the rest are forced to zero weight, and the softmax
    renormalizes over the survivors. `n_elite >= nworld` is a pure no-op (the
    threshold stays at -1e30 so every row is kept — full-population softmax).
    """
    tid = wp.tid()
    if tid > 0:
        return

    rew_ybar = rewards[nworld - 1]
    mean_val = float(0.0)
    min_val = rewards[0]
    max_val = rewards[0]
    for i in range(nworld):
        r = rewards[i]
        mean_val = mean_val + r
        if r < min_val:
            min_val = r
        if r > max_val:
            max_val = r
    mean_val = mean_val / float(nworld)

    var_val = float(0.0)
    for i in range(nworld):
        d = rewards[i] - mean_val
        var_val = var_val + d * d
    var_val = var_val / float(nworld)
    std_val = wp.sqrt(var_val + REWARD_VAR_EPS)

    # Elite threshold: keep rewards >= thresh (~top n_elite). Bisection converges
    # the threshold to the n_elite-th largest reward over [min, max]; a fixed loop
    # count keeps this CUDA-graph-capturable. -1e30 (no truncation) keeps all.
    thresh = float(-1.0e30)
    if n_elite < nworld:
        lo = min_val
        hi = max_val
        for _k in range(32):
            mid = 0.5 * (lo + hi)
            cnt = int(0)
            for i in range(nworld):
                if rewards[i] >= mid:
                    cnt = cnt + 1
            if cnt >= n_elite:
                lo = mid  # still keeping >= n_elite, raise threshold
            else:
                hi = mid  # dropped below n_elite, lower threshold
        thresh = lo

    max_logp = float(-1.0e30)
    for i in range(nworld):
        if rewards[i] >= thresh:
            logp = (rewards[i] - rew_ybar) / std_val / temp_sample
        else:
            logp = float(-1.0e30)  # non-elite → exp underflows to 0 below
        weights_out[i] = logp
        if logp > max_logp:
            max_logp = logp

    total = float(0.0)
    for i in range(nworld):
        e = wp.exp(weights_out[i] - max_logp)
        weights_out[i] = e
        total = total + e

    for i in range(nworld):
        weights_out[i] = weights_out[i] / total


@wp.kernel
def _weighted_sum_Y_kernel(
    weights: wp.array(dtype=float),
    Y0s: wp.array3d(dtype=float),
    Ybar_out: wp.array2d(dtype=float),
    nworld: int,
    n_ctrl_dofs: int,
):
    """Ybar = Σ_i w_i · Y0s[i]. Thread per node."""
    j = wp.tid()
    for k in range(n_ctrl_dofs):
        val = float(0.0)
        for i in range(nworld):
            val = val + weights[i] * Y0s[i, j, k]
        Ybar_out[j, k] = val

