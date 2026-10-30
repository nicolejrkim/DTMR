"""TrackerConfig: the sampling-MPC settings of the tracker (search shape, diffusion
schedule, annealing). Every field can be set from the run YAML."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackerConfig:
    """Search shape, diffusion schedule and annealing of the sampling MPC."""
    seed: int = 0
    n_steps: int = 420                 # most control steps to run; the run stops when the reference ends
    Nsample: int = 2048                # rollouts per annealing iteration
    Hsample: int = 25                  # planning horizon in control steps
    Hnode: int = 5                     # spline control nodes over the horizon
    Ndiffuse: int = 4                  # annealing iterations per MPC step
    Ndiffuse_init: int = 16            # annealing iterations on the cold-start step
    # Anneal start exponent of the cold-start step: negative starts above sigma_scale
    # and anneals down over Ndiffuse_init iterations; 0 starts at sigma_scale.
    Ndiffuse_init_start_exp: int = 0
    # Settle pre-roll options (see settle_steps): pin the root while the joints and
    # contacts settle, and seed the plan with the measured gravity-sag offsets.
    settle_pin_root: bool = False
    settle_preload: bool = False
    # Anneal start exponent of the warm-up ramp steps (t < Ndiffuse_warmup_steps).
    Ndiffuse_warmup_start_exp: int = 0
    # Reference dynamism signal (drives the dynamism ladder and the per-frame sigma):
    # the weight of the root angular speed, the top-k joint speeds instead of the mean
    # (0 = all joints) with their scale, and the boxcar smoothing window in frames.
    dyn_sig_ang: float = 0.0
    dyn_sig_topk: int = 0
    dyn_sig_topk_scale: float = 1.0
    dyn_sig_smooth: int = 25
    # Sampling-noise multiplier on the arm joints (the descriptor's `arms` group).
    sigma_arm_mult: float = 1.0
    # Per-frame sigma on the leg and waist joints from the reference dynamism:
    # g(t) = clip((d(t) / mean(d))^gamma, lo, hi), with a forward look-ahead of `lead`
    # seconds and a rolling normaliser of `norm_s` seconds (<= 0: the clip mean).
    # gamma 0 = off.
    sigma_frame_gamma: float = 0.0
    sigma_frame_lo: float = 0.85
    sigma_frame_hi: float = 1.65
    sigma_frame_lead: float = 0.5
    sigma_frame_norm_s: float = 2.5
    # Closed-loop certainty anneal: each MPC step enters at start_exp and, after every
    # iteration, reads the sample-reward distribution. A headroom (best sample minus
    # incumbent) below skip_imp skips two rungs; a normalised ESS above flat_ess stops
    # the step's anneal; the walk also stops at end_exp or after max_iters. 0 = off,
    # in which case the dynamism ladder below applies.
    sigma_cert: int = 0
    sigma_cert_probe: int = 0          # 1: only log the ESS per iteration, no behaviour change
    sigma_cert_start_exp: int = -1
    sigma_cert_end_exp: int = 2
    sigma_cert_flat_ess: float = 0.55
    sigma_cert_skip_imp: float = -3.0
    sigma_cert_max_iters: int = 6
    # Entry thermostat (entry_mode 1): the entry rung persists across steps and follows
    # an EMA over rew_ema steps of the executed tracking reward; below rew_bad it heats
    # one rung, above rew_good it cools one, clamped to [start_exp, end_exp].
    sigma_cert_entry_mode: int = 0
    sigma_cert_rew_bad: float = -0.5
    sigma_cert_rew_good: float = -0.25
    sigma_cert_rew_ema: int = 10
    sigma_cert_temp: float = -1.0      # softmax temperature of the certainty statistics; -1: temp_sample
    # Adaptive ladder: a warm step runs the full Ndiffuse ladder when the previous step's
    # final-rung gain exceeded thresh, else only the last certain_nd rungs. mode 1 spreads
    # those rungs over the exponent span from certain_start_exp; warm 1 applies the rule
    # on top of a warm-start schedule (a confused step re-runs the cold-start ladder).
    adapt_ladder: int = 0
    adapt_ladder_certain_nd: int = 3
    adapt_ladder_thresh: float = 0.02
    adapt_ladder_mode: int = 0
    adapt_ladder_certain_start_exp: int = 0
    adapt_ladder_warm: int = 0
    # Solve the MPC only every k-th control step and execute the shifted plan in between.
    mpc_decimation: int = 1
    temp_sample: float = 0.05          # softmax temperature that weights the samples
    horizon_diffuse_factor: float = 0.9  # per-node sigma envelope along the horizon
    traj_diffuse_factor: float = 0.5   # per-iteration sigma decay
    sigma_scale: float = 1.0           # sampling noise of the joint controls
    sigma_scale_phi: float = -1.0      # sampling noise of the phase rate; -1: sigma_scale
    # The phase-rate column can anneal on its own: its per-iteration decay and per-node
    # envelope (-1: the joint values), and whether it shares the per-step start exponent.
    traj_diffuse_factor_phi: float = -1.0
    horizon_diffuse_factor_phi: float = -1.0
    phi_share_band: bool = True
    # Warm-start tail of the phase-rate column: carry the previous plan's last phase
    # node forward (True) or reset it to the reference rate (False).
    phi_tail_carry: bool = True
    # AR(1) correlation of the sampling noise across spline nodes; <= 0 = independent draws.
    noise_time_corr: float = 0.0
    elite_frac: float = 1.0            # fraction of samples kept in the softmax; 1.0 = all
    # Early termination of the per-step diffusion loop: stop when the best sample beats
    # the incumbent by less than improvement_threshold for improvement_check_steps
    # consecutive iterations. 0.0 = off.
    improvement_threshold: float = 0.0
    improvement_check_steps: int = 1
    # Warm-start budget: warm steps run Ndiffuse_steady iterations over the low-noise
    # tail of the cold schedule, from Ndiffuse_warm_start_exp (-1: Ndiffuse_init minus
    # Ndiffuse_steady). Ndiffuse_steady -1 = off: warm steps run Ndiffuse from exponent 0.
    Ndiffuse_steady: int = -1
    Ndiffuse_warm_start_exp: int = -1
    # Warm-up ramp: the first Ndiffuse_warmup_steps warm steps run the full schedule, or
    # a budget decaying linearly to the steady tail when Ndiffuse_warmup_decay is set.
    Ndiffuse_warmup_steps: int = 0
    Ndiffuse_warmup_decay: bool = False
    # Settle pre-roll: this many unlogged physics steps holding the first reference frame
    # before the MPC loop, so the power-on transient happens off-camera; 0 = off.
    # settle_mpc runs the optimiser during the pre-roll (phase held at 0) instead of a
    # fixed PD hold.
    settle_steps: int = 0
    settle_mpc: bool = False
    # Flat horizon sigma envelope for the first N MPC steps, so the executed node is
    # annealed like every other node while the start corrections are found. 0 = off.
    sigma_flat_steps: int = 0
    # Dynamism ladder (needs Ndiffuse_steady >= 0): the per-step budget follows the
    # reference dynamism. Ndiffuse_dyn > 0 enables it and caps the iterations per step.
    # Frames at or below dyn_lo enter at the tail, at or above dyn_hi at exponent 0,
    # log-linear in between; dyn_thresh is the threshold of the binned mode. The anneal
    # ends at dyn_end_exp (-1: Ndiffuse_init - 1) and dyn_entry_floor_exp caps the entry
    # (-1: none). dyn_thresholds, when set, replaces the log-linear map with quantile
    # thresholds, one rung per threshold exceeded.
    Ndiffuse_dyn: int = 0
    dyn_thresh: float = 2.5
    dyn_lo: float = 1.5
    dyn_hi: float = 3.0
    dyn_end_exp: int = -1
    dyn_entry_floor_exp: int = -1
    dyn_thresholds: tuple = ()
