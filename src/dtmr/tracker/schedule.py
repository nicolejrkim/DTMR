"""How much search the tracker spends at each MPC step, and at what noise.

Mixed into `Tracker`; the methods read and write the controller's state on `self`.
  - noise per anneal iteration (`_sigma_iter`), the per-step base noise
    (`_select_sigma_base`) and its per-frame gain (`_build_sigma_frame_gain`);
  - the diffusion budget per step (`_diffusion_budget`): cold start at t=0, the
    warm-up ramp, the steady warm-start tail, and the extra iterations on dynamic
    reference frames (`_build_reference_dynamism`);
  - the certainty-scheduled noise ladder (`_init_certainty_anneal`,
    `_update_certainty_entry`), and the start-of-run summary (`_announce_search_schedule`).
"""
from __future__ import annotations

import numpy as np


class ScheduleMixin:
    def _sigma_iter(self, i: int, start_exp: int) -> np.ndarray:
        """Sigma matrix for anneal iteration `i`. Joint columns decay with
        traj_diffuse_factor at exponent (start_exp + i); the trailing dphi
        column may carry its own factor (traj_diffuse_factor_phi) and, with
        phi_share_band=False, ignores start_exp so phase noise re-anneals from
        full sigma every reverse pass."""
        sig = self._sigma_base * (self.traj_diffuse_factor ** (start_exp + i))
        if self._phi_sched_decoupled:
            exp_phi = (start_exp + i) if self.phi_share_band else i
            sig[:, -1] = self._sigma_base[:, -1] * (
                self.traj_diffuse_factor_phi ** exp_phi)
        return sig

    def _build_reference_dynamism(self) -> None:
        """Per-frame dynamism of the reference (`self._ref_dyn`), used by the
        Ndiffuse_dyn schedule to give busy frames a deeper search."""
        env = self.env
        # Reference-dynamism schedule (Ndiffuse_dyn): per-frame signal from the
        # reference — smoothed mean joint speed + 2*|dz/dt|. Frames above
        # dyn_thresh get deep search (sustains energy-pumping modes); quiet
        # frames keep the cheap tail.
        self._ref_dyn = None
        if (int(self.args.Ndiffuse_dyn) > 0
                and self.args.Ndiffuse_steady >= 0):
            _sr = getattr(env, "_state_ref", None)
            if _sr is not None and "qpos" in _sr:
                _rq = np.asarray(_sr["qpos"], dtype=np.float32)
                _jd = np.abs(np.diff(_rq[:, 7:], axis=0)) / self.ctrl_dt
                _tk = int(self.args.dyn_sig_topk)
                if _tk > 0:
                    _js = (np.sort(_jd, axis=1)[:, -_tk:].mean(axis=1)
                           * float(self.args.dyn_sig_topk_scale))
                else:
                    _js = _jd.mean(axis=1)
                _zv = np.abs(np.diff(_rq[:, 2])) / self.ctrl_dt
                _d = _js + 2.0 * _zv
                _wa = float(self.args.dyn_sig_ang)
                if _wa > 0.0:
                    _q = _rq[:, 3:7] / np.maximum(
                        np.linalg.norm(_rq[:, 3:7], axis=1, keepdims=True), 1e-9)
                    _dots = np.clip(np.abs(np.sum(_q[:-1] * _q[1:], axis=1)), -1.0, 1.0)
                    _d = _d + _wa * (2.0 * np.arccos(_dots) / self.ctrl_dt)
                    print(f"[dyn-sig] ang term on (w={_wa:g}), topk={_tk}, "
                          f"smooth={int(self.args.dyn_sig_smooth)}")
                _k = max(1, int(self.args.dyn_sig_smooth))
                _d = np.convolve(_d, np.ones(_k) / _k, mode="same")
                self._ref_dyn = np.concatenate([_d, _d[-1:]])  # (T,)
                _th = float(self.args.dyn_thresh)
                print(f"[dyn] reference dynamism: {100.0 * (self._ref_dyn > _th).mean():.0f}% "
                      f"of frames > thresh {_th:g} -> deep budget "
                      f"{int(self.args.Ndiffuse_dyn)} iters")

    def _build_sigma_frame_gain(self) -> None:
        """Per-frame noise gain (`self._sigma_g`) from the dynamism signal
        (sigma_frame_gamma); None when the feature is off."""
        # Per-frame sigma modulation (sigma_frame_gamma): reuse the reference
        # dynamism signal, self-normalized by the clip's own mean (burstiness,
        # not absolute level, predicts hot-sigma preference), with a forward-max
        # look-ahead so noise heats up while the burst is still in the horizon.
        self._sigma_g = None
        _sfg = float(self.args.sigma_frame_gamma)
        if _sfg > 0.0:
            if self._ref_dyn is None:
                print("[sigma-frame] WARNING: sigma_frame_gamma set but the "
                      "reference dynamism signal is unavailable (needs "
                      "Ndiffuse_dyn > 0) — per-frame sigma OFF")
            else:
                _d = self._ref_dyn
                _lo = float(self.args.sigma_frame_lo)
                _hi = float(self.args.sigma_frame_hi)
                # Normalize by a ROLLING local level (norm_s seconds), not the
                # clip mean: sustained periodic activity is its own steady
                # state (the conveyor amortizes it) and should read g ~ 1;
                # only transient elevation — one-shot bursts — heats up.
                # norm_s <= 0 falls back to clip-mean normalization.
                _ns = float(self.args.sigma_frame_norm_s)
                if _ns > 0.0:
                    _w = max(1, int(round(_ns / self.ctrl_dt)))
                    _rm = np.convolve(_d, np.ones(_w) / _w, mode="same")
                    _den = np.maximum(_rm, max(0.25 * float(_d.mean()), 1e-3))
                else:
                    _den = max(float(_d.mean()), 1e-3)
                _gf = (_d / _den) ** _sfg
                _L = max(0, int(round(float(self.args.sigma_frame_lead) / self.ctrl_dt)))
                if _L > 0:
                    _pad = np.concatenate(
                        [_gf, np.full(_L, _gf[-1], dtype=_gf.dtype)])
                    _gf = np.array(
                        [_pad[i:i + _L + 1].max() for i in range(len(_gf))],
                        dtype=np.float32)
                # Budget-neutral: renormalize to mean 1 AFTER the look-ahead
                # (which biases upward), so the schedule reallocates the same
                # average noise toward bursts rather than running hotter.
                _gf = _gf / max(float(_gf.mean()), 1e-3)
                _gf = np.clip(_gf, _lo, _hi)
                self._sigma_g = _gf.astype(np.float32)
                print(f"[sigma-frame] gamma={_sfg:g} norm_s={_ns:g} "
                      f"lead={_L} frames, "
                      f"g range [{_gf.min():.2f},{_gf.max():.2f}] "
                      f"mean {_gf.mean():.2f}, "
                      f"hot(g>1.1) {100.0 * (_gf > 1.1).mean():.0f}% of frames")

    def _init_certainty_anneal(self) -> None:
        """Reset the certainty-anneal state (sigma_cert) and announce its settings."""
        # Closed-loop certainty anneal / probe announce (see config.sigma_cert).
        self._cert_ess_log = None
        self._cert_entry = None
        self._cert_rew_ema = None
        if (int(self.args.sigma_cert)
                and int(self.args.sigma_cert_entry_mode)):
            self._cert_entry = int(self.args.sigma_cert_start_exp)
            print(f"[sigma-cert] entry thermostat ON: reward EMA "
                  f"{int(self.args.sigma_cert_rew_ema)} steps, "
                  f"heat below {float(self.args.sigma_cert_rew_bad):g}, "
                  f"cool above {float(self.args.sigma_cert_rew_good):g}")
        _sc = int(self.args.sigma_cert)
        _scp = int(self.args.sigma_cert_probe)
        if _sc or _scp:
            self._cert_ess_log = []
            if _sc:
                _se = int(self.args.sigma_cert_start_exp)
                _t = float(self.args.sigma_cert_temp)
                print(f"[sigma-cert] closed-loop anneal ON: entry exp {_se} "
                      f"(sigma {self.args.sigma_scale * self.traj_diffuse_factor ** _se:.3f}), "
                      f"floor exp {int(self.args.sigma_cert_end_exp)}, "
                      f"flat-stop ESS>{float(self.args.sigma_cert_flat_ess):g}, "
                      f"rung-skip headroom<{float(self.args.sigma_cert_skip_imp):g}, "
                      f"cap {int(self.args.sigma_cert_max_iters)}, "
                      f"temp {(_t if _t > 0 else float(self.args.temp_sample)):g}")
            else:
                print("[sigma-cert] PROBE mode: logging per-iteration ESS only")

    def _announce_search_schedule(self) -> None:
        """Reset the adaptive-ladder counters and announce the adaptive ladder and
        MPC decimation settings."""
        # Adaptive full-vs-tail ladder announce (see config.adapt_ladder).
        self._adapt_n_full = 0
        self._adapt_n_short = 0
        if int(self.args.adapt_ladder):
            _cnd = int(self.args.adapt_ladder_certain_nd)
            _th = float(self.args.adapt_ladder_thresh)
            if int(self.args.adapt_ladder_mode) == 1:
                _cse = max(0, min(int(self.args.adapt_ladder_certain_start_exp),
                    self.args.Ndiffuse - 1))
                _st = (((self.args.Ndiffuse - 1 - _cse) / (_cnd - 1))
                       if _cnd > 1 else 0.0)
                print(f"[adapt-ladder] ON (SPAN mode): last-iter gain > {_th:g} "
                      f"-> full {self.args.Ndiffuse} rungs; else {_cnd} rungs "
                      f"spread over exps {_cse}..{self.args.Ndiffuse - 1} "
                      f"(stride {_st:g})")
            else:
                print(f"[adapt-ladder] ON: last-iter gain > {_th:g} -> full "
                      f"{self.args.Ndiffuse} rungs (exp 0); else tail {_cnd} rungs "
                      f"(exps {self.args.Ndiffuse - _cnd}..{self.args.Ndiffuse - 1})")
        if (int(self.args.adapt_ladder)
                and int(self.args.adapt_ladder_warm)):
            _se = int(self.args.Ndiffuse_init_start_exp)
            print(f"[adapt-ladder] WARM mode: confused steps re-run the cold "
                  f"ladder ({self.args.Ndiffuse_init} rungs from exp {_se}, "
                  f"sigma {self.args.sigma_scale * self.traj_diffuse_factor ** _se:.3f}); "
                  f"certain steps keep the warm-start tail")
        if int(self.args.mpc_decimation) > 1:
            _k = int(self.args.mpc_decimation)
            print(f"[mpc-decimation] ON: solving every {_k}. step; "
                  f"plan executes open-loop in between")

    def _diffusion_budget(self, t: int, phi: float) -> tuple:
        """(number of diffusion iterations, first anneal exponent) for MPC step `t`."""
        # Cold step: full anneal from full noise. Warm steps (t>=1): if
        # Ndiffuse_steady is set, run that many iters over the low-noise TAIL
        # of the anneal (warm-start); else Ndiffuse iters from exponent 0.
        self._adapt_stride = 1.0
        if t == 0:
            n_diffuse = self.args.Ndiffuse_init
            start_exp = int(self.args.Ndiffuse_init_start_exp)
            if start_exp != 0:
                print(f"[cold-start] Ndiffuse_init_start_exp={start_exp}: "
                      f"t=0 anneal exps {start_exp}..{start_exp+n_diffuse-1}")
        elif self.args.Ndiffuse_steady >= 0:
            if t < int(self.args.Ndiffuse_warmup_steps):
                # Warm-up ramp: the full schedule while the plan conveyor is
                # still cold, with an optional linear decay to the steady floor.
                _wu = int(self.args.Ndiffuse_warmup_steps)
                if bool(self.args.Ndiffuse_warmup_decay):
                    _fl = max(1, self.args.Ndiffuse_steady)
                    n_diffuse = max(_fl, int(round(
                        self.args.Ndiffuse
                        - (self.args.Ndiffuse - _fl) * t / max(1, _wu))))
                else:
                    n_diffuse = self.args.Ndiffuse
                start_exp = int(self.args.Ndiffuse_warmup_start_exp)
                if start_exp != 0 and t == 1:
                    print(f"[ramp] Ndiffuse_warmup_start_exp={start_exp}: window anneals "
                          f"exps {start_exp}..{start_exp+n_diffuse-1}")
            elif self._ref_dyn is not None:
                # Dynamism ladder: entry sigma computed from the
                # reference's local dynamism (log-linear: sigma_entry
                # proportional to required correction), then anneal
                # down the full ladder to the small-sigma floor.
                # n_iters = rungs from e0 to Ndiffuse_init-1 (2..16).
                _f = min(int(phi * (len(self._ref_dyn) - 1)),
                         len(self._ref_dyn) - 1)
                _dyn = float(self._ref_dyn[_f])
                _lo = float(self.args.dyn_lo)
                _hi = float(self.args.dyn_hi)
                _ee = int(self.args.dyn_end_exp)
                _e_end = (_ee if _ee >= 0
                          else self.args.Ndiffuse_init - 1)
                # quiet entry: Ndiffuse_steady iters ending at _e_end
                _e_tail = max(0, _e_end
                              - (self.args.Ndiffuse_steady - 1))
                _ths = tuple(self.args.dyn_thresholds or ())
                if _ths:
                    # quantile map: one rung deeper per threshold crossed
                    _e0 = _e_tail - sum(1 for _t in _ths if _dyn > _t)
                elif _dyn <= _lo:
                    _e0 = _e_tail
                else:
                    _x = min(1.0, np.log(max(_dyn, 1e-6) / _lo)
                             / np.log(_hi / _lo))
                    _e0 = int(round((1.0 - _x) * _e_tail))
                _fl = int(self.args.dyn_entry_floor_exp)
                _e0 = max(0 if _fl < 0 else _fl, min(_e_tail, _e0))
                n_diffuse = min(_e_end - _e0 + 1,
                                int(self.args.Ndiffuse_dyn))
                start_exp = _e_end - n_diffuse + 1
            else:
                n_diffuse = self.args.Ndiffuse_steady
                _wse = self.args.Ndiffuse_warm_start_exp
                start_exp = (_wse if _wse >= 0
                             else max(0, self.args.Ndiffuse_init - n_diffuse))
        elif int(self.args.adapt_ladder):
            # Previous step's final diffusion iteration decides depth:
            # non-finite improvement (all-NaN rollouts) counts as
            # confused — the safe direction.
            _imp = float(getattr(self, "_last_improvement", np.inf))
            if not np.isfinite(_imp) or _imp > float(
                    self.args.adapt_ladder_thresh):
                n_diffuse, start_exp = self.args.Ndiffuse, 0
                self._adapt_n_full += 1
            else:
                n_diffuse = min(self.args.Ndiffuse, max(1, int(
                    self.args.adapt_ladder_certain_nd)))
                if int(self.args.adapt_ladder_mode) == 1:
                    # sparse span: certain_nd rungs from certain_start_exp
                    # to Ndiffuse-1 via fractional exponent stride
                    _cse = max(0, min(int(self.args.adapt_ladder_certain_start_exp),
                        self.args.Ndiffuse - 1))
                    start_exp = _cse
                    if n_diffuse > 1:
                        self._adapt_stride = (
                            (self.args.Ndiffuse - 1 - _cse)
                            / (n_diffuse - 1))
                else:
                    start_exp = self.args.Ndiffuse - n_diffuse
                self._adapt_n_short += 1
        else:
            n_diffuse, start_exp = self.args.Ndiffuse, 0
        # Warm-start + confusion rule: the warm branches above own the
        # schedule, so the override lands here — a confused step throws
        # away the cheap tail and re-runs the cold-start ladder.
        if (t > 0 and int(self.args.adapt_ladder)
                and int(self.args.adapt_ladder_warm)):
            _imp = float(getattr(self, "_last_improvement", np.inf))
            if not np.isfinite(_imp) or _imp > float(
                    self.args.adapt_ladder_thresh):
                n_diffuse = self.args.Ndiffuse_init
                start_exp = int(self.args.Ndiffuse_init_start_exp)
                self._adapt_n_full += 1
            else:
                self._adapt_n_short += 1

        return n_diffuse, start_exp

    def _select_sigma_base(self, t: int, phi: float) -> None:
        """Choose the noise envelope for this step (`self._sigma_base`)."""
        # Flat sigma envelope during the start window: node 0 gets full
        # sampling noise so MPPI can optimize the executed action too.
        self._sigma_base = (
            self.sigma_control_flat
            if t < int(self.args.sigma_flat_steps)
            else self.sigma_control)
        if self._sigma_g is not None:
            _fg = min(int(phi * (len(self._sigma_g) - 1)),
                      len(self._sigma_g) - 1)
            _gt = float(self._sigma_g[_fg])
            if _gt != 1.0:
                _sb = self._sigma_base.copy()
                _lower = self.env.joint_indices(
                    *[g for g in ("legs", "waist") if g in self.env.joint_groups])
                _sb[:, _lower] *= _gt  # legs/waist; arms + dphi keep base
                self._sigma_base = _sb

    def _update_certainty_entry(self, reward: float) -> None:
        """Entry thermostat of the certainty anneal: the executed reward moves the
        next step's entry rung."""
        if getattr(self, "_cert_entry", None) is not None:
            # entry thermostat: EMA of executed reward drives the
            # next step's entry rung (bad tracking -> hotter).
            _a = 1.0 / max(1, int(self.args.sigma_cert_rew_ema))
            self._cert_rew_ema = (
                reward if self._cert_rew_ema is None
                else (1 - _a) * self._cert_rew_ema + _a * reward)
            if self._cert_rew_ema < float(self.args.sigma_cert_rew_bad):
                self._cert_entry -= 1
            elif self._cert_rew_ema > float(self.args.sigma_cert_rew_good):
                self._cert_entry += 1
            self._cert_entry = int(np.clip(
                self._cert_entry,
                int(self.args.sigma_cert_start_exp),
                int(self.args.sigma_cert_end_exp)))
