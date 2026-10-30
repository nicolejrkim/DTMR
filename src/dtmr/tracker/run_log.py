"""The end-of-run summary, mixed into `Tracker` (it reads the controller's state on `self`)."""
from __future__ import annotations

import numpy as np


class RunLogMixin:
    def _summarise(self, log: dict) -> None:
        """Fill in the step count and print the end-of-run statistics."""
        log["steps"] = len(log["rewards"])
        if log["rewards"]:
            print(f"mean reward = {float(np.mean(log['rewards'])):.4f}")
        if int(self.args.adapt_ladder):
            _tot = max(1, self._adapt_n_full + self._adapt_n_short)
            _cnd = min(self.args.Ndiffuse,
                       int(self.args.adapt_ladder_certain_nd))
            _iters = (self._adapt_n_full * self.args.Ndiffuse
                      + self._adapt_n_short * _cnd) / _tot
            print(f"[adapt-ladder] full {self._adapt_n_full} / tail "
                  f"{self._adapt_n_short} steps "
                  f"({100.0 * self._adapt_n_full / _tot:.0f}% confused), "
                  f"mean iters/step {_iters:.2f} (full={self.args.Ndiffuse}, "
                  f"tail={_cnd})")
        if float(self.args.improvement_threshold) > 0.0 and log["diffuse_iters"]:
            di = np.asarray(log["diffuse_iters"])
            print(f"[early-term] diffuse iters/step: mean {di.mean():.2f} "
                  f"min {di.min()} max {di.max()} "
                  f"(cap {self.args.Ndiffuse}, thresh {self.args.improvement_threshold:g}, "
                  f"check {self.args.improvement_check_steps})")
        if getattr(self, "_cert_ess_log", None):
            _ce = np.asarray(self._cert_ess_log, dtype=np.float64)
            _mode = ("ACTIVE" if int(self.args.sigma_cert)
                     else "probe")
            print(f"[sigma-cert] ESS by rung exp over {len(_ce)} iterations "
                  f"({_mode}):")
            for _x in sorted(set(_ce[:, 1].astype(int))):
                _m = _ce[:, 1] == _x
                _v, _imp, _pb = _ce[_m, 2], _ce[_m, 3], _ce[_m, 4]
                _sig = self.args.sigma_scale * self.traj_diffuse_factor ** _x
                print(f"  exp {_x:+d} (sigma {_sig:.3f}): n={len(_v)} "
                      f"ess p10={np.percentile(_v, 10):.4f} "
                      f"p50={np.percentile(_v, 50):.4f} "
                      f"p90={np.percentile(_v, 90):.4f} | "
                      f"imp p50={np.percentile(_imp, 50):.4g} "
                      f"p90={np.percentile(_imp, 90):.4g} | "
                      f"p_beat p50={np.percentile(_pb, 50):.4f}")
            log["cert_ess"] = _ce.astype(np.float32)
            if int(self.args.sigma_cert) and log["diffuse_iters"]:
                di = np.asarray(log["diffuse_iters"])
                print(f"[sigma-cert] iters/step: mean {di.mean():.2f} "
                      f"min {di.min()} max {di.max()}")
                # entry histogram (first rung of each reverse call)
                _first = []
                for _c in np.unique(_ce[:, 0]):
                    _first.append(int(_ce[_ce[:, 0] == _c][0, 1]))
                _f = np.asarray(_first)
                _h = " ".join(f"{x:+d}:{(_f == x).mean() * 100:.0f}%"
                              for x in sorted(set(_f.tolist())))
                print(f"[sigma-cert] entry-rung distribution: {_h}")
