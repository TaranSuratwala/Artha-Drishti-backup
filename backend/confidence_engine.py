"""
confidence_engine.py -- Artha-Drishti calibrated-confidence + NSE-aware sizing layer.

Drop next to MLPredictor.py. numpy / scipy / scikit-learn only (no torch).

Pipeline (each stage is independently testable):
  1. RegimeCalibrator   : stage-2 Platt per India-VIX regime, hierarchically shrunk to global.
  2. decompose_*        : law-of-total-variance split of seed x MC draws (epistemic vs aleatoric).
  3. shrink_to_base     : normal-normal empirical-Bayes shrinkage -> p_shrunk, p_lcb, edge_confidence.
  4. AdaptiveCQR        : conformalised quantile regression + ACI on the P10/P90 return head.
  5. size_position      : net-of-cost Kelly x drawdown-fraction, capped by risk/ADV/gap/VIX/DD.

Time-series hygiene: fit / update ONLY with matured outcomes (t + h sessions) and keep an
h-day embargo between calibration and evaluation folds.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# --------------------------------------------------------------------------- #
# 0. Diagnostics
# --------------------------------------------------------------------------- #
def calibration_report(p, y, n_bins: int = 10) -> Dict[str, float]:
    """Equal-MASS bins (fixed-width bins are empty when p lives in 0.45-0.60)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = [b for b in np.array_split(np.argsort(p), n_bins) if len(b)]
    gaps = [abs(p[b].mean() - y[b].mean()) for b in bins]
    ece = float(sum(len(b) / len(p) * g for b, g in zip(bins, gaps)))
    base = y.mean() * (1 - y.mean())
    brier = float(np.mean((p - y) ** 2))
    slope = intercept = float("nan")
    if 0 < y.sum() < len(y):
        lr = LogisticRegression(C=1e6, max_iter=300).fit(_logit(p)[:, None], y)
        slope, intercept = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return dict(ece=ece, mce=float(max(gaps)), brier=brier,
                brier_skill=float(1 - brier / max(base, EPS)),
                slope=slope, intercept=intercept)


# --------------------------------------------------------------------------- #
# 1. Regime-conditional calibration (stage 2, on top of existing T/Platt/Iso)
# --------------------------------------------------------------------------- #
def vix_regime(vix_z: float, lo: float = -0.5, hi: float = 1.0) -> int:
    """0 = calm, 1 = normal, 2 = stress. vix_z = z-score of log(India VIX) vs 60d."""
    return 0 if vix_z < lo else (2 if vix_z > hi else 1)


class RegimeCalibrator:
    """Platt (a, b) per regime: theta_r = w*theta_hat_r + (1-w)*theta_global, w = n_r/(n_r+n0)."""

    def __init__(self, n0: int = 2000, min_fit: int = 200):
        self.n0, self.min_fit = n0, min_fit
        self.params: Dict[object, Tuple[float, float]] = {"global": (1.0, 0.0)}

    @staticmethod
    def _platt(z, y):
        if len(y) < 50 or y.min() == y.max():
            return 1.0, 0.0
        lr = LogisticRegression(C=1e4, max_iter=300).fit(z[:, None], y)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])

    def fit(self, p_in, y, regime) -> "RegimeCalibrator":
        z, y, regime = _logit(p_in), np.asarray(y, float), np.asarray(regime)
        ga, gb = self._platt(z, y)
        self.params = {"global": (ga, gb)}
        for r in np.unique(regime):
            m = regime == r
            if m.sum() < self.min_fit:
                continue
            a, b = self._platt(z[m], y[m])
            w = m.sum() / (m.sum() + self.n0)
            self.params[int(r)] = (w * a + (1 - w) * ga, w * b + (1 - w) * gb)
        return self

    def transform(self, p_in, regime: int):
        a, b = self.params.get(int(regime), self.params["global"])
        return _sigmoid(a * _logit(p_in) + b)


# --------------------------------------------------------------------------- #
# 2-3. Uncertainty decomposition and Bayesian shrinkage
# --------------------------------------------------------------------------- #
def decompose_uncertainty(draws) -> Dict[str, float]:
    d = np.asarray(draws, float).ravel()
    m = float(d.mean())
    v = float(d.var(ddof=1)) if d.size > 1 else 0.0
    return dict(p_mean=m, var_epistemic=v, aleatoric=max(m * (1 - m) - v, 0.0))


def estimate_tau2(p_cal_hist, var_epi_hist, p0: float = 0.5, floor: float = 1e-5) -> float:
    return float(max(np.var(np.asarray(p_cal_hist, float) - p0) - np.mean(var_epi_hist), floor))


def shrink_to_base(p_cal: float, var_epi: float, tau2: float, p0: float = 0.5,
                   delta: float = 0.20) -> Dict[str, float]:
    kappa = tau2 / (tau2 + var_epi)
    p_s = p0 + kappa * (p_cal - p0)
    sd = math.sqrt(kappa * var_epi)
    return dict(kappa=kappa, p_shrunk=p_s, sd_post=sd, p_lcb=p_s - norm.ppf(1 - delta) * sd)


# --------------------------------------------------------------------------- #
# 4. Conformalised quantile regression with adaptive alpha (ACI)
# --------------------------------------------------------------------------- #
class AdaptiveCQR:
    def __init__(self, alpha: float = 0.20, gamma: float = 0.01, window: int = 2000, min_n: int = 200):
        self.alpha, self.alpha_t, self.gamma, self.min_n = alpha, alpha, gamma, min_n
        self.scores: deque = deque(maxlen=window)

    def _q(self) -> Optional[float]:
        n = len(self.scores)
        if n < self.min_n:
            return None
        a = min(max(self.alpha_t, 1.0 / (n + 1)), 0.999)
        level = min(1.0, math.ceil((n + 1) * (1 - a)) / n)
        return float(np.quantile(np.fromiter(self.scores, float), level, method="higher"))

    def interval(self, q_lo: float, q_hi: float) -> Optional[Tuple[float, float]]:
        q = self._q()
        return None if q is None else (q_lo - q, q_hi + q)

    def update(self, y: float, q_lo: float, q_hi: float) -> None:
        s = max(q_lo - y, y - q_hi)
        q = self._q()
        if q is not None:
            err = 1.0 if s > q else 0.0
            self.alpha_t += self.gamma * (self.alpha - err)
        self.scores.append(s)


# --------------------------------------------------------------------------- #
# 5. NSE-aware position sizing
# --------------------------------------------------------------------------- #
@dataclass
class NSEContext:
    price: float
    adv_inr: float
    sigma_d: float
    spread_bps: float = 10.0
    vix_z: float = 0.0
    gap_adverse_p95: float = 0.03
    lock_recent: bool = False
    surveillance_flag: bool = False


@dataclass
class SizingConfig:
    equity: float = 1_000_000.0
    risk_per_trade: float = 0.005
    max_pos: float = 0.03
    dd_budget: float = 0.20
    dd_prob: float = 0.25
    dd_max: float = 0.15
    dd_exp: float = 1.5
    participation: float = 0.02
    fee_rt: float = 0.0022
    impact_y: float = 0.6
    vix_lambda: float = 0.35
    vix_free_z: float = 1.0
    vix_kill_z: float = 3.5
    min_edge_conf: float = 0.80
    min_adv_inr: float = 2e7


def kelly_fraction_for_drawdown(dd: float = 0.20, prob: float = 0.25) -> float:
    return float(np.clip(2.0 / (1.0 + math.log(prob) / math.log(1.0 - dd)), 0.05, 1.0))


def kelly_binary(p: float, gain: float, loss: float, cost: float) -> float:
    w, l = gain - cost, loss + cost
    return 0.0 if w <= 0 else max((p * w - (1 - p) * l) / (w * l), 0.0)


def breakeven_p(gain: float, loss: float, cost: float) -> float:
    return (loss + cost) / (gain + loss)


@dataclass
class SizingDecision:
    trade: bool
    reason: str
    fraction: float = 0.0
    notional: float = 0.0
    shares: int = 0
    diag: Dict[str, float] = field(default_factory=dict)

    def asdict(self):
        return asdict(self)


def kelly_gaussian(mu_lcb: float, sigma_h: float, cost: float) -> float:
    return max(mu_lcb - cost, 0.0) / max(sigma_h, 1e-6) ** 2


def size_position(p_win_lcb: float, gain: float, loss: float, edge_conf: float,
                  ctx: NSEContext, cfg: SizingConfig, drawdown: float = 0.0,
                  conformal_lo: Optional[float] = None,
                  mu_lcb: Optional[float] = None, sigma_h: Optional[float] = None) -> SizingDecision:
    def _kfull(c):
        return (kelly_gaussian(mu_lcb, sigma_h, c) if mu_lcb is not None
                else kelly_binary(p_win_lcb, gain, loss, c))

    if ctx.lock_recent: return SizingDecision(False, "circuit_lock_recent")
    if ctx.surveillance_flag: return SizingDecision(False, "surveillance_flag")
    if ctx.adv_inr < cfg.min_adv_inr: return SizingDecision(False, "illiquid")
    if ctx.vix_z >= cfg.vix_kill_z: return SizingDecision(False, "vix_kill_switch")
    if edge_conf < cfg.min_edge_conf: return SizingDecision(False, f"edge_conf<{cfg.min_edge_conf:.2f}")
    if drawdown >= cfg.dd_max: return SizingDecision(False, "drawdown_halt")

    c_frac = kelly_fraction_for_drawdown(cfg.dd_budget, cfg.dd_prob)
    tail = -conformal_lo if (conformal_lo is not None and conformal_lo < 0) else 0.0
    risk_unit = max(loss, ctx.gap_adverse_p95, tail)
    f_risk = cfg.risk_per_trade / max(risk_unit, EPS)
    f_adv = cfg.participation * ctx.adv_inr / cfg.equity

    vix_mult = math.exp(-cfg.vix_lambda * max(ctx.vix_z - cfg.vix_free_z, 0.0))
    dd_mult = max(1.0 - drawdown / cfg.dd_max, 0.0) ** cfg.dd_exp

    f = min(cfg.max_pos, f_risk, f_adv)
    cost = cfg.fee_rt + ctx.spread_bps / 1e4
    for _ in range(5):
        q = f * cfg.equity
        cost = cfg.fee_rt + ctx.spread_bps / 1e4 + 2 * cfg.impact_y * ctx.sigma_d * math.sqrt(q / ctx.adv_inr)
        f_k = _kfull(cost) * c_frac
        f = min(f_k, f_risk, f_adv, cfg.max_pos)
    f_full = _kfull(cost)
    f *= vix_mult * dd_mult

    diag = dict(p_be=breakeven_p(gain, loss, cost), cost_rt=cost, kelly_full=f_full, kelly_mult=c_frac,
                f_risk=f_risk, f_adv=f_adv, risk_unit=risk_unit, vix_mult=vix_mult, dd_mult=dd_mult)
    if f <= 0: return SizingDecision(False, "no_net_edge_at_p_lcb", diag=diag)
    shares = int(f * cfg.equity // ctx.price)
    if shares <= 0: return SizingDecision(False, "size_below_one_share", diag=diag)
    return SizingDecision(True, "ok", f, shares * ctx.price, shares, diag)


# --------------------------------------------------------------------------- #
# Facade used by MLPredictor.predict() / nse_tasks.score_universe()
# --------------------------------------------------------------------------- #
class ConfidenceEngine:
    def __init__(self, cfg: Optional[SizingConfig] = None, tau2: float = 1e-3, var_scale: float = 1.0,
                 p0: float = 0.5, delta: float = 0.20):
        self.cfg = cfg or SizingConfig()
        self.tau2, self.var_scale, self.p0, self.delta = tau2, var_scale, p0, delta
        self.calibrator = RegimeCalibrator()
        self.cqr = AdaptiveCQR()

    def assess(self, draws, vix_z: float, gain: float, loss: float, cost: float,
               is_long: bool = True) -> Dict[str, float]:
        reg = vix_regime(vix_z)
        d = np.asarray(draws, float).ravel()
        d = self.calibrator.transform(d if is_long else 1 - d, reg)
        u = decompose_uncertainty(d)
        var_e = u["var_epistemic"] * self.var_scale
        sh = shrink_to_base(u["p_mean"], var_e, self.tau2, self.p0, self.delta)
        p_be = breakeven_p(gain, loss, cost)
        edge_conf = float(norm.cdf((sh["p_shrunk"] - p_be) / max(sh["sd_post"], EPS)))
        return dict(regime=reg, p_win=u["p_mean"], var_epistemic=var_e, aleatoric=u["aleatoric"],
                    p_breakeven=p_be, edge_confidence=edge_conf, **sh)
