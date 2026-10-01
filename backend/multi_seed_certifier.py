"""Multi-seed certification: frozen cache + frozen Nifty, per-seed artifact isolation, IC-first bars."""
import os, json, time, logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List
import numpy as np, pandas as pd, joblib
from nse_research import ensemble_gain

logger = logging.getLogger(__name__)

@dataclass
class SeedResult:
    seed: int
    ic_open: float = 0.0
    ic_close: float = 0.0
    ic_nw_t: float = 0.0
    icir_nw: float = 0.0
    decile_t_nw: float = 0.0
    sharpe_cohort: float = 0.0
    buy_p: float = 1.0
    sell_p: float = 1.0
    train_s: float = 0.0

@dataclass
class CertificationReport:
    timestamp: str = ""
    cache_hash: str = ""
    seeds: List[SeedResult] = field(default_factory=list)
    med_ic: float = 0.0; worst_ic: float = 0.0
    med_icir: float = 0.0
    med_sharpe: float = 0.0; worst_sharpe: float = 0.0
    med_open_over_close: float = 0.0
    seed_rank_corr: float = 0.0
    projected_ic_5seed_ensemble: float = 0.0
    deployment_ready: bool = False
    rejection_reasons: List[str] = field(default_factory=list)

class MultiSeedCertifier:
    BARS = dict(
        ic_med_min=0.02, 
        ic_worst_min=0.01, 
        icir_med_min=0.5,
        open_over_close_min=0.6, 
        sharpe_med_min=0.5, 
        sharpe_worst_min=0.0
    )

    def __init__(self, predictor_class, k_seeds=5, base_seed=42, cache_path=None,
                 nifty_path=None, output_dir="certification_runs"):
        self.predictor_class = predictor_class
        self.seeds = [base_seed + 7 * i for i in range(k_seeds)]
        self.cache_path = cache_path or os.path.join('unified_models', 'engineered_features_cache.pkl')
        self.nifty_path = nifty_path or os.path.join(output_dir, 'nifty_frozen.pkl')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _hash(self):
        import hashlib
        h = hashlib.sha256()
        with open(self.cache_path, 'rb') as f:
            for ch in iter(lambda: f.read(1 << 20), b''):
                h.update(ch)
        if os.path.exists(self.nifty_path):
            with open(self.nifty_path, 'rb') as f:
                for ch in iter(lambda: f.read(1 << 20), b''):
                    h.update(ch)
        return h.hexdigest()[:16]

    def _train_one_seed(self, seed, frozen_df):
        import MLPredictor as M
        old_seed, old_nifty = M.CONFIG.get('random_seed'), M.CONFIG.get('frozen_nifty_path')
        M.CONFIG['random_seed'] = seed
        M.CONFIG['frozen_nifty_path'] = self.nifty_path        # first run writes it, later runs read it
        M.CONFIG['test_embargo_days'] = max(45, int(M.CONFIG.get('pred_days', 5) * 1.5)) # Prevent 30d head leakage
        M.CONFIG['purge_gap_calendar_days'] = M.CONFIG['test_embargo_days']
        p = self.predictor_class()
        p._set_artifact_base_dir(os.path.join(self.output_dir, f"seed_{seed}"))
        p.load_or_engineer_features = lambda **kw: frozen_df.copy()
        t0 = time.time()
        try:
            p.train()
        finally:
            M.CONFIG['random_seed'], M.CONFIG['frozen_nifty_path'] = old_seed, old_nifty
        tm = joblib.load(f"{M.METRICS_DIR}/test_metrics.pkl")
        bt = tm['backtest']['long_only_variant']
        ric = getattr(p, '_rank_ic_report', {}) or {}
        r = SeedResult(
            seed=seed,
            ic_open=float(ric.get('rank_ic_mean', 0)), ic_close=float(ric.get('rank_ic_mean_close_entry', 0)),
            ic_nw_t=float(ric.get('rank_ic_nw_t', 0)), icir_nw=float(ric.get('rank_ic_ir_annualized', 0)),
            decile_t_nw=float(ric.get('decile_spread_t_stat', 0)),
            sharpe_cohort=float(bt.get('sharpe_ratio', 0)),
            buy_p=float(p._buy_side_significance.get('permutation', {}).get('p_value', 1.0)),
            sell_p=float(p._sell_side_significance.get('permutation', {}).get('p_value', 1.0)),
            train_s=time.time() - t0)
        logger.info(f"[seed {seed}] IC_open={r.ic_open:+.4f} IC_close={r.ic_close:+.4f} NW-t={r.ic_nw_t:.2f} "
                    f"Sharpe={r.sharpe_cohort:.2f} buy-p={r.buy_p:.3f} ({r.train_s:.0f}s)")
        return r, np.asarray(p._test_logits, np.float32), np.asarray(p._test_dates)

    def certify(self) -> CertificationReport:
        frozen = pd.read_pickle(self.cache_path)
        res, logits, dates = [], [], None
        for s in self.seeds:
            r, lg, dates = self._train_one_seed(s, frozen)
            res.append(r); logits.append(lg)
        ic = np.array([r.ic_open for r in res]); sh = np.array([r.sharpe_cohort for r in res])
        oc = np.array([r.ic_open / r.ic_close if abs(r.ic_close) > 1e-9 else 0.0 for r in res])
        df = pd.DataFrame({f's{i}': l for i, l in enumerate(logits)}); df['d'] = dates
        cols = [c for c in df if c != 'd']
        R = df.groupby('d')[cols].rank(pct=True).corr().values
        rho = float(R[np.triu_indices(len(res), 1)].mean()) if len(res) > 1 else 1.0
        B = self.BARS
        rep = CertificationReport(
            timestamp=datetime.now().isoformat(), cache_hash=self._hash(), seeds=res,
            med_ic=float(np.median(ic)), worst_ic=float(ic.min()),
            med_icir=float(np.median([r.icir_nw for r in res])),
            med_sharpe=float(np.median(sh)), worst_sharpe=float(sh.min()),
            med_open_over_close=float(np.median(oc)), seed_rank_corr=rho,
            projected_ic_5seed_ensemble=float(np.median(ic) * ensemble_gain(rho, 5)))
        rej = []
        if rep.med_ic < B['ic_med_min']: rej.append(f"median IC {rep.med_ic:.4f} < {B['ic_med_min']}")
        if rep.worst_ic < B['ic_worst_min']: rej.append(f"worst IC {rep.worst_ic:.4f} < {B['ic_worst_min']}")
        if rep.med_icir < B['icir_med_min']: rej.append(f"median NW-ICIR {rep.med_icir:.2f} < {B['icir_med_min']}")
        if rep.med_open_over_close < B['open_over_close_min']:
            rej.append(f"IC_open/IC_close {rep.med_open_over_close:.2f} < {B['open_over_close_min']} (edge is close-only)")
        if rep.med_sharpe < B['sharpe_med_min']: rej.append(f"median cohort Sharpe {rep.med_sharpe:.2f} < {B['sharpe_med_min']}")
        if rep.worst_sharpe < B['sharpe_worst_min']: rej.append(f"worst cohort Sharpe {rep.worst_sharpe:.2f} < {B['sharpe_worst_min']}")
        rep.rejection_reasons, rep.deployment_ready = rej, not rej
        logger.info(f"CERT: IC med/worst {rep.med_ic:+.4f}/{rep.worst_ic:+.4f} | Sharpe med/worst "
                    f"{rep.med_sharpe:.2f}/{rep.worst_sharpe:.2f} | seed rho={rho:.2f} -> 5-seed IC≈"
                    f"{rep.projected_ic_5seed_ensemble:+.4f} | {'READY' if rep.deployment_ready else 'NOT READY'}")
        for x in rej: logger.info(f"  REJECT: {x}")
        path = os.path.join(self.output_dir, f"cert_{rep.cache_hash}_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, 'w') as f: json.dump(asdict(rep), f, indent=2, default=str)
        return rep

    def horizon_sweep(self, horizons=(3, 5, 10, 20)):
        """Run with k_seeds=1 first (4 x ~17 min); confirm on K=5 only for the winner."""
        import MLPredictor as M
        out, orig = {}, M.CONFIG['pred_days']
        try:
            for h in horizons:
                M.CONFIG['pred_days'] = h
                out[h] = self.certify()
        finally:
            M.CONFIG['pred_days'] = orig
        best = max(out, key=lambda h: out[h].med_ic)
        logger.info(f"Best horizon {best}d (IC={out[best].med_ic:+.4f})")
        return out

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from MLPredictor import UnifiedStockPredictor
    MultiSeedCertifier(UnifiedStockPredictor, k_seeds=5).certify()