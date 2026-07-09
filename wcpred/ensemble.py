"""wcpred.ensemble — blend the XGBoost W/D/L model with the Dixon-Coles goals
model into a single calibrated predictor.

Method
------
1. Train XGBoost (`model_wdl.train_model`) and fit Dixon-Coles
   (`model_goals.fit`) on the same temporal split — everything strictly
   before `asof`, via `model_wdl.split_by_date` — so neither model ever sees
   future data relative to the match being predicted.
2. Blend: on the validation split, find a single scalar `alpha` in [0, 1]
   minimizing log-loss of `alpha * p_xgb + (1 - alpha) * p_dc` via a bounded
   1-D scalar search (`scipy.optimize.minimize_scalar`). A single global
   weight is enough here — both models are already fit on the full training
   history, so alpha is just "how much to trust the classifier vs. the goals
   model" on average.
3. Calibrate: fit one `IsotonicRegression` per outcome class (home/draw/away)
   mapping blended probability -> observed frequency on the *same* temporal
   validation split (one-vs-rest isotonic calibration, a simple and standard
   approach — see sklearn's `CalibratedClassifierCV` docs for the same idea
   applied per-class). At prediction time each class probability is pushed
   through its isotonic curve and the three outputs are renormalized to sum
   to 1.

This keeps the well-tested pieces (`predict_symmetric`, `score_matrix`,
`wdl_from_matrix`) untouched and only adds a thin blending/calibration layer
on top.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from wcpred import model_goals
from wcpred.model_wdl import (
    TRAIN_START,
    VAL_START,
    predict_symmetric,
    split_by_date,
    train_model,
)

EPS = 1e-9


def _normalize_rows(p):
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def _dc_val_probs(params, val):
    """Dixon-Coles W/D/L for every validation row, using the recorded
    home/away teams as-is (the label already reflects that assignment)."""
    out = np.empty((len(val), 3))
    for i, row in enumerate(val.itertuples()):
        M = model_goals.score_matrix(params, row.home_team, row.away_team,
                                     neutral=bool(row.neutral))
        p_home, p_draw, p_away = model_goals.wdl_from_matrix(M)
        out[i] = (p_home, p_draw, p_away)
    return out


class Predictor:
    def __init__(self, model, long, final_elo, params, alpha, calibrators, asof):
        self.model = model
        self.long = long
        self.final_elo = final_elo
        self.params = params
        self.alpha = alpha
        self.calibrators = calibrators  # list of 3 fitted IsotonicRegression
        self.asof = asof
        # ── ledger-feedback layer (set by wcpred.feedback.apply for LIVE
        # predictions only; defaults here reproduce the un-adjusted model) ──
        self.tournament_temperature = 1.0   # >1 softens confidence, <1 sharpens
        self.alpha_effective = None         # tournament-adjusted blend; None => self.alpha
        self.feedback_info = None           # dict of diagnostics for display

    def _calibrate(self, blended):
        calibrated = np.array([
            self.calibrators[c].predict([blended[c]])[0] for c in range(3)
        ])
        calibrated = np.clip(calibrated, EPS, None)
        return calibrated / calibrated.sum()

    def _temper(self, p):
        """Apply the tournament-fit temperature to a categorical: p_i^(1/T),
        renormalized. T=1.0 is a no-op."""
        T = self.tournament_temperature
        if T is None or T == 1.0:
            return p
        adj = np.clip(p, EPS, None) ** (1.0 / T)
        return adj / adj.sum()

    def predict(self, home, away, neutral, weight):
        p_xgb = np.array(predict_symmetric(
            self.model, self.long, self.final_elo, home, away, self.asof, neutral, weight
        ))
        M = model_goals.score_matrix(self.params, home, away, neutral=bool(neutral))
        p_dc = np.array(model_goals.wdl_from_matrix(M))

        alpha = self.alpha if self.alpha_effective is None else self.alpha_effective
        blended = alpha * p_xgb + (1 - alpha) * p_dc
        blended = blended / blended.sum()
        p_home, p_draw, p_away = self._temper(self._calibrate(blended))

        return {"p_home": float(p_home), "p_draw": float(p_draw),
                "p_away": float(p_away), "score_matrix": M,
                "p_xgb": [float(x) for x in p_xgb], "p_dc": [float(x) for x in p_dc]}


def _fit_calibrators(blended_val, y_val_arr, sample_weight=None):
    calibrators = []
    for c in range(3):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(blended_val[:, c], (y_val_arr == c).astype(float), sample_weight=sample_weight)
        calibrators.append(iso)
    return calibrators


def _apply_calibrators(calibrators, blended_val):
    calibrated = np.array([
        [calibrators[c].predict([blended_val[i, c]])[0] for c in range(3)]
        for i in range(len(blended_val))
    ])
    return _normalize_rows(calibrated)


def build(dataset, long, final_elo, asof, recency_halflife_days=365):
    """Build the ensemble Predictor.

    `recency_halflife_days` lets recently-played matches (this tournament's
    own results, as they land) count more toward the blend weight `alpha` and
    the isotonic calibration than older validation-split history — the
    "learn from what just happened" step of the closed prediction loop. Two
    guardrails keep this from letting a handful of new matches swing the
    model: a weight floor (0.10, so 2023-2025 history never vanishes) and
    shrinkage of the final alpha toward the unweighted optimum, proportional
    to how few recent matches there are (`n_recent / (n_recent + 20)`). Pass
    `recency_halflife_days=None` to reproduce the original unweighted
    behavior exactly (used for A/B comparison and as a regression check).
    """
    asof = pd.Timestamp(asof)
    train, val = split_by_date(dataset, TRAIN_START, VAL_START, asof)
    model, X_val, y_val = train_model(train, val)
    params = model_goals.fit(long, asof, halflife_days=180)

    p_xgb_val = model.predict_proba(X_val)
    p_dc_val = _dc_val_probs(params, val)
    y_val_arr = y_val.values if hasattr(y_val, "values") else np.asarray(y_val)

    def neg_ll(alpha, weight=None):
        blended = _normalize_rows(alpha * p_xgb_val + (1 - alpha) * p_dc_val)
        return log_loss(y_val_arr, blended, labels=[0, 1, 2], sample_weight=weight)

    alpha_base = float(minimize_scalar(neg_ll, bounds=(0.0, 1.0), method="bounded").x)

    if recency_halflife_days is None:
        w, n_recent, shrink, alpha_recency, alpha = None, 0, 0.0, alpha_base, alpha_base
    else:
        age_days = (asof - val["date"]).dt.days.to_numpy()
        w = np.clip(0.5 ** (age_days / recency_halflife_days), 0.10, None)
        n_recent = int(np.sum(age_days <= recency_halflife_days))
        alpha_recency = float(minimize_scalar(
            lambda a: neg_ll(a, weight=w), bounds=(0.0, 1.0), method="bounded"
        ).x)
        shrink = n_recent / (n_recent + 20.0)
        alpha = alpha_base + shrink * (alpha_recency - alpha_base)

    blended_val = _normalize_rows(alpha * p_xgb_val + (1 - alpha) * p_dc_val)
    calibrators = _fit_calibrators(blended_val, y_val_arr, sample_weight=w)

    # unweighted alpha_base + base calibration, kept only as an A/B diagnostic
    # against the recency-weighted predictor actually used for predictions.
    blended_val_base = _normalize_rows(alpha_base * p_xgb_val + (1 - alpha_base) * p_dc_val)
    calibrators_base = _fit_calibrators(blended_val_base, y_val_arr)

    predictor = Predictor(model, long, final_elo, params, alpha, calibrators, asof)
    predictor.val_log_loss_blended = float(neg_ll(alpha))

    calibrated_val = _apply_calibrators(calibrators, blended_val)
    predictor.val_log_loss_calibrated = float(log_loss(y_val_arr, calibrated_val, labels=[0, 1, 2]))
    predictor.val_log_loss_xgb_only = float(log_loss(y_val_arr, p_xgb_val, labels=[0, 1, 2]))

    calibrated_val_base = _apply_calibrators(calibrators_base, blended_val_base)
    predictor.val_log_loss_calibrated_base = float(log_loss(y_val_arr, calibrated_val_base, labels=[0, 1, 2]))

    predictor.alpha = alpha
    predictor.alpha_base = alpha_base
    predictor.alpha_recency = alpha_recency
    predictor.shrink = shrink
    predictor.n_recent_val = n_recent

    return predictor


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    from wcpred.data import load_results, per_team_long, tournament_today
    from wcpred.features import build_dataset

    print("Loading results + building dataset ...")
    results = load_results()
    dataset, final_elo = build_dataset(results)
    long = per_team_long(results)

    asof = tournament_today()
    predictor = build(dataset, long, final_elo, asof)

    print(f"\nBlend weight alpha (XGBoost share) : {predictor.alpha:.3f}")
    print(f"  alpha_base (unweighted)          : {predictor.alpha_base:.3f}")
    print(f"  alpha_recency (fully weighted)   : {predictor.alpha_recency:.3f}")
    print(f"  shrink toward alpha_base         : {predictor.shrink:.3f}  (n_recent_val={predictor.n_recent_val})")
    print(f"Validation log-loss, XGBoost only   : {predictor.val_log_loss_xgb_only:.3f}")
    print(f"Validation log-loss, blended (raw)  : {predictor.val_log_loss_blended:.3f}")
    print(f"Validation log-loss, blended+calib  : {predictor.val_log_loss_calibrated:.3f}")
    print(f"  ...vs unweighted (old) calib      : {predictor.val_log_loss_calibrated_base:.3f}")

    home, away = "Spain", "Saudi Arabia"
    out = predictor.predict(home, away, neutral=True, weight=4)
    total = out["p_home"] + out["p_draw"] + out["p_away"]

    print(f"\n{home} vs {away} (as of {asof.date()}, neutral group match):")
    print(f"  {home:<14} win : {out['p_home'] * 100:5.1f}%")
    print(f"  {'Draw':<14}     : {out['p_draw'] * 100:5.1f}%")
    print(f"  {away:<14} win : {out['p_away'] * 100:5.1f}%")
    print(f"  sum = {total:.4f}")
    print(f"  score_matrix shape = {out['score_matrix'].shape}, sum = {out['score_matrix'].sum():.6f}")
    print("  (real result: Spain 4-0 Saudi Arabia, 2026-06-21 group stage)")
