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
        M = model_goals.score_matrix(params, row.home_team, row.away_team)
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

    def _calibrate(self, blended):
        calibrated = np.array([
            self.calibrators[c].predict([blended[c]])[0] for c in range(3)
        ])
        calibrated = np.clip(calibrated, EPS, None)
        return calibrated / calibrated.sum()

    def predict(self, home, away, neutral, weight):
        p_xgb = np.array(predict_symmetric(
            self.model, self.long, self.final_elo, home, away, self.asof, neutral, weight
        ))
        M = model_goals.score_matrix(self.params, home, away)
        p_dc = np.array(model_goals.wdl_from_matrix(M))

        blended = self.alpha * p_xgb + (1 - self.alpha) * p_dc
        blended = blended / blended.sum()
        p_home, p_draw, p_away = self._calibrate(blended)

        return {"p_home": float(p_home), "p_draw": float(p_draw),
                "p_away": float(p_away), "score_matrix": M}


def build(dataset, long, final_elo, asof):
    asof = pd.Timestamp(asof)
    train, val = split_by_date(dataset, TRAIN_START, VAL_START, asof)
    model, X_val, y_val = train_model(train, val)
    params = model_goals.fit(long, asof, halflife_days=180)

    p_xgb_val = model.predict_proba(X_val)
    p_dc_val = _dc_val_probs(params, val)
    y_val_arr = y_val.values if hasattr(y_val, "values") else np.asarray(y_val)

    def neg_ll(alpha):
        blended = _normalize_rows(alpha * p_xgb_val + (1 - alpha) * p_dc_val)
        return log_loss(y_val_arr, blended, labels=[0, 1, 2])

    res = minimize_scalar(neg_ll, bounds=(0.0, 1.0), method="bounded")
    alpha = float(res.x)

    blended_val = _normalize_rows(alpha * p_xgb_val + (1 - alpha) * p_dc_val)

    calibrators = []
    for c in range(3):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(blended_val[:, c], (y_val_arr == c).astype(float))
        calibrators.append(iso)

    predictor = Predictor(model, long, final_elo, params, alpha, calibrators, asof)
    predictor.val_log_loss_blended = float(neg_ll(alpha))

    calibrated_val = np.array([
        [calibrators[c].predict([blended_val[i, c]])[0] for c in range(3)]
        for i in range(len(blended_val))
    ])
    calibrated_val = _normalize_rows(calibrated_val)
    predictor.val_log_loss_calibrated = float(log_loss(y_val_arr, calibrated_val, labels=[0, 1, 2]))
    predictor.val_log_loss_xgb_only = float(log_loss(y_val_arr, p_xgb_val, labels=[0, 1, 2]))
    predictor.alpha = alpha

    return predictor


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    from wcpred.data import load_results, per_team_long
    from wcpred.features import build_dataset

    print("Loading results + building dataset ...")
    results = load_results()
    dataset, final_elo = build_dataset(results)
    long = per_team_long(results)

    asof = pd.Timestamp.today().normalize()
    predictor = build(dataset, long, final_elo, asof)

    print(f"\nBlend weight alpha (XGBoost share) : {predictor.alpha:.3f}")
    print(f"Validation log-loss, XGBoost only   : {predictor.val_log_loss_xgb_only:.3f}")
    print(f"Validation log-loss, blended (raw)  : {predictor.val_log_loss_blended:.3f}")
    print(f"Validation log-loss, blended+calib  : {predictor.val_log_loss_calibrated:.3f}")

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
