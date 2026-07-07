"""wcpred.feedback — turn the prediction ledger's track record into a small,
regularized adjustment on future predictions. This is the "learn from whether
it was right" step, done on ACCUMULATED evidence rather than single matches.

Two adjustments, each fit on the ledger's scored honest predictions and each
shrunk toward a no-op so a handful of matches can't swing the model:

  temperature T  — a single scalar softening/sharpening confidence
                   (p_i^(1/T), renormalized). Fit to minimize log-loss over
                   the scored matches; if the model has been confidently wrong
                   T rises (softer), if underconfident T falls (sharper).
                   Works on the probabilities already stored in the ledger, so
                   it's active as soon as any predictions are scored.

  blend alpha    — which of XGBoost / Dixon-Coles has actually been more right
                   this tournament. Refit the blend weight on the scored
                   matches' per-model component probabilities (stored in the
                   ledger going forward), shrunk toward the model's base alpha.
                   No change until enough scored rows carry components.

Applied for LIVE predictions only (cli match/today, server, dashboard). Never
used inside `ensemble.build` / the backtest — using scored predictions that
postdate a match to score that same match would be leakage. As a guard we only
ever use ledger rows with `match_date < predictor.asof`.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from wcpred import ledger

EPS = 1e-9
_COMP_COLS = ["p_xgb_home", "p_xgb_draw", "p_xgb_away", "p_dc_home", "p_dc_draw", "p_dc_away"]


def _temper(probs, T):
    adj = np.clip(probs, EPS, None) ** (1.0 / T)
    return adj / adj.sum(axis=1, keepdims=True)


def fit(results, predictor, min_matches=3, temp_k=10.0, alpha_k=15.0):
    """Fit the tournament adjustment from the ledger. Returns a dict of
    {temperature, alpha_effective, n_used, base_log_loss, adj_log_loss, ...}
    or a no-op adjustment (temperature=1.0, alpha_effective=None) when there
    isn't enough scored history yet."""
    noop = {"temperature": 1.0, "alpha_effective": None, "n_used": 0,
            "base_log_loss": None, "adj_log_loss": None,
            "alpha_base": predictor.alpha, "alpha_tourn": None}

    scored = ledger.score(results)
    if scored.empty:
        return noop
    asof = predictor.asof
    rows = scored[scored["scored"] & (~scored["result_known_at_log"])
                  & (scored["match_date"] < asof)].copy()
    # one row per match (latest prediction), matching the report card
    rows = rows.sort_values("logged_at").drop_duplicates(
        subset=["home", "away", "match_date"], keep="last")
    if len(rows) < min_matches:
        return noop

    y = rows["y"].astype(int).to_numpy()
    probs = rows[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    base_ll = float(log_loss(y, probs, labels=[0, 1, 2]))

    # ── temperature ──────────────────────────────────────────────────────
    def temp_ll(T):
        return log_loss(y, _temper(probs, T), labels=[0, 1, 2])

    T_opt = float(minimize_scalar(temp_ll, bounds=(0.3, 4.0), method="bounded").x)
    shrink_t = len(rows) / (len(rows) + temp_k)
    T_eff = 1.0 + shrink_t * (T_opt - 1.0)
    adj_ll = float(log_loss(y, _temper(probs, T_eff), labels=[0, 1, 2]))

    # ── tournament blend (only if component probs are present) ───────────
    alpha_tourn = None
    alpha_eff = None
    comp = rows.copy()
    for col in _COMP_COLS:
        comp[col] = pd.to_numeric(comp[col], errors="coerce")
    comp = comp.dropna(subset=_COMP_COLS)
    if len(comp) >= min_matches:
        yc = comp["y"].astype(int).to_numpy()
        p_xgb = comp[["p_xgb_home", "p_xgb_draw", "p_xgb_away"]].to_numpy(dtype=float)
        p_dc = comp[["p_dc_home", "p_dc_draw", "p_dc_away"]].to_numpy(dtype=float)

        def blend_ll(a):
            b = a * p_xgb + (1 - a) * p_dc
            b = b / b.sum(axis=1, keepdims=True)
            return log_loss(yc, b, labels=[0, 1, 2])

        alpha_tourn = float(minimize_scalar(blend_ll, bounds=(0.0, 1.0), method="bounded").x)
        shrink_a = len(comp) / (len(comp) + alpha_k)
        alpha_eff = float(predictor.alpha + shrink_a * (alpha_tourn - predictor.alpha))

    return {"temperature": T_eff, "alpha_effective": alpha_eff, "n_used": int(len(rows)),
            "base_log_loss": base_ll, "adj_log_loss": adj_ll,
            "alpha_base": predictor.alpha, "alpha_tourn": alpha_tourn,
            "T_opt": T_opt}


def apply(predictor, results, **kwargs):
    """Fit the adjustment and attach it to `predictor` (in place). Returns the
    adjustment dict."""
    adj = fit(results, predictor, **kwargs)
    predictor.tournament_temperature = adj["temperature"]
    predictor.alpha_effective = adj["alpha_effective"]
    predictor.feedback_info = adj
    return adj


def summary_line(adj):
    """One-line human-readable summary for CLI output."""
    if not adj or adj["n_used"] == 0:
        return "Feedback: no scored predictions yet -- model unadjusted."
    parts = [f"Feedback from {adj['n_used']} scored match(es): temperature T={adj['temperature']:.3f}"]
    if adj["temperature"] > 1.001:
        parts.append("(softening overconfident picks)")
    elif adj["temperature"] < 0.999:
        parts.append("(sharpening underconfident picks)")
    if adj["alpha_effective"] is not None:
        parts.append(f"| blend alpha {adj['alpha_base']:.3f} -> {adj['alpha_effective']:.3f}")
    if adj["base_log_loss"] is not None:
        parts.append(f"| log-loss on those matches {adj['base_log_loss']:.3f} -> {adj['adj_log_loss']:.3f}")
    return " ".join(parts)
