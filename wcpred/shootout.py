"""wcpred.shootout — is a penalty shootout really a coin flip?

Fits a single-parameter logistic P(home side wins the shootout) =
sigmoid(c * elo_diff / 400) on the martj42 shootouts.csv history (CC0),
joined against the feature dataset for pre-match Elo ratings. The
coefficient is shrunk by n/(n+200) and zeroed entirely if the shrunk model
can't beat a plain coin on log-loss — the football literature says shootout
outcomes are close to independent of team strength, so c=0 (a 50/50 flip)
is the expected and acceptable outcome. The extra-time half of the
knockout-draw model (see wcpred.simulate) is where the real signal lives.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

_SHRINK_K = 200.0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_shootout_edge(shootouts, dataset, before=None, verbose=True):
    """MLE of the Elo-edge coefficient `c` on historical shootouts. `dataset`
    is wcpred.features.build_dataset's frame (carries pre-match home_elo/
    away_elo per match). Only shootouts with `date < before` are used when
    given, preserving the nothing-trained-at/after-the-match convention."""
    s = shootouts
    if before is not None:
        s = s[s["date"] < pd.Timestamp(before)]
    joined = s.merge(
        dataset[["date", "home_team", "away_team", "home_elo", "away_elo"]],
        on=["date", "home_team", "away_team"], how="inner",
    ).dropna(subset=["winner", "home_elo", "away_elo"])
    if joined.empty:
        return 0.0

    y = (joined["winner"] == joined["home_team"]).to_numpy(dtype=float)
    x = ((joined["home_elo"] - joined["away_elo"]) / 400.0).to_numpy(dtype=float)
    n = len(y)

    def nll(c):
        p = np.clip(_sigmoid(c * x), 1e-9, 1 - 1e-9)
        return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    c_opt = float(minimize_scalar(nll, bounds=(-5.0, 5.0), method="bounded").x)
    c_eff = c_opt * n / (n + _SHRINK_K)

    coin_ll = np.log(2.0)
    model_ll = nll(c_eff)
    if model_ll >= coin_ll - 1e-4:
        if verbose:
            print(f"[shootout] n={n}: Elo edge doesn't beat a coin "
                  f"(log-loss {model_ll:.4f} vs {coin_ll:.4f}) — using 50/50.")
        return 0.0
    if verbose:
        print(f"[shootout] n={n}: fitted Elo-edge coefficient c={c_eff:.3f} "
              f"(raw {c_opt:.3f}), log-loss {model_ll:.4f} vs coin {coin_ll:.4f}.")
    return float(c_eff)


def fit_from_data(dataset, before=None):
    """Convenience wrapper for call sites: fetch shootouts.csv and fit the
    edge coefficient, degrading to 0.0 (a fair coin) on any failure so an
    offline environment never breaks the simulator."""
    try:
        from wcpred.data import load_shootouts
        return fit_shootout_edge(load_shootouts(), dataset, before=before)
    except Exception as exc:
        print(f"[shootout] shootout data unavailable ({exc}); using 50/50 shootouts.")
        return 0.0


def p_shootout_home(c, final_elo, home, away, elo_base=1500.0):
    """P(`home` side wins a shootout) under the fitted edge coefficient."""
    if not c or final_elo is None:
        return 0.5
    diff = final_elo.get(home, elo_base) - final_elo.get(away, elo_base)
    return float(_sigmoid(c * diff / 400.0))
