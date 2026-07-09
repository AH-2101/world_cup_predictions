"""wcpred.model_goals — Dixon-Coles bivariate Poisson goals model.

Predicts full scorelines (not just W/D/L) via time-decayed attack/defense team
strengths + a home-advantage term, with the Dixon-Coles (1997) low-score
correlation correction. This is what fixes the base XGBoost W/D/L model's
tendency to under-call draws: instead of classifying an outcome directly, we
model the two teams' goal counts and read W/D/L off the resulting scoreline
matrix.

Model
-----
    home_goals ~ Poisson(lambda),  lambda = exp(home_adv·is_true_home + attack[home] - defense[away])
    away_goals ~ Poisson(mu),      mu     = exp(attack[away] - defense[home])

where `is_true_home` is 0 for neutral-venue matches (most World Cup finals
games), so home advantage is neither learned from nor applied to them.

with the independence assumption relaxed for the four low-score cells via the
Dixon-Coles tau adjustment (see `_tau_scalar` / the vectorized version inside
`fit`). Parameters (attack_i, defense_i per team, home_adv, rho) are fit by
maximum likelihood (`scipy.optimize.minimize`, L-BFGS-B, analytic gradient),
with each match down-weighted by an exponential time-decay from `asof` and a
small L2 ridge on attack/defense for identifiability (adding a constant to
every attack_i and every defense_i leaves the likelihood unchanged, since only
attack_i - defense_j ever appears; the ridge penalty picks the minimum-norm
solution, which centers ratings at ~0 = league average).

`long` is expected to be exactly the shape produced by
`wcpred.data.per_team_long(results)`: two rows per historical match (one from
each team's point of view), concatenated home-block-first / away-block-second
with no reordering in between. That's the only place a `fit()`-only caller
can recover which side had home advantage, so `_pair_matches` below relies on
that row order (with a defensive, best-effort fallback if it doesn't hold).
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

RIDGE = 1e-3                 # L2 penalty on attack/defense for identifiability
WEIGHT_PRUNE_THRESHOLD = 1e-8  # drop matches whose time-decay weight is negligible
HOME_ADV_INIT = 0.25
RHO_INIT = -0.05

# Safety clip on the log-rate (home_adv + attack - defense) before exponentiating,
# both during fitting and at prediction time. Unbounded L-BFGS-B can occasionally
# take a bad step early in optimization that overflows np.exp(); once that happens
# the gradient becomes non-finite and the optimizer never recovers, "converging" to
# an unusable fit (observed: home_advantage=109, attack as extreme as -1064). A
# healthy fit's log-rate sits well within [-1, 2] (lambda ~ 0.4-7 goals), so this
# clip only ever activates in a diverging fit and is otherwise a no-op. It also
# guarantees score_matrix's Poisson pmf always has non-negligible mass within
# max_goals, so normalizing (M / M.sum()) can never divide by ~0 into NaN.
LOG_RATE_MIN, LOG_RATE_MAX = -6.0, 3.0   # lambda/mu in [~0.0025, ~20]
PARAM_BOUND = 6.0                          # hard ceiling on attack/defense/home_adv

_CACHE = {}  # (asof: pd.Timestamp, halflife_days: int) -> params dict


# ── match reconstruction ─────────────────────────────────────────────────────────
def _pair_matches(long):
    """Recover one row per real match (date, home, away, hg, ag, neutral) from
    the per_team_long two-rows-per-match shape. See module docstring for the
    ordering assumption this relies on."""
    long = long.reset_index(drop=True)
    if "neutral" not in long.columns:
        warnings.warn(
            "model_goals: `long` has no 'neutral' column; treating every match "
            "as a true home match (pre-neutral-awareness behavior).", RuntimeWarning,
        )
        long = long.assign(neutral=0)
    n = len(long)
    if n == 0:
        return pd.DataFrame(columns=["date", "home", "away", "hg", "ag", "neutral"])

    if n % 2 == 0:
        h = n // 2
        home_half = long.iloc[:h].reset_index(drop=True)
        away_half = long.iloc[h:].reset_index(drop=True)
        ok = (
            (home_half["team"].values == away_half["opp"].values)
            & (home_half["opp"].values == away_half["team"].values)
            & (home_half["gf"].values == away_half["ga"].values)
            & (home_half["ga"].values == away_half["gf"].values)
            & (pd.to_datetime(home_half["date"]).values == pd.to_datetime(away_half["date"]).values)
        )
        if len(ok) and ok.mean() > 0.99:
            return pd.DataFrame({
                "date": pd.to_datetime(home_half["date"].values),
                "home": home_half["team"].values,
                "away": home_half["opp"].values,
                "hg": home_half["gf"].values.astype(float),
                "ag": home_half["ga"].values.astype(float),
                "neutral": home_half["neutral"].values.astype(float),
            })

    # Fallback: `long` wasn't in the standard home-block/away-block order (e.g.
    # it was filtered/reordered before being passed in). Reconstruct matches by
    # pairing each team-row with its mirror row; "home" is then an arbitrary but
    # deterministic pick, so home-advantage is still estimable on average even
    # though individual-match home/away assignment isn't guaranteed correct.
    warnings.warn(
        "model_goals: could not positionally recover home/away order from `long`; "
        "falling back to a best-effort match reconstruction.", RuntimeWarning,
    )
    tmp = long.copy().reset_index(drop=True)
    tmp["_pair_key"] = [tuple(sorted((t, o))) for t, o in zip(tmp["team"], tmp["opp"])]
    tmp["_date_key"] = pd.to_datetime(tmp["date"]).values.astype("int64")
    tmp["_row"] = np.arange(len(tmp))
    rows = []
    for _, grp in tmp.groupby(["_date_key", "_pair_key"], sort=False):
        if len(grp) < 2:
            continue
        r0 = grp.sort_values("_row").iloc[0]
        rows.append({"date": r0["date"], "home": r0["team"], "away": r0["opp"],
                     "hg": float(r0["gf"]), "ag": float(r0["ga"]),
                     "neutral": float(r0["neutral"])})
    return pd.DataFrame(rows)


# ── fitting ──────────────────────────────────────────────────────────────────────
def fit(long, asof, halflife_days=180):
    """Fit Dixon-Coles attack/defense/home-advantage/rho on matches strictly
    before `asof`, with exponential time-decay weighting. Cached in memory per
    (asof, halflife_days)."""
    asof = pd.Timestamp(asof)
    cache_key = (asof, int(halflife_days))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    matches = _pair_matches(long)
    matches = matches[matches["date"] < asof].reset_index(drop=True)
    if matches.empty:
        raise ValueError(f"model_goals.fit: no matches strictly before asof={asof.date()}")

    days_before = (asof - matches["date"]).dt.days.values.astype(float)
    weight = 0.5 ** (days_before / float(halflife_days))
    keep = weight > WEIGHT_PRUNE_THRESHOLD
    matches = matches[keep].reset_index(drop=True)
    weight = weight[keep]

    teams = sorted(set(matches["home"]) | set(matches["away"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    h_idx = matches["home"].map(team_idx).values
    a_idx = matches["away"].map(team_idx).values
    hg = matches["hg"].values.astype(float)
    ag = matches["ag"].values.astype(float)
    # home advantage only applies where the nominal home side truly played at
    # home; neutral-site matches (most World Cup finals games) get none.
    home_ind = 1.0 - matches["neutral"].values.astype(float)

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    def unpack(x):
        return x[:n], x[n:2 * n], x[2 * n], x[2 * n + 1]

    def nll_and_grad(x):
        attack, defense, home_adv, rho = unpack(x)
        log_lam = np.clip(home_adv * home_ind + attack[h_idx] - defense[a_idx], LOG_RATE_MIN, LOG_RATE_MAX)
        log_mu = np.clip(attack[a_idx] - defense[h_idx], LOG_RATE_MIN, LOG_RATE_MAX)
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)

        loglik = weight * (hg * np.log(lam) - lam - gammaln(hg + 1)
                           + ag * np.log(mu) - mu - gammaln(ag + 1))

        tau = np.ones_like(lam)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        tau_safe = np.clip(tau, 1e-8, None)
        loglik = loglik + weight * np.log(tau_safe)

        reg = RIDGE * (np.sum(attack ** 2) + np.sum(defense ** 2))
        nll = -loglik.sum() + reg

        # d(log tau)/dlam, d(log tau)/dmu, d(log tau)/drho (zero outside the 4 cells)
        dlam_tau = np.zeros_like(lam)
        dmu_tau = np.zeros_like(mu)
        drho = np.zeros_like(lam)

        dlam_tau[m00] = -mu[m00] * rho / tau_safe[m00]
        dmu_tau[m00] = -lam[m00] * rho / tau_safe[m00]
        drho[m00] = -lam[m00] * mu[m00] / tau_safe[m00]

        dlam_tau[m01] = rho / tau_safe[m01]
        drho[m01] = lam[m01] / tau_safe[m01]

        dmu_tau[m10] = rho / tau_safe[m10]
        drho[m10] = mu[m10] / tau_safe[m10]

        drho[m11] = -1.0 / tau_safe[m11]

        d_attack_h = weight * ((hg - lam) + dlam_tau * lam)
        d_defense_a = weight * (-(hg - lam) - dlam_tau * lam)
        d_attack_a = weight * ((ag - mu) + dmu_tau * mu)
        d_defense_h = weight * (-(ag - mu) - dmu_tau * mu)
        d_home_adv = weight * ((hg - lam) + dlam_tau * lam) * home_ind
        d_rho = weight * drho

        g_attack = np.zeros(n)
        g_defense = np.zeros(n)
        np.add.at(g_attack, h_idx, d_attack_h)
        np.add.at(g_attack, a_idx, d_attack_a)
        np.add.at(g_defense, a_idx, d_defense_a)
        np.add.at(g_defense, h_idx, d_defense_h)

        g_attack = -g_attack + 2 * RIDGE * attack
        g_defense = -g_defense + 2 * RIDGE * defense
        g_home_adv = -d_home_adv.sum()
        g_rho = -d_rho.sum()

        grad = np.concatenate([g_attack, g_defense, [g_home_adv, g_rho]])
        return nll, grad

    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = HOME_ADV_INIT
    x0[2 * n + 1] = RHO_INIT
    bounds = [(-PARAM_BOUND, PARAM_BOUND)] * (2 * n + 1) + [(-0.9, 0.9)]

    # 300 iterations stops just short of L-BFGS-B's own convergence test on the
    # full ~12.5k-match fit (it needs ~450); the extra headroom costs <1s.
    res = minimize(nll_and_grad, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 800})

    attack, defense, home_adv, rho = unpack(res.x)
    params = {
        "teams": teams,
        "attack": dict(zip(teams, attack.tolist())),
        "defense": dict(zip(teams, defense.tolist())),
        "home_advantage": float(home_adv),
        "rho": float(rho),
        "avg_attack": float(np.mean(attack)) if n else 0.0,
        "avg_defense": float(np.mean(defense)) if n else 0.0,
        "asof": asof,
        "halflife_days": int(halflife_days),
        "n_matches": int(len(matches)),
        "converged": bool(res.success),
    }
    _CACHE[cache_key] = params
    return params


# ── prediction ───────────────────────────────────────────────────────────────────
def rates(params, home, away, neutral=False):
    """(lam, mu) expected-goal rates for home vs away. Home advantage is only
    applied when the match is genuinely at `home`'s ground (`neutral=False`);
    at a neutral venue the nominal home side gets no boost."""
    attack, defense = params["attack"], params["defense"]
    avg_a, avg_d = params["avg_attack"], params["avg_defense"]
    home_adv = 0.0 if neutral else params["home_advantage"]

    a_h = attack.get(home, avg_a)
    d_h = defense.get(home, avg_d)
    a_a = attack.get(away, avg_a)
    d_a = defense.get(away, avg_d)

    lam = float(np.exp(np.clip(home_adv + a_h - d_a, LOG_RATE_MIN, LOG_RATE_MAX)))
    mu = float(np.exp(np.clip(a_a - d_h, LOG_RATE_MIN, LOG_RATE_MAX)))
    return lam, mu


def matrix_from_rates(lam, mu, rho, max_goals=10):
    """Scoreline matrix from Poisson rates: outer product of pmfs, Dixon-Coles
    tau on the 4 low-score cells, clipped and normalized to sum to 1."""
    goals = np.arange(max_goals + 1)
    p_home_goals = poisson.pmf(goals, lam)
    p_away_goals = poisson.pmf(goals, mu)
    M = np.outer(p_home_goals, p_away_goals)

    tau = np.ones_like(M)
    tau[0, 0] = 1 - lam * mu * rho
    tau[0, 1] = 1 + lam * rho
    tau[1, 0] = 1 + mu * rho
    tau[1, 1] = 1 - rho
    M = M * tau
    M = np.clip(M, 0, None)
    M = M / M.sum()
    return M


def score_matrix(params, home, away, max_goals=10, neutral=False):
    """P(home scores i, away scores j) for i, j in 0..max_goals, Dixon-Coles
    tau-corrected on the 4 low-score cells, normalized to sum to 1."""
    lam, mu = rates(params, home, away, neutral=neutral)
    return matrix_from_rates(lam, mu, params["rho"], max_goals=max_goals)


def wdl_from_matrix(M):
    """(p_home, p_draw, p_away) from a score matrix M[i, j] = P(home i, away j)."""
    n = M.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    p_home = float(M[ii > jj].sum())
    p_draw = float(M[ii == jj].sum())
    p_away = float(M[ii < jj].sum())
    return p_home, p_draw, p_away


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    from wcpred.data import load_results, per_team_long, tournament_today

    print("Loading results + building per-team long format ...")
    results = load_results()
    long = per_team_long(results)

    asof = tournament_today()
    params = fit(long, asof, halflife_days=180)
    print(f"Fit Dixon-Coles as of {asof.date()}  "
          f"(teams={len(params['teams'])}, matches={params['n_matches']}, "
          f"home_adv={params['home_advantage']:.3f}, rho={params['rho']:.3f}, "
          f"converged={params['converged']})")

    home, away = "Portugal", "Spain"
    M = score_matrix(params, home, away)
    p_home, p_draw, p_away = wdl_from_matrix(M)

    print(f"\n{home} vs {away} scoreline matrix sum: {M.sum():.6f}")
    print(f"  {home:<10} win : {p_home * 100:5.1f}%")
    print(f"  {'Draw':<10}     : {p_draw * 100:5.1f}%")
    print(f"  {away:<10} win : {p_away * 100:5.1f}%")
