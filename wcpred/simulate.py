"""wcpred.simulate — Monte Carlo tournament simulator for the 2026 knockout bracket.

It's 2026-07-03: the group stage is finished, so there's no group-stage
simulation to do -- only the knockout bracket (fixtures.csv Match 73-104)
remains. `load_state` parses that bracket (see `wcpred.fixtures.parse_bracket`);
`run` plays it out `n` times against a `predictor` (anything shaped like
`wcpred.ensemble.Predictor`: `.predict(home, away, neutral, weight) ->
{"p_home","p_draw","p_away","score_matrix"}`) and tallies, per still-alive
team, the fraction of simulations in which it reaches each remaining round.

Already-played matches (fixed historical fact) are never simulated -- their
recorded winner is reused every time. Only genuinely pending matches draw a
result from the predictor. A regulation draw is broken with a coin flip
skewed toward whichever side the predictor favored (representing extra time
+ penalties without a separate ET/PK model).
"""

import sys

import numpy as np
import pandas as pd

from wcpred.data import load_results
from wcpred.fixtures import FIXTURES_PATH, parse_bracket

MATCH_WEIGHT = 4      # FIFA World Cup tournament_weight (see wcpred.model_wdl)
NEUTRAL = 1           # remaining 2026 knockout matches are effectively neutral-site
ET_PK_SKEW = 0.55     # extra-time/penalty coin-flip skew toward the run-of-play favorite
N_SIMS = 20000

# round played in a slot -> the milestone its WINNER earns (i.e. which round
# they advance *into*)
MILESTONE_FOR_ROUND = {
    "R32": "p_r16", "R16": "p_qf", "QF": "p_sf", "SF": "p_final", "F": "p_champion",
}
MILESTONE_COLS = ["p_r16", "p_qf", "p_sf", "p_final", "p_champion"]


def load_state(results, fixtures_path=FIXTURES_PATH):
    """Thin wrapper around `wcpred.fixtures.parse_bracket`."""
    return parse_bracket(fixtures_path, results)


def _alive_teams(bracket):
    """Teams still mathematically alive: every R32 (Match 73-88) participant
    whose match isn't decided yet, plus every already-decided R32 winner.
    R32 losers are excluded from the output entirely (as opposed to shown
    with all-zero rows) -- they can no longer reach any tracked milestone."""
    alive = set()
    for slot in bracket:
        if slot["round"] != "R32":
            continue
        if slot["played"]:
            alive.add(slot["winner"])
        else:
            alive.add(slot["resolved_home"])
            alive.add(slot["resolved_away"])
    return alive


def _resolve_side(slots, sim, match_number, side):
    """Home/away team for `match_number` in the simulation-in-progress `sim`:
    concrete if already resolved on the bracket, else looked up from an
    earlier match's simulated outcome (`sim`), which is always available
    because match numbers are processed in increasing order and every
    reference in this bracket points to a strictly lower match number."""
    slot = slots[match_number]
    resolved = slot["resolved_home"] if side == "home" else slot["resolved_away"]
    if resolved is not None:
        return resolved
    kind, ref = slot["home_source"] if side == "home" else slot["away_source"]
    if kind == "team":
        return ref
    outcome = sim[ref]
    return outcome["winner"] if kind == "winner_of" else outcome["loser"]


def _play(predictor, home, away, rng, cache):
    """Sample a W/D/L outcome for `home` vs `away` and return the winner.
    Predictions are cached per (home, away) pair within a `run()` call --
    many bracket slots (any match whose two teams are already fixed, win or
    lose, e.g. an already-known-but-unplayed R16 pairing) have the exact same
    matchup in every single simulation, so this avoids re-querying the
    (potentially expensive, XGBoost + Dixon-Coles) predictor tens of
    thousands of times for an identical input."""
    key = (home, away)
    pred = cache.get(key)
    if pred is None:
        pred = predictor.predict(home, away, neutral=NEUTRAL, weight=MATCH_WEIGHT)
        cache[key] = pred

    p = np.array([pred["p_home"], pred["p_draw"], pred["p_away"]], dtype=float)
    p = p / p.sum()
    outcome = rng.choice(3, p=p)  # 0=home, 1=draw, 2=away

    if outcome == 1:  # regulation draw -> extra time / penalties
        favorite = "home" if pred["p_home"] >= pred["p_away"] else "away"
        underdog = "away" if favorite == "home" else "home"
        side = favorite if rng.random() < ET_PK_SKEW else underdog
    else:
        side = "home" if outcome == 0 else "away"

    return home if side == "home" else away


def run(bracket, predictor, n=20000, seed=42):
    """Monte Carlo the remaining knockout rounds `n` times.

    Returns a DataFrame[team, p_r16, p_qf, p_sf, p_final, p_champion], one
    row per still-alive team (see `_alive_teams`), sorted by p_champion
    descending. Across the *full* returned field, p_champion sums to ~1.0,
    since every simulated champion is necessarily one of these teams.
    """
    rng = np.random.default_rng(seed)
    slots = bracket.slots
    match_numbers = sorted(slots.keys())

    alive = _alive_teams(bracket)
    counts = {t: {c: 0 for c in MILESTONE_COLS} for t in alive}
    pred_cache = {}

    for _ in range(n):
        sim = {}
        for mno in match_numbers:
            slot = slots[mno]
            if slot["played"]:
                home, away, winner = slot["resolved_home"], slot["resolved_away"], slot["winner"]
            else:
                home = _resolve_side(slots, sim, mno, "home")
                away = _resolve_side(slots, sim, mno, "away")
                winner = _play(predictor, home, away, rng, pred_cache)
            loser = away if winner == home else home
            sim[mno] = {"home": home, "away": away, "winner": winner, "loser": loser}

            milestone = MILESTONE_FOR_ROUND.get(slot["round"])
            if milestone is not None and winner in counts:
                counts[winner][milestone] += 1

    rows = [{"team": t, **{c: cnt[c] / n for c in MILESTONE_COLS}} for t, cnt in counts.items()]
    df = pd.DataFrame(rows, columns=["team"] + MILESTONE_COLS)
    return df.sort_values("p_champion", ascending=False).reset_index(drop=True)


class _FallbackPredictor:
    """Minimal stand-in for `wcpred.ensemble.Predictor`, used only if
    `wcpred.ensemble` isn't importable yet. XGBoost W/D/L only
    (`model_wdl.predict_symmetric`), plus a naive independent-Poisson score
    matrix just to satisfy the interface (simulate.run never reads
    score_matrix, only p_home/p_draw/p_away)."""

    def __init__(self, results):
        from wcpred.data import per_team_long
        from wcpred.features import build_dataset
        from wcpred.model_wdl import TRAIN_START, VAL_START, split_by_date, train_model

        dataset, self.final_elo = build_dataset(results)
        self.long = per_team_long(results)
        self.asof = pd.Timestamp.today().normalize()
        train, val = split_by_date(dataset, TRAIN_START, VAL_START, self.asof)
        self.model, _, _ = train_model(train, val)

    def predict(self, home, away, neutral, weight):
        from wcpred.model_wdl import predict_symmetric

        p_home, p_draw, p_away = predict_symmetric(
            self.model, self.long, self.final_elo, home, away, self.asof, neutral, weight
        )
        goals = np.arange(11)
        lam = 1.6 * (0.5 + p_home) if p_home >= p_away else 1.1
        mu = 1.6 * (0.5 + p_away) if p_away > p_home else 1.1
        from scipy.stats import poisson
        M = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        M = M / M.sum()
        return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away, "score_matrix": M}


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    print("Loading live results feed ...")
    results = load_results()

    print("Parsing knockout bracket (Match 73-104) ...")
    bracket = load_state(results)

    used_fallback = False
    try:
        from wcpred.data import per_team_long
        from wcpred.ensemble import build as build_ensemble
        from wcpred.features import build_dataset

        print("Building ensemble predictor (XGBoost + Dixon-Coles, blended + calibrated) ...")
        dataset, final_elo = build_dataset(results)
        long = per_team_long(results)
        asof = pd.Timestamp.today().normalize()
        predictor = build_ensemble(dataset, long, final_elo, asof)
    except Exception as exc:
        used_fallback = True
        print(f"[simulate] wcpred.ensemble unavailable ({exc!r}) -- falling back to a "
              f"model_wdl-only stub predictor. Re-run once wcpred/ensemble.py lands to "
              f"get the real (calibrated) numbers.")
        predictor = _FallbackPredictor(results)

    print(f"Simulating {N_SIMS} tournament completions "
          f"({'FALLBACK predictor' if used_fallback else 'ensemble predictor'}) ...")
    table = run(bracket, predictor, n=N_SIMS, seed=42)

    label = "FALLBACK predictor (model_wdl only)" if used_fallback else "Ensemble predictor (XGBoost + Dixon-Coles)"
    print(f"\n=== Championship odds -- {label} ===")
    print(f"(top 15 of {len(table)} still mathematically alive teams)\n")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.head(15).to_string(index=False))

    total = table["p_champion"].sum()
    print(f"\nSum of p_champion across all {len(table)} alive teams: {total:.4f}  (expect ~1.0)")

    if used_fallback:
        print("\nNOTE: ran with the fallback predictor -- re-verify once wcpred/ensemble.py lands.")
