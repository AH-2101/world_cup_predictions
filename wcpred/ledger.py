"""wcpred.ledger — append-only prediction ledger + scoring against real
results, the seed of the closed learning loop.

Every prediction (from the CLI or the server) is logged to `ledger/
predictions.csv` at the moment it's made, tagged with whether the real result
was already known then (`result_known_at_log`). Once results land, `score()`
joins the ledger back against `wcpred.data.load_results()` and computes
per-prediction correctness/log-loss/Brier; `report_card()` summarizes that
into headline numbers (vs a no-skill baseline and, where priced, vs
Polymarket) plus a per-match trend. Only honest rows (result unknown at the
time of logging) count toward the headline — a prediction made after the
result was already known would make the model look artificially good.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from wcpred.fixtures import resolve_slots_for_date
from wcpred.model_wdl import MATCH_NEUTRAL, MATCH_WEIGHT

LEDGER_DIR = "ledger"
LEDGER_PATH = os.path.join(LEDGER_DIR, "predictions.csv")

COLUMNS = [
    "logged_at", "match_number", "match_date", "home", "away", "round",
    "asof", "alpha", "p_home", "p_draw", "p_away", "pick",
    "mkt_home", "mkt_draw", "mkt_away",
    # per-model component probabilities (used by wcpred.feedback to learn which
    # of XGBoost / Dixon-Coles has been more right); blank for pre-migration rows
    "p_xgb_home", "p_xgb_draw", "p_xgb_away", "p_dc_home", "p_dc_draw", "p_dc_away",
    "result_known_at_log", "source",
]

_MATCH_WINDOW_DAYS = 1  # fixtures.csv / results.csv dates can differ by a day


# ── writing ──────────────────────────────────────────────────────────────────
def load_ledger():
    if not os.path.exists(LEDGER_PATH):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(LEDGER_PATH)
    # reindex so callers always see every current column, even when reading an
    # older ledger written before newer columns (e.g. the p_xgb_*/p_dc_*
    # components) were added — missing ones come back as NaN.
    df = df.reindex(columns=COLUMNS)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df["logged_at"] = pd.to_datetime(df["logged_at"])
    return df


def _result_known(results, home, away, match_date):
    lo, hi = match_date - pd.Timedelta(days=_MATCH_WINDOW_DAYS), match_date + pd.Timedelta(days=_MATCH_WINDOW_DAYS)
    hit = results[
        (results["date"] >= lo) & (results["date"] <= hi)
        & (((results["home_team"] == home) & (results["away_team"] == away))
           | ((results["home_team"] == away) & (results["away_team"] == home)))
    ]
    return not hit.empty


def log_prediction(m, pred, predictor, results, source, market_probs=None):
    """Append one ledger row for a single prediction. `m` is a fixture dict
    (as returned by find_fixture / resolve_slots_for_date / find_bracket_match
    — home/away plus either "match_number" or "match", and "group" or
    "round"). `pred` is predictor.predict(...)'s output dict.

    Writes by rewriting the whole file (it's tiny — dozens of rows), which
    keeps append-time schema fragility out of the picture and transparently
    migrates an older ledger to any newly-added COLUMNS (old rows get blanks)."""
    os.makedirs(LEDGER_DIR, exist_ok=True)
    match_date = pd.Timestamp(m["date"])
    p = [pred["p_home"], pred["p_draw"], pred["p_away"]]
    pick = ["home", "draw", "away"][int(np.argmax(p))]
    market_probs = market_probs or {}
    p_xgb = pred.get("p_xgb") or ["", "", ""]
    p_dc = pred.get("p_dc") or ["", "", ""]
    row = {
        "logged_at": pd.Timestamp.now(),
        "match_number": m.get("match_number", m.get("match", "")),
        "match_date": match_date,
        "home": m["home"], "away": m["away"],
        "round": m.get("group", m.get("round", "")),
        "asof": predictor.asof, "alpha": predictor.alpha,
        "p_home": pred["p_home"], "p_draw": pred["p_draw"], "p_away": pred["p_away"],
        "pick": pick,
        "mkt_home": market_probs.get("p_home", ""), "mkt_draw": market_probs.get("p_draw", ""),
        "mkt_away": market_probs.get("p_away", ""),
        "p_xgb_home": p_xgb[0], "p_xgb_draw": p_xgb[1], "p_xgb_away": p_xgb[2],
        "p_dc_home": p_dc[0], "p_dc_draw": p_dc[1], "p_dc_away": p_dc[2],
        "result_known_at_log": _result_known(results, m["home"], m["away"], match_date),
        "source": source,
    }
    existing = load_ledger()
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    combined.reindex(columns=COLUMNS).to_csv(LEDGER_PATH, index=False)
    return row


def log_upcoming(predictor, results, days_ahead=3, source="auto"):
    """Pre-register predictions for every resolvable fixture over the next
    `days_ahead` days, so the loop scores predictions made BEFORE the result
    was known. Idempotent per calendar day: won't double-log the same
    (source, home, away, match_date) on the same day."""
    existing = load_ledger()
    today = pd.Timestamp.today().normalize()
    already = set()
    if not existing.empty:
        todays = existing[(existing["source"] == source)
                           & (existing["logged_at"].dt.normalize() == today)]
        already = set(zip(todays["home"], todays["away"], todays["match_date"]))

    n_new = 0
    for offset in range(days_ahead + 1):
        date_str = (today + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        for m in resolve_slots_for_date(results, date_str):
            key = (m["home"], m["away"], pd.Timestamp(m["date"]))
            if key in already:
                continue
            pred = predictor.predict(m["home"], m["away"], MATCH_NEUTRAL, MATCH_WEIGHT)
            log_prediction(m, pred, predictor, results, source)
            already.add(key)
            n_new += 1
    return n_new


# ── scoring ──────────────────────────────────────────────────────────────────
def _label_for(results, home, away, match_date):
    """0/1/2 (home/draw/away) outcome for `home` vs `away`, resolved against
    whichever orientation results.csv actually recorded, or None if unplayed
    / not found."""
    lo, hi = match_date - pd.Timedelta(days=_MATCH_WINDOW_DAYS), match_date + pd.Timedelta(days=_MATCH_WINDOW_DAYS)
    hit = results[
        (results["date"] >= lo) & (results["date"] <= hi)
        & (((results["home_team"] == home) & (results["away_team"] == away))
           | ((results["home_team"] == away) & (results["away_team"] == home)))
    ]
    if hit.empty:
        return None
    row = hit.iloc[(hit["date"] - match_date).abs().argsort().iloc[0]]
    hs, as_ = row["home_score"], row["away_score"]
    swapped = row["home_team"] == away
    if swapped:
        hs, as_ = as_, hs
    if hs > as_:
        return 0
    if hs == as_:
        return 1
    return 2


def score(results):
    """Ledger rows joined against real results: adds y/correct/log_loss/brier
    columns (NaN/None where the match hasn't been played yet)."""
    df = load_ledger()
    if df.empty:
        return df.assign(y=[], correct=[], log_loss=[], brier=[], scored=[])

    ys, correct, lls, briers, scored = [], [], [], [], []
    for row in df.itertuples():
        y = _label_for(results, row.home, row.away, row.match_date)
        p = np.array([row.p_home, row.p_draw, row.p_away])
        if y is None:
            ys.append(None); correct.append(None); lls.append(None); briers.append(None); scored.append(False)
            continue
        onehot = np.eye(3)[y]
        ys.append(y)
        correct.append(bool(int(np.argmax(p)) == y))
        lls.append(float(-np.log(max(p[y], 1e-9))))
        briers.append(float(np.sum((p - onehot) ** 2)))
        scored.append(True)

    df["y"] = ys
    df["correct"] = correct
    df["log_loss"] = lls
    df["brier"] = briers
    df["scored"] = scored
    return df


def report_card(results):
    df = score(results)
    n_logged = len(df)
    if df.empty:
        return {"n_logged": 0, "n_scored": 0, "accuracy": None, "log_loss": None, "brier": None,
                "baseline_log_loss": None, "baseline_brier": None,
                "market_log_loss": None, "market_n": 0, "per_match": []}

    honest_scored = df[df["scored"] & (~df["result_known_at_log"])].copy()
    # headline uses the LATEST honest prediction per match
    honest_scored = honest_scored.sort_values("logged_at").drop_duplicates(
        subset=["home", "away", "match_date"], keep="last"
    ).sort_values("match_date")

    n_scored = len(honest_scored)
    if n_scored == 0:
        return {"n_logged": n_logged, "n_scored": 0, "accuracy": None, "log_loss": None, "brier": None,
                "baseline_log_loss": None, "baseline_brier": None,
                "market_log_loss": None, "market_n": 0, "per_match": []}

    y = honest_scored["y"].astype(int).to_numpy()
    probs = honest_scored[["p_home", "p_draw", "p_away"]].to_numpy()
    freq = np.bincount(y, minlength=3) / len(y)
    base_probs = np.tile(freq, (len(y), 1))

    market_rows = honest_scored[honest_scored["mkt_home"].notna() & (honest_scored["mkt_home"] != "")]
    market_n = len(market_rows)
    market_log_loss = None
    if market_n:
        mkt_probs = market_rows[["mkt_home", "mkt_draw", "mkt_away"]].astype(float).to_numpy()
        market_log_loss = float(log_loss(market_rows["y"].astype(int), mkt_probs, labels=[0, 1, 2]))

    per_match = []
    correct_running = 0
    for i, row in enumerate(honest_scored.itertuples(), start=1):
        correct_running += int(row.correct)
        pick_prob = {"home": row.p_home, "draw": row.p_draw, "away": row.p_away}[row.pick]
        per_match.append({
            "date": row.match_date.strftime("%Y-%m-%d"), "home": row.home, "away": row.away,
            "pick": row.pick, "pick_prob": round(float(pick_prob), 4),
            "y": int(row.y), "correct": bool(row.correct),
            "log_loss": round(float(row.log_loss), 4),
            "cum_accuracy": round(correct_running / i, 4),
        })

    return {
        "n_logged": n_logged, "n_scored": n_scored,
        "accuracy": float(np.mean(y == probs.argmax(axis=1))),
        "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((probs - np.eye(3)[y]) ** 2, axis=1))),
        "baseline_log_loss": float(log_loss(y, base_probs, labels=[0, 1, 2])),
        "baseline_brier": float(np.mean(np.sum((base_probs - np.eye(3)[y]) ** 2, axis=1))),
        "market_log_loss": market_log_loss, "market_n": market_n,
        "per_match": per_match,
    }


if __name__ == "__main__":
    from wcpred.data import load_results

    results = load_results()
    card = report_card(results)
    print(f"Logged predictions   : {card['n_logged']}")
    print(f"Scored (honest)       : {card['n_scored']}")
    if card["n_scored"]:
        print(f"Accuracy               : {card['accuracy']:.3f}")
        print(f"Log-loss (vs baseline) : {card['log_loss']:.3f} vs {card['baseline_log_loss']:.3f}")
        print(f"Brier (vs baseline)    : {card['brier']:.3f} vs {card['baseline_brier']:.3f}")
        if card["market_n"]:
            print(f"Log-loss vs market ({card['market_n']} priced): "
                  f"{card['log_loss']:.3f} vs {card['market_log_loss']:.3f}")
        print("\nPer-match:")
        for r in card["per_match"]:
            mark = "Y" if r["correct"] else "n"
            print(f"  [{mark}] {r['date']}  {r['home']:<20} v {r['away']:<20}  "
                  f"pick={r['pick']:<5} ({r['pick_prob']*100:5.1f}%)  log_loss={r['log_loss']:.3f}")
