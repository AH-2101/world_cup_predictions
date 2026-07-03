"""wcpred.backtest — walk-forward proof that the ensemble Predictor beats a
no-skill baseline and (where a market exists) Polymarket, on REAL match
outcomes.

Match sample
------------
We evaluate every match tagged ``tournament_weight == 4`` (i.e. an actual
FIFA World Cup finals match — see ``wcpred.data.tournament_weight``) with
``date >= since``. That set is exactly "the 2022 World Cup + this
tournament's already-played 2026 matches", which is both the highest-stakes
subset of the data and, for 2026, the one with real Polymarket markets to
compare against. This is explicitly the "all World-Cup-tournament-weighted
matches since `since`" sampling option called out in the task spec, chosen
over a generic every-Nth-match sample of *all* football because it keeps the
evaluation squarely on the matches this project is actually about.

The VAL_START floor (a real, documented limitation)
----------------------------------------------------
``wcpred.ensemble.build`` (frozen, not owned by this module) calls
``wcpred.model_wdl.split_by_date(dataset, TRAIN_START, VAL_START, asof)`` with
the hardcoded ``VAL_START = "2023-01-01"``. It fits the XGBoost/Dixon-Coles
blend weight and the per-class isotonic calibrators on the
``[VAL_START, asof)`` slice — so ``asof`` must be *strictly after*
``VAL_START`` or that slice is empty and ``build()`` raises. But leakage
safety additionally requires ``asof <= match_date`` for every match we
evaluate against a given fit. Combined, those two constraints mean there is
NO leakage-safe ``asof`` for any match with ``date <= VAL_START`` — asof
would have to be both after 2023-01-01 (for a non-empty calibration slice)
and at/before the match's own (earlier) date, which is impossible.
Concretely, this rules out the 2022 World Cup (Nov-Dec 2022) entirely. We
detect and skip those matches rather than silently mis-scoring them or
crashing, and report the skip count. This is an inherited property of the
frozen wave-2 interface, not a bug in this file.

asof-bucketing scheme
----------------------
Re-fitting the full ensemble per match would be far too slow (~15s per
`ensemble.build` call). Instead we bucket evaluable matches by calendar month
of their date (`asof` = the 1st of that month) and fit the ensemble ONCE per
bucket, then score every match in the bucket against that single fit. Since
every match date in a "YYYY-MM" bucket is `>= the 1st of that month`, using
`asof = YYYY-MM-01` for the whole bucket never trains on a match's own (or
any later) data — the no-leakage invariant holds for every match in the
bucket, even though matches later in the month are scored by a slightly
"stale" (start-of-month) fit. In practice the WC-weighted matches since 2022
fall into only a handful of distinct months (the 2022 group+knockout window
and the 2026 group+R32 window), so this keeps runtime to a handful of
`ensemble.build` calls.

Metrics
-------
accuracy / log_loss (sklearn, 3-class) / Brier (standard multiclass form:
mean over matches of sum over classes of (p_c - onehot_c)^2). The no-skill
baseline reuses the exact convention already used by
`wcpred.model_wdl.evaluate()` — the class-frequency vector of the evaluated
outcomes, tiled as a constant prediction for every match.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from wcpred import market as _market
from wcpred.ensemble import build as _build_ensemble
from wcpred.model_wdl import VAL_START

VAL_START_TS = pd.Timestamp(VAL_START)


# ── match selection ──────────────────────────────────────────────────────────
def _select_matches(dataset, since):
    since_ts = pd.Timestamp(since)
    sub = dataset[(dataset["date"] >= since_ts) & (dataset["tournament_weight"] == 4)].copy()
    return sub.sort_values("date").reset_index(drop=True)


def _bucket_start(date):
    return date.to_period("M").to_timestamp()


# ── metrics ──────────────────────────────────────────────────────────────────
def _brier_multiclass(y_true, probs):
    onehot = np.eye(3)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _metrics(y_true, probs):
    if len(y_true) == 0:
        return {"accuracy": np.nan, "log_loss": np.nan, "brier": np.nan}
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(np.mean(pred == y_true)),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1, 2])),
        "brier": _brier_multiclass(y_true, probs),
    }


# ── walk-forward backtest ─────────────────────────────────────────────────────
def run(dataset, long, final_elo, since="2022-01-01"):
    """Walk-forward evaluate the ensemble Predictor on real FIFA World Cup
    matches (tournament_weight == 4) with date >= `since`, against a no-skill
    baseline and Polymarket (when a market exists for that fixture).

    Returns a DataFrame with one row per metric (accuracy/log_loss/brier) and
    columns:
      - ensemble                : metric over ALL evaluated matches
      - baseline                : no-skill (class-frequency) metric over the
                                  same matches
      - ensemble_market_subset  : ensemble's metric restricted to ONLY the
                                  matches that had a Polymarket price (for a
                                  fair apples-to-apples market comparison)
      - market                  : Polymarket's metric on that same subset
      - market_subset_n         : size of that subset (constant per row)

    Metadata (match counts) is stashed on `out.attrs` and also printed by the
    `__main__` block: n_matches, n_2026_matches, n_market_matches,
    n_skipped_pre_val.
    """
    candidates = _select_matches(dataset, since)
    if candidates.empty:
        raise ValueError(f"backtest.run: no FIFA World Cup matches found since {since}")

    candidates["bucket"] = candidates["date"].map(_bucket_start)

    skipped = candidates[candidates["bucket"] <= VAL_START_TS]
    evalset = candidates[candidates["bucket"] > VAL_START_TS].reset_index(drop=True)

    if len(skipped):
        print(
            f"[backtest] skipping {len(skipped)} match(es) at/before {VAL_START} "
            f"(e.g. the 2022 World Cup) — ensemble.build() needs asof strictly "
            f"after model_wdl.VAL_START={VAL_START!r} for a non-empty calibration "
            f"split, so no leakage-safe cutoff exists for matches at or before it."
        )
    if evalset.empty:
        raise ValueError(
            "backtest.run: no evaluable matches remain after the VAL_START floor "
            f"({VAL_START}) — try a `since` further in the future."
        )

    records = []
    for bucket_asof, bucket_df in evalset.groupby("bucket"):
        predictor = _build_ensemble(dataset, long, final_elo, asof=bucket_asof)
        for row in bucket_df.itertuples():
            probs = predictor.predict(
                row.home_team, row.away_team, bool(row.neutral), row.tournament_weight
            )
            p_ens = np.array([probs["p_home"], probs["p_draw"], probs["p_away"]])
            mkt = _market.match(row.home_team, row.away_team)
            rec = {
                "date": row.date,
                "home": row.home_team,
                "away": row.away_team,
                "y": int(row.label),
                "p_ens": p_ens,
                "is_2026": row.date >= pd.Timestamp("2026-01-01"),
            }
            if mkt is not None:
                rec["p_mkt"] = np.array([mkt["p_home"], mkt["p_draw"], mkt["p_away"]])
            records.append(rec)

    y_all = np.array([r["y"] for r in records])
    p_ens_all = np.vstack([r["p_ens"] for r in records])

    # no-skill baseline: class-frequency of the evaluated outcomes, same
    # convention as wcpred.model_wdl.evaluate()'s validation baseline.
    freq = np.bincount(y_all, minlength=3) / len(y_all)
    p_base_all = np.tile(freq, (len(y_all), 1))

    mkt_idx = [i for i, r in enumerate(records) if "p_mkt" in r]
    n_market = len(mkt_idx)
    if n_market:
        y_mkt = y_all[mkt_idx]
        p_ens_mkt = p_ens_all[mkt_idx]
        p_mkt = np.vstack([records[i]["p_mkt"] for i in mkt_idx])
    else:
        y_mkt = np.array([], dtype=int)
        p_ens_mkt = np.zeros((0, 3))
        p_mkt = np.zeros((0, 3))

    m_ens = _metrics(y_all, p_ens_all)
    m_base = _metrics(y_all, p_base_all)
    m_ens_sub = _metrics(y_mkt, p_ens_mkt)
    m_mkt = _metrics(y_mkt, p_mkt)

    n_2026 = int(sum(1 for r in records if r["is_2026"]))

    out = pd.DataFrame({
        "metric": ["accuracy", "log_loss", "brier"],
        "ensemble": [m_ens["accuracy"], m_ens["log_loss"], m_ens["brier"]],
        "baseline": [m_base["accuracy"], m_base["log_loss"], m_base["brier"]],
        "ensemble_market_subset": [m_ens_sub["accuracy"], m_ens_sub["log_loss"], m_ens_sub["brier"]],
        "market": [m_mkt["accuracy"], m_mkt["log_loss"], m_mkt["brier"]],
        "market_subset_n": [n_market, n_market, n_market],
    })
    out.attrs["n_matches"] = len(records)
    out.attrs["n_2026_matches"] = n_2026
    out.attrs["n_market_matches"] = n_market
    out.attrs["n_skipped_pre_val"] = int(len(skipped))
    return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    from wcpred.data import load_results, per_team_long
    from wcpred.features import build_dataset

    print("Loading results + building dataset ...")
    results = load_results()
    dataset, final_elo = build_dataset(results)
    long = per_team_long(results)

    print("Running walk-forward backtest (since=2022-01-01, monthly asof buckets) ...\n")
    table = run(dataset, long, final_elo, since="2022-01-01")

    print(table.round(4).to_string(index=False))

    print(f"\nTotal matches evaluated          : {table.attrs['n_matches']}")
    print(f"  of which 2026 tournament matches: {table.attrs['n_2026_matches']}")
    print(f"  of which had a Polymarket price : {table.attrs['n_market_matches']}")
    print(f"Matches skipped (pre-VAL_START)   : {table.attrs['n_skipped_pre_val']}")

    ll = table.set_index("metric")
    ll_ens, ll_base = ll.loc["log_loss", "ensemble"], ll.loc["log_loss", "baseline"]
    br_ens, br_base = ll.loc["brier", "ensemble"], ll.loc["brier", "baseline"]
    print(f"\nEnsemble vs no-skill baseline (log-loss) : {ll_ens:.4f} vs {ll_base:.4f} "
          f"({'ensemble wins' if ll_ens < ll_base else 'baseline wins'})")
    print(f"Ensemble vs no-skill baseline (Brier)    : {br_ens:.4f} vs {br_base:.4f} "
          f"({'ensemble wins' if br_ens < br_base else 'baseline wins'})")

    n_mkt = int(ll.loc["log_loss", "market_subset_n"])
    if n_mkt:
        ll_ens_sub, ll_mkt = ll.loc["log_loss", "ensemble_market_subset"], ll.loc["log_loss", "market"]
        verdict = "ensemble BEATS the market" if ll_ens_sub < ll_mkt else "ensemble LOSES to the market"
        print(f"\nOn the {n_mkt}-match Polymarket subset, log-loss: "
              f"ensemble {ll_ens_sub:.4f} vs market {ll_mkt:.4f} -> {verdict}.")
    else:
        print("\nNo matches in this sample had an available Polymarket price — no market comparison.")
