# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is
A 2026 FIFA World Cup match predictor. It began as a single file (`predict_today.py`) and is now a small,
forkable `wcpred/` package: an XGBoost + Dixon-Coles ensemble that predicts Win/Draw/Loss + full scorelines,
Monte-Carlo's the knockout bracket for championship odds, compares itself to Polymarket, and **learns from its
own track record** — every prediction is logged and, once results land, scored and fed back into future
predictions. It ships both a CLI and an interactive web front end. Still no notebooks, no heavy framework.

## How to run
```
pip install -r requirements.txt

# interactive web front end (recommended) — pick teams, live bracket, self-scoring report card
python -m wcpred.cli serve                        # http://localhost:8000, auto-refreshes every 30 min

# CLI
python -m wcpred.cli match "Portugal" "Spain"     # single match (order doesn't matter)
python predict_today.py "Portugal" "Spain"        # same thing, back-compat shim
python -m wcpred.cli today                         # every resolvable fixture today
python -m wcpred.cli sim --n 20000                 # championship odds (Monte Carlo bracket)
python -m wcpred.cli bracket                        # resolved bracket + survival probabilities
python -m wcpred.cli edge                           # model vs. Polymarket disagreement
python -m wcpred.cli backtest                       # walk-forward accuracy/log-loss/Brier
python -m wcpred.cli score                           # score the prediction ledger vs real results
python -m wcpred.cli dashboard                        # write a self-contained static dashboard.html
```
Fixtures (date/group/stadium) are read from `data_cache/fixtures.csv`. The ~5MB historical results file
auto-downloads to `data_cache/results.csv` and refreshes when >6h stale (gitignored).

## Architecture
Flat `wcpred/` package. Reuse these — don't reinvent them:
- **`data.py`** — `fetch_results`/`load_results` pull the open `martj42/international_results` CSV (6h refresh
  TTL, stale-cache-beats-failure); `load_shootouts` pulls its `shootouts.csv` the same way;
  `NAME_MAP`/`FIXTURE_NAME_MAP` normalize spellings; `per_team_long` (carries the `neutral` flag),
  `add_label_and_context`, `tournament_weight`. (martj42's `former_names.csv` was evaluated and skipped —
  the results feed already backfills current names in the 2006+ window.)
- **`features.py`** — `compute_elo` (from-scratch Elo since 2006, margin/upset-adjusted, +60 home), form &
  h2h features, `build_dataset`, `build_match_row` (leakage-safe as-of features).
- **`model_wdl.py`** — XGBoost `multi:softprob`; `split_by_date` (TRAIN_START 2006, VAL_START 2023),
  `train_model`, `predict_symmetric` (A-vs-B / B-vs-A averaging to kill home-order bias).
- **`model_goals.py`** — Dixon-Coles bivariate Poisson: time-decayed, neutral-aware `fit` (home advantage
  is learned from and applied to true-home matches only), `rates`, `matrix_from_rates`, `score_matrix`
  (takes `neutral=`), `wdl_from_matrix`.
- **`ensemble.py`** — `build` blends XGBoost + Dixon-Coles (log-loss-optimal `alpha`) and fits per-class
  isotonic calibrators on the temporal val slice, recency-weighted. `Predictor.predict` returns W/D/L +
  score matrix + the per-model components; carries the optional ledger-feedback layer (temperature +
  `alpha_effective`) — defaults are a no-op, so `build` stays pure.
- **`feedback.py`** — the "learn from whether it was right" layer. Fits a small, regularized adjustment
  (confidence temperature + tournament blend re-weight) on the ledger's ACCUMULATED scored predictions;
  shrinks to a no-op with few matches. Applied for LIVE predictions only, never inside `build`/`backtest`.
- **`ledger.py`** — append-only `ledger/predictions.csv` (git-tracked). `log_prediction`/`log_upcoming`
  (pre-registration), `score`, `report_card`. Honest-only headline (results unknown at log time).
- **`simulate.py`** — `run` Monte Carlo's the knockout bracket → per-team P(R16..Champion). Regulation
  draws resolve through a real ET/PK model: 1/3-length Poisson extra time at the match's expected-goal
  rates, then a shootout from `shootout.py`.
- **`shootout.py`** — penalty-shootout model fit on martj42 `shootouts.csv`: a shrunk 1-param Elo-edge
  logistic (empirically beats a coin: c≈0.45, a 400-Elo favorite wins ~61% of shootouts); degrades to
  50/50 offline via `fit_from_data`.
- **`fixtures.py`** — `parse_bracket` (Match 73-104, resolves winners from the live feed), `find_fixture`,
  `find_bracket_match`, `resolve_slots_for_date`, `compute_group_standings`.
- **`market.py`** — Polymarket Gamma API (`tournament_winner`, `match`), de-vig, 6h cache.
- **`backtest.py`** — walk-forward vs no-skill baseline AND Polymarket. Uses `ensemble.build` directly
  (NO feedback — that would be leakage).
- **`viz.py`** — charts (palette `ORANGE/BLUE/GRAY`, `INK/MUTE/GRID`).
- **`dashboard.py`** — writes a self-contained static `dashboard.html` (no server).
- **`server.py`** + **`static/`** — Flask API + vanilla-JS front end; builds state once at startup, a daemon
  thread auto-refreshes (results re-pull + rescore + recalibrate) every `--refresh-interval` min, and the
  page auto-polls. Reuses `dashboard.py`'s payload builders + design system.
- **`cli.py`** — argparse: `match today sim bracket backtest edge dashboard score serve`.

## Current tournament state (important)
Mid-tournament, **R16 complete, quarter-finals pending** as of 2026-07-09 (final is 2026-07-19). The live
`martj42` feed is authoritative:
played matches carry real scores, not-yet-played show `NA`, and it lags real kickoff by hours-to-days (so a
just-ended match may not be scoreable immediately — the report card surfaces `results_max_date` for this).
The bracket parser takes however many knockout rows the feed has resolved (not a fixed count) and falls back
to symbolic "Winner match N" refs for the rest. A tied played knockout match (90-minute score, decided on
PKs) gets its winner read off the next round's already-resolved pairing via the fixture placeholders.

## The closed learning loop (how "it learns from being right/wrong" actually works)
1. Every prediction (CLI or web) is logged to the ledger, tagged honest if the result wasn't yet known.
2. As results land (auto-refresh, `score`, or the UI Refresh button), honest predictions are scored
   (accuracy / log-loss / Brier vs no-skill baseline and vs Polymarket).
3. New results retrain Elo/form/Dixon-Coles/XGBoost, AND `feedback.py` fits a regularized temperature +
   blend adjustment on the accumulated scored ledger, applied to future live predictions. It grows with
   evidence and shrinks toward no-op when few matches exist — so single-match noise can't swing the model.

## Evaluated and rejected (don't redo these without new evidence)
Both were implemented and walk-forward backtested (2026-07); both regressed log-loss and were reverted:
- **Confederation features** (team→UEFA/CONMEBOL/... codes in `FEATURES`): backtest log-loss 0.810→0.815.
  Elo already encodes regional strength.
- **A feature-driven XGBoost-Poisson expected-goals model as a third ensemble member** (nested blend
  beta·DC + (1−beta)·gx): improved the goals side alone on validation (0.834→0.823) but regressed the
  final blend (backtest 0.810→0.815) — it consumes the same features as the W/D/L classifier, so the
  ensemble double-counts that view instead of gaining information.

## Known limitations
Rates teams not lineups (no injuries/suspensions/xG/tactics). W/D/L models under-call draws (the Dixon-Coles
half mitigates this). Feedback is deliberately conservative: with a handful of scored matches it barely moves
the model (by design). Free-data feed lag means "live" scoring isn't instant. Judge quality over many games
with log-loss, not single results. xG deferred — no clean free *international* xG feed exists.

## Conventions
- Free data only (open `martj42` dataset + Polymarket free API). No API keys, no paid feeds.
- Never let a model train on data at/after the match date — preserve the `split_by_date` cutoff discipline,
  and keep `feedback` out of `ensemble.build`/`backtest` (leakage).
- Keep it forkable: small readable files, no framework beyond a thin Flask app, no notebooks.
- Deps in `requirements.txt`: pandas, numpy, requests, xgboost, scikit-learn, matplotlib, scipy, flask.
