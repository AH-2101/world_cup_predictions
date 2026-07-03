# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is
A 2026 FIFA World Cup match predictor. Today it is a single well-documented file, `predict_today.py` (~418
lines): given two team names it prints Win/Draw/Loss probabilities and saves a branded bar chart to
`predictions/<date>/`. Built as a readable, forkable "first real ML project" — no notebooks, no framework.

## How to run
```
pip install -r requirements.txt
python predict_today.py "Portugal" "Spain"      # order doesn't matter
python predict_today.py                          # prompts for the two teams
```
Fixtures (date/group/stadium) are read from `data_cache/fixtures.csv`. The ~5MB historical results file
auto-downloads to `data_cache/results.csv` on first run (gitignored).

## Architecture (current)
Everything lives in `predict_today.py`, in this pipeline:
- **Data** — `fetch_results` / `load_results` pull the open `martj42/international_results` CSV; `NAME_MAP` /
  `FIXTURE_NAME_MAP` normalize country spellings.
- **Features** — `compute_elo` (from-scratch Elo since 2006, margin- and upset-adjusted, +60 home bonus),
  `add_form_features` (win rate + goal diff over last 5/10, rest days), `add_h2h_features` (historical
  head-to-head). Assembled by `build_dataset`. `per_team_long` reshapes matches to one row per team-match.
- **Model** — `train_model`: XGBoost `multi:softprob` (3 classes), trained per-match on data strictly before
  the match date (`split_by_date`) so there's no future leakage. `predict_symmetric` averages the A-vs-B and
  B-vs-A predictions to remove home-order bias.
- **Fixtures/CLI** — `find_fixture`, `map_fixture_name`, `list_team_names` resolve user input against
  `fixtures.csv`. `make_chart` renders the branded PNG; `tag_match` labels LOCK/LEAN/TOSS-UP + upset flag.

Reuse these functions — don't reinvent them. Key line refs: `compute_elo`:137, `per_team_long`:156,
`predict_symmetric`:268, `find_fixture`:290, `make_chart`:320, `split_by_date`:211, `train_model`:217.

## Current tournament state (important)
It is **2026-07-03, mid-tournament**. The group stage is finished and the Round of 32 is in progress. The live
`martj42` results feed is current through today: played matches (incl. R32 through 07-02) have real scores;
07-03+ matches show `NA`. So Elo/form are already up to date and the bracket is deterministic from here.
Bracket map in `fixtures.csv`: Match 73–88 = R32, 89–96 = R16, 97–100 = QF, 101–102 = SF, 103 = 3rd place,
104 = Final.

## Known limitations (from the README)
Rates teams not lineups (no injuries/suspensions/xG/tactics); under-calls draws (W/D/L models do); single-match
only — can't give tournament-winner odds. Judge quality over many games with log-loss, not single results.

## Planned upgrade: "supercharged" predictor
Scope = major upgrade keeping the forkable spirit; grow the one file into a small `wcpred/` package (~6 modules
+ thin CLI). `predict_today.py` stays as a back-compat entry point. Roadmap:
1. **Dixon-Coles bivariate Poisson goals model** (`model_goals.py`) — time-decayed attack/defense strengths +
   low-score correction. Predicts full scoreline matrices, fixing the draw under-call. Reuse `per_team_long`.
2. **Ensemble + calibration** (`ensemble.py`) — blend XGBoost W/D/L with Dixon-Coles, calibrate on a temporal
   holdout (isotonic/temperature). Keep `predict_symmetric` de-biasing.
3. **Tournament Monte Carlo** (`simulate.py`) — parse the resolved bracket, simulate remaining knockouts ~20k×
   (extra-time/penalty logic for drawn knockouts), output per-team P(R16/QF/SF/Final/Champion). Knockout-only
   since groups are done.
4. **Polymarket integration** (`market.py`) — free public Gamma API (`https://gamma-api.polymarket.com`, no
   auth) for tournament-winner + per-match odds; de-vig; power an `edge` (model-vs-market) command + optional
   prior.
5. **Backtest harness** (`backtest.py`) — walk-forward accuracy/log-loss/Brier vs the no-skill baseline AND vs
   Polymarket, using this tournament's already-played matches as ground truth.
6. **Shareable viz** (`viz.py`) — scoreline heatmaps, championship-odds bars, "road to the final" bracket,
   optional self-contained `dashboard.html`. Reuse existing palette (`ORANGE/BLUE/GRAY`, INK/MUTE/GRID).
7. **CLI** (`cli.py`, argparse) — `match`, `today`, `sim`, `bracket`, `backtest`, `edge`.

xG is deferred to future work: no clean free *international* xG feed exists; the goals model captures most of
that signal for national teams.

## Conventions
- Free data only (open `martj42` dataset + Polymarket free API). No API keys, no paid feeds.
- Never let a model train on data at/after the match date — preserve the `split_by_date` cutoff discipline.
- Keep it forkable: small readable files, no framework, no notebooks.
- Deps live in `requirements.txt` (pandas, numpy, requests, xgboost, scikit-learn, matplotlib; +scipy for the
  goals model).
