# 2026 FIFA World Cup match predictor

A small, readable model that predicts World Cup outcomes — win / draw / loss probabilities, full scoreline
distributions, tournament-winner odds, and a comparison against the live betting market. You give it two team
names (or nothing, and let it run today's slate), it does the rest.

I built this as a "first real ML project" you can fork and learn from. It started as one file. It's now a small
package — still no notebooks, no framework soup, every module short enough to read in five minutes.

```
python predict_today.py "Portugal" "Spain"
```

```
============================================================
  Portugal vs Spain
  2026-07-06  ·  R16  ·  Dallas Stadium
============================================================
  Portugal               win    24.9%
  Draw                          30.5%
  Spain                  win    44.5%
------------------------------------------------------------
  PICK: Spain  (44.5%)   [TOSS-UP]
============================================================
```

Or the full CLI:

```
python -m wcpred.cli today
python -m wcpred.cli sim
python -m wcpred.cli edge
```

## Two ways in

- **`predict_today.py`** — the original one-file experience. Type two teams, get a prediction and a chart.
  Still works exactly like it always did; internally it's now a thin shim into `wcpred.cli`.
- **`wcpred/`** — the full package, with a real CLI (`wcpred/cli.py`) and six subcommands (below). This is
  where the goals model, ensemble, tournament simulator, and market integration live.

```
pip install -r requirements.txt
python predict_today.py "Spain" "Cabo Verde"        # quick single match
python -m wcpred.cli match "Spain" "Cabo Verde"      # same thing, full CLI
```

Team order doesn't matter, and common spellings work (typing `Iran` is fine even though the schedule lists
`IR Iran`). Run either entry point with no arguments and it'll ask you for the two teams. The historical
results file (~5MB) auto-downloads the first time you run it.

## CLI commands

All six live under `python -m wcpred.cli <command>`.

- **`match A B`** — the original single-match prediction, now backed by the full ensemble: W/D/L probabilities,
  a bar chart, *and* a Dixon-Coles scoreline heatmap. Resolves already-decided knockout ties (e.g. "Portugal
  vs Spain") even though `fixtures.csv` still shows placeholder bracket text for them.
  ```
  python -m wcpred.cli match "Portugal" "Spain"
  ```
- **`today`** — runs `match` for every fixture actually scheduled on a given date (default: today), pulling the
  real matchups off the live results feed / resolved bracket.
  ```
  python -m wcpred.cli today --date 2026-07-03
  ```
- **`sim`** — Monte Carlo simulates the remaining knockout bracket thousands of times and reports each
  still-alive team's probability of reaching the R16/QF/SF/Final/winning it all. Knockout-only, since the group
  stage is already finished.
  ```
  python -m wcpred.cli sim --n 20000
  ```
- **`bracket`** — prints the resolved Match 73–104 bracket (finished results + still-pending slots) plus a
  road-to-the-final survival-probability chart.
  ```
  python -m wcpred.cli bracket
  ```
- **`backtest`** — walk-forward accuracy / log-loss / Brier score versus a no-skill baseline and versus
  Polymarket, using this tournament's already-played matches as ground truth.
  ```
  python -m wcpred.cli backtest
  ```
- **`edge`** — pulls live Polymarket odds (tournament winner + today's matches, no API key needed) and prints a
  model-vs-market disagreement table.
  ```
  python -m wcpred.cli edge
  ```

## How it works

Most of the work is still in the features, not any one model. For any match it builds:

- **Elo ratings.** From scratch over every international result since 2006, margin- and upset-adjusted, with a
  home-field bonus.
- **Recent form.** Win rate and goal difference over each team's last 5 and 10 matches, plus rest days.
- **Head-to-head.** How these two specific teams have done against each other historically.
- **Context flags.** Neutral venue, and how much a match "counts" (World Cup > friendly).

Those features feed two different models, blended together:

1. **XGBoost classifier** (`wcpred/model_wdl.py`) — the original model, `multi:softprob` over three classes.
   Good at picking up feature interactions on tabular data this size.
2. **Dixon-Coles bivariate Poisson** (`wcpred/model_goals.py`) — models each team's goals directly
   (time-decayed attack/defense strengths, a home-advantage term, and the classic Dixon-Coles low-score
   correction) and reads win/draw/loss off the resulting scoreline matrix. This is what fixes the one real
   weakness of a pure classifier: W/D/L models built by predicting the *label* directly tend to under-call
   draws, because "draw" has no clean statistical fingerprint of its own. A goals model doesn't have that
   problem — draws just fall out naturally wherever the two teams' scoring distributions overlap.

`wcpred/ensemble.py` blends the two (a single learned weight, found by minimizing log-loss on a held-out
temporal slice) and then calibrates the blended probabilities with per-class isotonic regression, so "60%"
actually means "wins about 60% of the time" rather than just "the model's raw score." Both models are always
trained only on data strictly *before* the match being predicted — there's no future leakage, calibration
cutoff included.

On a held-out temporal validation split, the calibrated ensemble log-loss is **0.834** (XGBoost alone: 0.859).
Walk-forward backtesting on 2023+ matches including this tournament puts the ensemble at **log-loss 0.844 /
accuracy 63.5%**, against a no-skill baseline of **1.044 / 49.4%** — a real, repeatable edge.

The honest part: on the subset of matches Polymarket also priced (72 this tournament), the market wins
outright — log-loss **0.501** and 98.6% accuracy versus the model's 0.808 and 66.7%. A liquid market pricing a
single well-known matchup is a very high bar to clear, and this project isn't trying to beat it head-on. Where
the model is actually useful is the volume game a market doesn't bother with in the same depth: full scoreline
distributions, per-round survival odds for every team still alive, and flagging the specific matches/teams
where model and market disagree (`wcpred.cli edge`) so you can decide for yourself who's more likely right.

## Tournament simulation

`wcpred/simulate.py` Monte Carlos the rest of the bracket (knockout-only — Match 73–104 — since the group stage
finished before this feature existed). Already-played matches are fixed historical fact and never re-simulated;
only genuinely pending slots draw an outcome from the ensemble predictor. A simulated regulation draw is broken
by a coin flip skewed toward whichever side the model favored, standing in for extra time + penalties (see
"what it doesn't do" below).

```
python -m wcpred.cli sim
```

## Market integration

`wcpred/market.py` pulls free, public odds from Polymarket's Gamma API (`https://gamma-api.polymarket.com`, no
key required) for the tournament-winner market and individual matches, de-vigs them into proper probabilities,
and maps Polymarket's team names onto the same names used everywhere else in the project. `wcpred.cli edge`
uses this to build the model-vs-market table above; `wcpred.cli backtest` uses it as a second point of
comparison alongside the no-skill baseline.

## Setup

```
pip install -r requirements.txt
python predict_today.py "Spain" "Cabo Verde"
```

Fixtures (date/group/stadium) are read from `data_cache/fixtures.csv`. The historical results file
auto-downloads to `data_cache/results.csv` on first run (gitignored); Polymarket responses are cached to
`data_cache/` too.

Each `match`/`today` run drops a branded probability bar chart *and* a scoreline heatmap in
`predictions/<date>/`; `sim`/`bracket` add a championship-odds bar chart and a road-to-the-final chart.

## What it doesn't do (yet)

Being honest about this matters more than any accuracy number.

- **No player-level detail.** Still rates teams, not the eleven players on the pitch — no injuries,
  suspensions, lineups, manager/tactics, or "only needs a draw to advance" context.
- **No xG.** There's no clean, free *international* xG feed to pull from; the goals model captures a good chunk
  of that signal for national teams, but it's not the same thing.
- **Extra time / penalties are a coin flip, not a shootout model.** When simulating a drawn knockout match, the
  winner is picked by a skewed coin flip favoring whichever team the model liked pre-match — there's no
  separate penalty-shootout model (goalkeeper save rates, kicker history, etc.).
  Reasonable stand-in, but not a real simulation of what actually happens after 120 minutes.
- **Group tiebreakers are simplified.** The (now-finished) group standings use points → goal difference →
  goals scored, which is *not* FIFA's full tiebreaker rule set (no head-to-head sub-rule, no fair-play points,
  no full best-8-of-12 third-place lookup table). This tournament's bracket doesn't actually depend on it —
  the live results feed's own already-resolved knockout matchups are taken as ground truth — but it's worth
  knowing the simplified table isn't the official one.
- **Loses to Polymarket on log-loss for the matches the market also prices.** See above — a liquid market is a
  genuinely hard thing to beat, and this project doesn't try to pretend otherwise.

Draws and single-match-only were the two big gaps in the original version — both are addressed now (Dixon-Coles
for draws, the ensemble/simulator/backtest for going beyond one match at a time).

## Data

- Historical results: the open [martj42/international_results](https://github.com/martj42/international_results)
  dataset.
- Fixtures: the official 2026 schedule (`data_cache/fixtures.csv`).
- Market odds: [Polymarket](https://polymarket.com)'s public Gamma API — free, no key required.

## License

MIT — do whatever you want with it. If you build something cool on top, I'd love to see it and make sure you
tag me @mar_antaya on Tiktok, Youtube and Instagram or Mariana Antaya on Linkedin!
