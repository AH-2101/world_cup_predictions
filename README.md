# 2026 FIFA World Cup match predictor

A small, readable model that predicts World Cup outcomes — from a single match's win / draw / loss
probabilities and scoreline up through full tournament-winner odds — plus charts you can actually post. You
give it two teams (or nothing at all), it does the rest.

I built this as a "first real ML project" you can fork and learn from, then kept growing it without ever
turning it into a framework. It started as one file. It's now a small package (`wcpred/`) with about a dozen
short, single-purpose modules — still no notebooks, still nothing you can't read top to bottom in a few
minutes per file.

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
  Bar chart saved       -> predictions/2026-07-06/viz_Portugal_vs_Spain.png
  Scoreline heatmap     -> predictions/2026-07-06/viz_heatmap_Portugal_vs_Spain.png
```

## Two ways to run it

**`predict_today.py`** is still the one-file experience: type two teams, get a prediction and a chart. It's
now a two-line shim into the real package, kept around so the original quickstart never breaks.

```
pip install -r requirements.txt
python predict_today.py "Portugal" "Spain"      # order doesn't matter
python predict_today.py                          # prompts for the two teams
```

**`wcpred.cli`** is the full-featured entry point, with six subcommands covering everything from a single
match to the whole tournament:

```
python -m wcpred.cli match "Portugal" "Spain"     # same as predict_today.py, via the real subcommand name
python -m wcpred.cli today                        # every predictable fixture today
python -m wcpred.cli sim                           # Monte Carlo the bracket -> championship odds
python -m wcpred.cli bracket                       # resolved bracket + round-by-round survival odds
python -m wcpred.cli backtest                      # walk-forward accuracy vs. baseline and Polymarket
python -m wcpred.cli edge                          # model vs. live Polymarket odds
```

| Command    | What it does | Example |
|------------|---------------|---------|
| `match`    | Predicts one match — win/draw/win, scoreline heatmap, pick + confidence tag. Resolves already-decided knockout ties (fixtures.csv still shows placeholder text like "Group J winners v Group H runners-up" for those rows) via the live bracket. | `python -m wcpred.cli match "Portugal" "Spain"` |
| `today`    | Runs `match` for every fixture resolvable on a given date (default: today). | `python -m wcpred.cli today --date 2026-07-03` |
| `sim`      | Monte Carlo's the remaining knockout bracket ~5,000–20,000× and reports each live team's P(R16 / QF / SF / Final / Champion). | `python -m wcpred.cli sim --n 20000` |
| `bracket`  | Prints the resolved Match 73–104 bracket (winners where played, matchups where known, TBD where not) plus a survival-probability chart. | `python -m wcpred.cli bracket` |
| `backtest` | Walk-forward accuracy / log-loss / Brier on real FIFA World Cup matches, vs. a no-skill baseline and vs. Polymarket where a price exists. | `python -m wcpred.cli backtest` |
| `edge`     | Live Polymarket tournament-winner (and, where available, per-match) odds side-by-side with the model's, sorted by biggest disagreement. | `python -m wcpred.cli edge` |

Fixtures (date/group/stadium) are read from `data_cache/fixtures.csv`. The ~5MB historical results file
auto-downloads to `data_cache/results.csv` on first run (gitignored). Every run drops its charts in
`predictions/<date>/`.

## How it works

Most of the original approach is unchanged — the features still carry most of the signal — but the model
that turns them into a probability is now three models blended into one.

- **Elo ratings**, computed from scratch over every international result since 2006, plus **recent form**
  (win rate / goal diff over the last 5 and 10 matches), **rest days**, **head-to-head** history, and context
  flags (neutral venue, match importance).
- **XGBoost W/D/L classifier** (`wcpred/model_wdl.py`) — the original model, trained only on matches strictly
  before the one being predicted (`split_by_date`), with home/away order bias removed by averaging the A-vs-B
  and B-vs-A predictions (`predict_symmetric`).
- **Dixon-Coles goals model** (`wcpred/model_goals.py`, new) — a bivariate-Poisson model that fits time-decayed
  attack/defense ratings per team plus a home-advantage term and the classic Dixon-Coles low-score correlation
  correction, then reads win/draw/loss off the full predicted scoreline matrix instead of classifying the
  outcome directly. This is what fixes the old model's habit of under-calling draws — a Poisson-style model
  naturally produces draws as often as goal counts actually tie, instead of having to learn "draw" as an
  awkward third class squeezed between two more clearly-signaled outcomes.
- **Ensemble + calibration** (`wcpred/ensemble.py`, new) — blends the two models' probabilities with a single
  learned weight, then calibrates each outcome class (isotonic regression) against real observed frequencies
  on a held-out temporal slice, so "60% win probability" run through a lot of matches actually wins about 60%
  of the time.
- **Tournament Monte Carlo** (`wcpred/simulate.py`, new) — the group stage is finished, so this is
  knockout-only: it parses the resolved bracket (Match 73–104, `wcpred/fixtures.py`), replays already-decided
  matches as fixed history, and simulates every remaining tie thousands of times (a regulation draw goes to a
  coin flip skewed toward the model's favorite, standing in for extra time + penalties without a dedicated
  shootout model) to produce per-team odds of reaching each round.
- **Polymarket integration** (`wcpred/market.py`, new) — pulls the free, no-auth Polymarket Gamma API for
  tournament-winner and per-match odds, de-vigs them into real probabilities, and powers the `edge` command
  (plus the market comparison inside `backtest`).

The package layout:

```
wcpred/
  data.py         fetch/load/normalize results, per-team-match reshaping
  features.py     Elo, form, head-to-head, feature assembly
  model_wdl.py    XGBoost win/draw/loss classifier (the original model)
  model_goals.py  Dixon-Coles bivariate Poisson goals model
  ensemble.py     blend + calibration -> one unified predictor
  fixtures.py     fixture/name resolution + knockout bracket parser
  simulate.py     tournament Monte Carlo simulator
  market.py       Polymarket odds + de-vig
  backtest.py     walk-forward accuracy/log-loss/Brier report
  viz.py          all charts (bar, scoreline heatmap, championship bar, road-to-final)
  cli.py          the six subcommands above (plus a bonus `dashboard` command that
                   writes a single self-contained dashboard.html)
predict_today.py  thin back-compat shim -> `wcpred.cli match`
```

## Honest numbers

On a temporal holdout, the calibrated ensemble's log-loss is **0.834** (XGBoost alone: 0.859; a no-skill
baseline that always predicts the training class frequencies scores meaningfully worse).

The real test is the walk-forward backtest (`wcpred.cli backtest`) over every actual FIFA World Cup match
since 2022 (2022 Qatar matches themselves can't be scored without leakage — see below — so this run covers
this tournament's 85 already-played 2026 matches):

```
  metric  ensemble  baseline  ensemble_market_subset  market
accuracy    0.635     0.494                   0.667   0.986
log_loss    0.844     1.044                   0.808   0.501
   brier    0.494     0.628                   0.469   0.028
```

The ensemble clearly beats the no-skill baseline (log-loss 0.844 vs. 1.044, Brier 0.494 vs. 0.628). It does
**not** beat Polymarket on the 72 matches priced by both (log-loss 0.808 vs. the market's 0.501) — and I'm not
going to dress that up. A liquid, real-money market pricing a single well-known outcome (who wins a match
everyone's watching) is about as hard a benchmark as exists in forecasting; beating a no-skill baseline is a
real, useful signal, beating a deep market is a different and much higher bar. If there's an edge here, it's
more likely to show up in the day-to-day match calls the market doesn't bother to price precisely than in
head-to-head favorite-vs-favorite games everyone already agrees on — `edge` is there so you can go look for
it yourself instead of taking my word for it.

## What it doesn't do (yet)

Being honest about this matters more than any accuracy number. The model still rates *teams*, not the eleven
players actually on the pitch, so:

- No injuries or suspensions.
- No expected goals (xG) — no clean free international xG feed exists, so this is deferred; the goals model
  captures a good chunk of that signal for national teams, but not all of it.
- No lineups, no manager/tactics, no "this team only needs a draw to advance" context.

Two things from the original list are now meaningfully addressed: draws are no longer badly under-called
(that's what the Dixon-Coles model is for), and this is no longer single-match-only — `sim`/`bracket` give you
tournament-level odds. New, honest limitations that come with that:

- Extra time and penalties aren't modeled as a real shootout — a drawn knockout match is resolved with a coin
  flip skewed toward the model's favorite, not a dedicated ET/PK model.
- Group tiebreakers (`compute_group_standings`) use points / goal difference / goals scored, which is a
  reasonable simplification of FIFA's actual rules but skips the head-to-head sub-rule and fair-play points.
  It doesn't currently matter for anything the CLI does (the group stage is over and the bracket trusts the
  live results feed's already-resolved matchups rather than re-deriving them), but it's not a byte-for-byte
  reimplementation of the tiebreaker rules.
- The bracket parser takes the live results feed's already-resolved knockout matchups as ground truth rather
  than independently re-deriving FIFA's round-of-32 seeding table (which of the 8 group runners-up plays which
  bracket slot) — a deliberate, documented "simplest correct option" rather than a re-derivation of a fairly
  involved lookup table.
- It's still a win/draw/loss (plus now goals) model, not a crystal ball. Judge it over many games with
  log-loss, not on any single result.

## Data

Historical results come from the open [martj42/international_results](https://github.com/martj42/international_results)
dataset. Fixtures are the official 2026 schedule. Market odds come from the free, no-auth
[Polymarket Gamma API](https://gamma-api.polymarket.com) — no API keys anywhere in this project.

## License

MIT — do whatever you want with it. If you build something cool on top, I'd love to see it and make sure you tag me @mar_antaya on Tiktok, Youtube and Instagram or Mariana Antaya on Linkedin!
