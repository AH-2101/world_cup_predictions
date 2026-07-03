# Implementation Plan — Supercharged World Cup Predictor (subagent-driven)

## How to run this plan
Work proceeds in **waves**. Launch all agents in a wave in a single message (parallel Agent calls,
`general-purpose` type). A wave completes and is smoke-tested before the next starts. Every agent **owns a
disjoint set of files** — no two agents in the same wave touch the same file — so no worktree isolation is
needed. Agents must honor the frozen **interface contracts** below so downstream waves compile against stable
signatures. After each agent: run its acceptance check; do not proceed on red.

## Target layout
```
wcpred/
  __init__.py
  data.py         # (moved) fetch/load/normalize + per_team_long + labels/weights
  features.py     # (moved) elo, form, h2h, build_dataset, *_as_of, FEATURES
  model_wdl.py    # (moved) split_by_date, train_model, evaluate, predict_symmetric, build_match_row
  model_goals.py  # NEW Dixon-Coles bivariate Poisson
  ensemble.py     # NEW blend + calibration -> unified predict
  simulate.py     # NEW bracket parse + tournament Monte Carlo
  market.py       # NEW Polymarket Gamma odds + de-vig
  backtest.py     # NEW walk-forward metrics
  viz.py          # (moved) make_chart + new charts
  cli.py          # NEW argparse subcommands
predict_today.py  # thin back-compat shim -> `wcpred.cli match A B`
requirements.txt  # +scipy
```

## Frozen interface contracts (all waves code to these)
- `data.load_results() -> DataFrame`  (normalized, sorted by date)
- `data.per_team_long(r) -> DataFrame`  (one row per team-match: team, opp, gf, ga, result, gd, date)
- `features.build_dataset(r) -> (dataset_df, final_elo: dict)`
- `features.FEATURES: list[str]`  (the 16 existing model columns)
- `model_wdl.predict_symmetric(model, long, final_elo, a, b, asof, neutral, weight) -> (p_home,p_draw,p_away)`
- `model_goals.fit(long, asof, halflife_days=180) -> params`  (attack/def/home-adv/rho, only data < asof)
- `model_goals.score_matrix(params, home, away, max_goals=10) -> np.ndarray[11,11]`  (P[i,j] = home i, away j)
- `model_goals.wdl_from_matrix(M) -> (p_home,p_draw,p_away)`
- `ensemble.build(dataset, long, final_elo, asof) -> Predictor`  with
  `Predictor.predict(home, away, neutral, weight) -> {p_home,p_draw,p_away,score_matrix}`  (calibrated)
- `market.tournament_winner() -> {team: prob}`  (de-vigged; team names already normalized to results.csv)
- `market.match(home, away) -> {p_home,p_draw,p_away} | None`
- `simulate.load_state(results, fixtures_path) -> Bracket`  (resolved + pending knockout slots)
- `simulate.run(bracket, predictor, n=20000, seed=42) -> DataFrame[team, p_r16,p_qf,p_sf,p_final,p_champion]`
- `backtest.run(dataset, long, final_elo, since="2022-01-01") -> DataFrame`  (accuracy/logloss/brier vs baseline & market)
- `viz.*` chart functions return the saved PNG path (match current `make_chart` convention)

---
## WAVE 0 — package skeleton (1 agent, blocks everything)
**Agent A0 · owns:** whole repo move. Extract the existing functions from `predict_today.py` verbatim into
`wcpred/{data,features,model_wdl,viz}.py` + `wcpred/fixtures.py`; add `__init__.py`; add `scipy` to
`requirements.txt`; replace `predict_today.py` with a shim that calls `wcpred.cli` (cli is a stub for now that
just reproduces today's single-match output). **Reuse, don't rewrite** the functions at
`compute_elo`:137, `per_team_long`:156, `predict_symmetric`:268, `find_fixture`:290, `make_chart`:320,
`split_by_date`:211, `train_model`:217.
**Accept:** `python predict_today.py "Portugal" "Spain"` produces the same prediction + chart as before.

## WAVE 1 — independent building blocks (3 agents, parallel)
- **Agent B1 · owns `wcpred/model_goals.py`** — Dixon-Coles bivariate Poisson: MLE via `scipy.optimize.minimize`
  on `per_team_long`, exponential time-decay weighting (`halflife_days`), low-score `rho` correction, `asof`
  cutoff (no leakage). Cache fitted params per `asof`. Implements `fit/score_matrix/wdl_from_matrix`.
  **Accept:** a standalone `__main__` prints a Portugal-vs-Spain scoreline matrix summing to ~1.0 and its W/D/L.
- **Agent B2 · owns `wcpred/market.py`** — pull from `https://gamma-api.polymarket.com` (no auth) the WC-2026
  tournament-winner + per-match markets; de-vig to probabilities; map Polymarket team names → results.csv names
  (extend the existing `NAME_MAP` idea); cache JSON to `data_cache/`. Implements `tournament_winner/match`.
  **Accept:** `python -m wcpred.market` prints a de-vigged winner table summing to ~1.0.
- **Agent B3 · owns new chart fns in `wcpred/viz.py`** (append only; do not touch `make_chart`) — `scoreline_heatmap`,
  `championship_bar`, `road_to_final`. Reuse the palette (`ORANGE/BLUE/GRAY`, INK/MUTE/GRID rcParams).
  **Accept:** each fn renders a PNG from synthetic inputs without error.

## WAVE 2 — composition (2 agents, parallel; need Wave 1)
- **Agent C1 · owns `wcpred/ensemble.py`** — blend `model_wdl` (XGBoost, via existing `predict_symmetric`) with
  `model_goals` W/D/L; fit blend weight + isotonic/temperature calibration on a temporal holdout; expose
  `build()`/`Predictor.predict()`. **Accept:** calibrated val log-loss ≤ the current 0.86 baseline number.
- **Agent C2 · owns `wcpred/simulate.py` + bracket parser in `wcpred/fixtures.py`** — parse knockout skeleton
  (Match 73–104) from `fixtures.csv`, resolve played slots from the live results feed, leave future slots
  symbolic; Monte-Carlo the remaining knockouts sampling scorelines from a passed-in `predictor` (draws →
  ET/penalty coin-flip weighted by favorite). Implements `load_state/run`. **Accept:** live-team P(champion)
  sums to ~1.0 and pre-tournament favorites still alive rank on top.

## WAVE 3 — surface + proof (2 agents, parallel; need Wave 2)
- **Agent D1 · owns `wcpred/backtest.py`** — walk-forward over 2022 WC + this tournament's played matches;
  report accuracy/log-loss/Brier vs the no-skill baseline AND vs Polymarket closing prices. **Accept:**
  `python -m wcpred.cli backtest` prints the metrics table; ensemble beats baseline log-loss.
- **Agent D2 · owns `wcpred/cli.py` + `predict_today.py` shim finalize** — argparse subcommands
  `match / today / sim / bracket / backtest / edge`, wiring every module + the new viz. **Accept:** all six
  subcommands run end-to-end on live data (see Verification).

## WAVE 4 — verification (1 agent)
- **Agent E1** — run the full Verification checklist below, fix any wiring gaps in files it owns, update
  `README.md` with the new commands + honest refreshed accuracy numbers. **Accept:** all checklist items green.

---
## Verification (end-to-end, on real current data — 2026-07-03)
1. `pip install -r requirements.txt`
2. `python predict_today.py "Portugal" "Spain"` — back-compat shim works; now shows a scoreline heatmap (real R16 tie 07-06).
3. `python -m wcpred.cli backtest` — ensemble beats baseline log-loss (~0.86); Brier vs Polymarket printed on played R32 matches (ground truth thru 07-02).
4. `python -m wcpred.cli sim` — championship-odds table; live favorites (Spain/France/Argentina/Brazil/Portugal) on top; P(champion) sums to ~1.0.
5. `python -m wcpred.cli edge` — live Polymarket odds pulled with no key; model-vs-market disagreement table populated.
6. `python -m wcpred.cli bracket` — resolved bracket + per-round survival probabilities render.
7. Leakage guard: for any played match, models train only on data strictly before its date (`split_by_date` discipline preserved).

## Conventions (enforce in every agent prompt)
Free data only (martj42 + Polymarket free API), no keys. Never train on data at/after the match date. Keep files
small/readable/forkable — no framework, no notebooks. Reuse existing functions; do not rewrite working code.
Each agent edits only the files it owns.
