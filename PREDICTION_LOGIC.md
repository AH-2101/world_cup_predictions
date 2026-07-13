# Prediction Logic

A deep-dive into how `wcpred` turns raw match history into a Win/Draw/Loss
probability, a full scoreline, and a championship-odds simulation — and how it
revises itself as tournament results come in. This is implementation detail;
for the high-level "what is this and how do I run it" see
[CLAUDE.md](CLAUDE.md).

## Pipeline overview

```
results.csv (martj42, free)          fixtures.csv (bracket/groups)
        │                                     │
        ▼                                     ▼
   wcpred/data.py                      wcpred/fixtures.py
   (normalize names,                   (parse_bracket, find_fixture,
    per_team_long, tournament_weight)   resolve_slots_for_date)
        │
        ▼
   wcpred/features.py
   (Elo, form, h2h → build_dataset)
        │
        ├─────────────────────────────┬───────────────────────────┐
        ▼                             ▼                            ▼
  wcpred/model_wdl.py           wcpred/model_goals.py        wcpred/shootout.py
  XGBoost multi:softprob        Dixon-Coles bivariate         Elo-edge shootout
  (W/D/L classifier)            Poisson (scoreline matrix)    logistic
        │                             │
        └───────────┬─────────────────┘
                     ▼
              wcpred/ensemble.py
              blend (alpha) + per-class isotonic calibration
                     │
                     ▼
              wcpred/feedback.py   ◄── wcpred/ledger.py (scored track record)
              tournament temperature + blend re-weight,
              applied to LIVE predictions only
                     │
        ┌────────────┼─────────────────┐
        ▼            ▼                 ▼
  single match   wcpred/simulate.py   ledger.log_prediction
  (cli match)    Monte Carlo bracket   (every prediction recorded,
                 (ET + shootout)        later scored by ledger.score)
```

Two independent models score every match — a classifier (XGBoost) that
predicts Win/Draw/Loss directly from hand-built features, and a generative
model (Dixon-Coles) that predicts the two teams' goal counts and derives
W/D/L from the resulting scoreline distribution. They're blended, calibrated
against held-out matches, and then nudged by a small, regularized adjustment
learned from this tournament's own scored predictions.

---

## 1. Data (`wcpred/data.py`)

- **Source**: the free, open `martj42/international_results` GitHub CSV
  (`results.csv`, 1872-present international matches; `shootouts.csv` for
  penalty-shootout history). No API keys, no paid feeds — see
  [CLAUDE.md](CLAUDE.md) conventions.
- **Caching**: downloaded to `data_cache/`, refreshed if the cached file is
  older than `RESULTS_STALE_SECONDS` (6h). A failed refresh falls back to the
  stale cache with a warning rather than hard-failing (`_fetch_csv`).
- **Name normalization**: `NAME_MAP` / `FIXTURE_NAME_MAP` reconcile spelling
  differences between the results feed, the fixtures file, and Polymarket
  (e.g. `"USA"` → `"United States"`, `"Türkiye"` → `"Turkey"`).
- **`tournament_weight(name)`**: a 1-4 importance score used both as an
  XGBoost feature and as a match-weighting rule elsewhere — World Cup finals
  matches = 4, qualifiers/continental championships = 3, generic
  competitions = 2, friendlies = 1.
- **`per_team_long(r)`**: reshapes one row-per-match into two rows-per-match
  (one from each team's point of view: `team`, `opp`, `gf`, `ga`, `neutral`,
  plus a `result` in {0, 0.5, 1}). This "long" shape is what form/h2h feature
  engineering and the Dixon-Coles fit both consume. Its exact row ordering
  (home-block then away-block) is also how `model_goals._pair_matches`
  recovers which side was actually "home" — see §3.
- **`tournament_today()`**: "today" is anchored to `America/Los_Angeles`
  (the westmost 2026 host timezone), not UTC — otherwise a game still in
  progress in the Americas evening would already have rolled to "tomorrow"
  in UTC and silently drop off "today's fixtures."

---

## 2. Feature engineering (`wcpred/features.py`)

`build_dataset(results)` produces one row per historical match with the
`FEATURES` XGBoost trains on, all leakage-safe (computed only from matches
strictly *before* the row's own date):

| Feature | What it captures |
|---|---|
| `neutral` | Whether the match was at a neutral venue (kills home advantage) |
| `tournament_weight` | Match importance (1-4, see above) |
| `home_elo`, `away_elo`, `elo_diff` | Pre-match Elo ratings and their difference |
| `home_win5`, `away_win5`, `home_gd5`, `away_gd5` | Last-5-match win rate and goal differential |
| `home_win10`, `away_win10` | Last-10-match win rate |
| `home_rest_days`, `away_rest_days` | Days since each side's previous match |
| `h2h_n`, `h2h_home_winrate`, `h2h_home_gd` | Head-to-head match count / win rate / avg goal differential between these two specific teams |

### Elo (`compute_elo`)

A from-scratch Elo implementation, not a canned library, fit fresh over the
whole results history in date order:

- Base rating 1500, K-factor 32, **+60 rating-point home bonus** (only for
  non-neutral matches).
- Expected score: standard logistic,
  `E_home = 1 / (1 + 10^(-((R_home + bonus) - R_away)/400))`.
- **Margin-of-victory + upset multiplier**: the rating update is scaled by
  `log(margin+1) * (2.2 / (|R_home-R_away| * 0.001 + 2.2))` — a bigger margin
  moves ratings more, but that effect is *damped* the more of an upset it
  already is (a 5-0 win by a 1900-rated side over a 1400-rated side moves
  ratings less than the same scoreline would for two evenly matched teams,
  since the outcome was already expected).
- Ratings are computed once over the full history and the *pre-match* rating
  (`home_pre`/`away_pre`, i.e. the state before that match updates it) is
  what's stored as the feature — this is what makes it leakage-safe for
  training. For live prediction, `build_match_row` instead uses each team's
  *final* Elo (`final_elo`, the value after the last historical match) as
  the best current estimate.

### Form and head-to-head (`add_form_features`, `add_h2h_features`)

Both are rolling/expanding aggregates over `per_team_long`, shifted by one
(`.shift(1)`) so a team's own about-to-be-played match is never included in
its own form features. Form uses fixed 5- and 10-match trailing windows;
head-to-head uses the full history between exactly these two teams,
expanding (not windowed).

### Live prediction rows (`form_as_of`, `h2h_as_of`, `build_match_row`)

For a match that hasn't happened yet, these recompute the same features
as-of an arbitrary date by filtering `long` to `date < asof` and taking the
tail — no leakage possible because it's re-derived on demand from real
history, not read off a training-time cache. A team with no history at all
gets a neutral prior (`win5=0.5`, `gd5=0.0`, `rest_days=30`).

---

## 3. Model A — XGBoost W/D/L classifier (`wcpred/model_wdl.py`)

- `xgb.XGBClassifier(objective="multi:softprob", num_class=3, ...)`, 600
  trees, `max_depth=5`, `learning_rate=0.05`, subsample/colsample 0.85,
  L2 `reg_lambda=1.0`, early stopping (50 rounds) on validation log-loss.
- **Temporal split** (`split_by_date`), never random: `TRAIN_START =
  2006-01-01`, `VAL_START = 2023-01-01`, and an `asof` cutoff — train on
  `[2006, 2023)`, validate/calibrate on `[2023, asof)`. 2006 is chosen as the
  training floor (older results are noisier/less representative of the
  modern game); the val split doubles as the ensemble's calibration data
  (§5).
- **Symmetric prediction (`predict_symmetric`)**: the model is trained on
  rows where "home" and "away" are arbitrary labels (whoever the data source
  called home), so a single softmax prediction is subtly biased by row order.
  To cancel that out, every live prediction queries the model twice — once
  as A-vs-B, once as B-vs-A — and averages `P(A wins)` from the two calls
  (`(p_ab[home] + p_ba[away]) / 2`), same for draw/away, then renormalizes.
  This mechanically removes any left/right ordering artifact the classifier
  might have picked up.

---

## 4. Model B — Dixon-Coles goals model (`wcpred/model_goals.py`)

Where XGBoost classifies an outcome directly, this model predicts the full
**scoreline distribution** — and W/D/L falls out of that distribution rather
than being predicted directly. This is specifically what fixes XGBoost's
tendency to under-call draws (see [CLAUDE.md](CLAUDE.md) Known Limitations):
a generative goals model naturally assigns draws their correct share of
probability mass wherever two teams' expected-goal rates are close, without
needing a draw-specific feature or class-imbalance correction.

### The model

```
home_goals ~ Poisson(λ),  λ = exp(home_adv · is_true_home + attack[home] − defense[away])
away_goals ~ Poisson(μ),  μ = exp(attack[away] − defense[home])
```

`is_true_home` is 0 at neutral venues (most 2026 matches for the "nominal
home" side), so home advantage is neither learned from nor applied to them —
critical for a tournament played almost entirely at neutral venues.

**Dixon-Coles low-score correction**: plain independent Poissons
systematically mis-predict the four low-scoring cells (0-0, 1-0, 0-1, 1-1)
relative to reality (real matches have slightly more 0-0/1-1 draws and
slightly fewer 1-0/0-1 than independence implies). A single correlation
parameter `rho` reweights exactly those four cells:

```
τ(0,0) = 1 − λμρ      τ(0,1) = 1 + λρ
τ(1,0) = 1 + μρ       τ(1,1) = 1 − ρ      τ(i,j) = 1 otherwise
```

### Fitting (`fit`)

Maximum likelihood via `scipy.optimize.minimize` (L-BFGS-B, with an
**analytic gradient** — not finite-difference, for both speed and precision
on ~12.5k matches). Parameters: `attack_i`, `defense_i` per team, one global
`home_advantage`, one global `rho`.

- **Time decay**: every match is weighted `0.5 ** (days_before / halflife_days)`
  (default half-life 180 days), so recent form dominates but old results
  aren't discarded outright. Matches with negligible weight
  (`< WEIGHT_PRUNE_THRESHOLD`) are dropped before fitting, for speed.
- **Identifiability ridge**: adding a constant to every `attack_i` and every
  `defense_i` leaves the likelihood unchanged (only `attack_i − defense_j`
  ever appears in the model). A small L2 penalty (`RIDGE = 1e-3`) on
  attack/defense picks the minimum-norm solution, which centers ratings at
  ≈0 = league average — otherwise the optimizer would wander along that flat
  direction indefinitely.
- **Numerical safety**: the log-rate `home_adv + attack − defense` is clipped
  to `[-6, 3]` before exponentiating (`LOG_RATE_MIN/MAX`), and every
  parameter is bounded to `±6` in L-BFGS-B. This exists because an
  unbounded optimizer can occasionally take a bad early step that overflows
  `exp()`, after which the gradient goes non-finite and the fit "converges"
  to garbage (the module docstring records an observed failure:
  `home_advantage=109`, `attack` as extreme as `-1064`). A healthy fit sits
  well inside these bounds, so the clip is a no-op except when a fit is
  actually diverging.
- **Match reconstruction (`_pair_matches`)**: `fit` is handed `long`
  (`per_team_long`'s two-rows-per-match shape), not the original one-row
  results table, and needs to recover which side was truly "home" per match.
  It relies on `per_team_long`'s row order (home-block first, then the
  mirrored away-block) to pair rows back into matches positionally; if that
  invariant doesn't hold (e.g. `long` was filtered/reordered upstream) it
  falls back to a best-effort pairing by `(date, {team, opp})` — home/away
  becomes an arbitrary but deterministic pick in that fallback, so average
  home-advantage is still estimable even though any single match's
  assignment isn't guaranteed correct.
- **Caching**: fits are memoized in-process by `(asof, halflife_days)`
  (`_CACHE`), since the same fit is reused across many predictions within one
  run (e.g. every knockout matchup in a Monte Carlo simulation).

### Prediction (`rates`, `matrix_from_rates`, `score_matrix`, `wdl_from_matrix`)

`rates(params, home, away, neutral)` → `(λ, μ)`; unknown teams (no history)
fall back to the league-average attack/defense. `matrix_from_rates` builds
the full scoreline matrix as an outer product of the two Poisson pmfs
(0..10 goals each), applies the four-cell `tau` correction, clips negatives,
and renormalizes to sum to 1. `wdl_from_matrix` sums the upper/diagonal/lower
triangle of that matrix to get `(p_home, p_draw, p_away)`.

---

## 5. Ensemble: blend + calibration (`wcpred/ensemble.py`)

Three steps, all fit on the same temporal validation split as XGBoost
(`[VAL_START, asof)`), so nothing here trains on data at or after the match
being predicted:

### Step 1 — build both models on the identical split
`model_wdl.train_model` and `model_goals.fit` are both fit on
`[TRAIN_START, VAL_START)` / decayed history before `asof` respectively (see
§3-4). `_dc_val_probs` runs the Dixon-Coles model over every validation-split
match to get its W/D/L too.

### Step 2 — blend weight `alpha`
A single scalar in `[0, 1]`: `blended = alpha * p_xgb + (1 - alpha) * p_dc`.
Found by 1-D bounded minimization (`scipy.optimize.minimize_scalar`) of
log-loss on the validation split. A single global weight (not per-match or
per-feature) is enough because both component models are already fit on the
full training history — `alpha` just answers "on average, how much more
should I trust the classifier vs. the goals model?"

**Recency weighting** (`recency_halflife_days=365`, the "learn from what
just happened" step): validation rows are additionally weighted by
`0.5 ** (age_days / 365)` when solving for `alpha`, so this tournament's own
already-played matches count more than older validation-split history. Two
guardrails stop a handful of new matches from swinging the blend too hard:
- a weight floor of 0.10 (older 2023-2025 validation history never drops to
  zero influence),
- shrinkage of the final `alpha` toward the *unweighted* optimum
  (`alpha_base`), proportional to how many recent matches actually exist:
  `shrink = n_recent / (n_recent + 20)`, `alpha = alpha_base + shrink *
  (alpha_recency - alpha_base)`.

Passing `recency_halflife_days=None` reproduces the original unweighted
behavior exactly — used as an A/B regression check (`val_log_loss_calibrated`
vs `val_log_loss_calibrated_base` are both computed and compared every build).

### Step 3 — per-class isotonic calibration
One `sklearn.isotonic.IsotonicRegression` **per outcome class** (home/draw/
away), each fit mapping that class's blended probability → observed
frequency on the validation split (one-vs-rest isotonic calibration — the
same idea `sklearn.CalibratedClassifierCV` uses per-class). At prediction
time, each of the three blended probabilities is pushed through its own
isotonic curve and the three outputs renormalized to sum to 1. This corrects
systematic over/under-confidence left over after blending (e.g. if matches
blended to "70% home win" actually resolve to a home win 62% of the time,
isotonic calibration learns that correction directly from data rather than
assuming a parametric shape).

### `Predictor.predict(home, away, neutral, weight)`
Runs both models, blends (using `alpha_effective` if the feedback layer has
set one — see §6, else `alpha`), calibrates, then applies the tournament
temperature (§6, a no-op by default). Returns `p_home/p_draw/p_away`, the
full `score_matrix`, both models' raw component probabilities (`p_xgb`,
`p_dc` — stored in the ledger for feedback to use later), and the raw
expected-goal rates `lam_home`/`lam_away` (consumed by the extra-time model
in §7).

---

## 6. The closed learning loop (`wcpred/feedback.py`)

This is the "learns from being right or wrong" piece described in
[CLAUDE.md](CLAUDE.md). It is explicitly **not** part of `ensemble.build` or
the backtest — it only ever runs for live predictions (CLI `match`/`today`,
the server), and only ever reads ledger rows with `match_date < predictor.asof`
to avoid using a result to score the very match that produced it.

Two adjustments, both fit by minimizing log-loss over the ledger's scored,
*honest* (result unknown at logging time), *deduplicated* (latest prediction
per match) rows, and both shrunk toward a no-op so a handful of matches can't
swing the live model:

**Temperature `T`** — `p_i ** (1/T)`, renormalized, applied to the *final*
calibrated probabilities already stored in the ledger. `T > 1` softens
(pulls toward uniform, when the model has been confidently wrong); `T < 1`
sharpens (when it's been correct but underconfident). Fit by 1-D
minimization over `T ∈ [0.3, 4.0]`, then shrunk:
`T_eff = 1 + (n/(n+10)) * (T_opt - 1)`.

**Blend re-weight `alpha_tournament`** — re-solves the same
`alpha * p_xgb + (1-alpha) * p_dc` blend, but only against *this
tournament's* scored matches (using the per-model component probabilities
that were also logged), to see whether XGBoost or Dixon-Coles has actually
been more accurate so far this World Cup specifically. Only computed once
≥3 scored rows carry components; shrunk toward the base `alpha`:
`alpha_eff = alpha_base + (n/(n+15)) * (alpha_tournament - alpha_base)`.

Both need `min_matches=3` scored rows to activate at all; below that, `fit`
returns a strict no-op (`temperature=1.0`, `alpha_effective=None`). `apply`
fits and attaches the adjustment to a `Predictor` in place
(`predictor.tournament_temperature`, `predictor.alpha_effective`,
`predictor.feedback_info`), which `Predictor.predict` then applies
automatically. `summary_line` renders the one-line CLI/server status message
(e.g. *"Feedback from 9 scored match(es): temperature T=0.927 (sharpening
underconfident picks) | blend alpha 0.143 -> 0.388 | log-loss on those
matches 0.952 -> 0.950"*).

---

## 7. Extra time + penalty shootouts (`wcpred/shootout.py`, part of `wcpred/simulate.py`)

Knockout matches can't end in a draw, so any simulated regulation draw needs
a principled way to pick a winner — this replaced an earlier fixed 0.55
coin-flip.

**Extra time**: modeled as a *shorter* independent-Poisson period at the
same match's expected-goal rates, scaled by `ET_LENGTH_FACTOR = 1/3` (30 of
90 minutes) — `model_goals.matrix_from_rates(λ/3, μ/3, rho=0, max_goals=6)`.
Dixon-Coles's `rho` correction is dropped here (`rho=0`) since it corrects
regulation-time low-score biases, not modeled as relevant at this scale.
Because it reuses the match's own `(λ, μ)`, a heavily lopsided matchup keeps
a proportionally larger edge in extra time, rather than reverting to 50/50.

**Penalty shootout**: if still tied after (simulated) extra time,
`wcpred.shootout` decides it. This module first asks a real empirical
question — *is a shootout actually just a coin flip?* — by fitting
`P(home wins shootout) = sigmoid(c * elo_diff / 400)` via MLE on
`shootouts.csv` joined to pre-match Elo ratings, shrinking the fitted `c` by
`n/(n+200)`, and **zeroing it entirely if the shrunk model can't beat a
plain coin on log-loss** (the football literature broadly says shootout
outcomes are close to independent of team strength). Empirically here it
does beat the coin: `n=681`, `c≈0.428` (raw MLE 0.553), log-loss 0.6889 vs
0.6931 for a coin — a 400-Elo favorite wins a shootout ~61% of the time
rather than 50%. If shootout data is unavailable this degrades cleanly to
`c=0.0` (fair coin) rather than erroring.

`_p_home_given_draw` in `simulate.py` combines both:
`P(home advances | draw) = P(home wins ET) + P(ET still drawn) * P(home wins shootout)`.

---

## 8. Monte Carlo bracket simulation (`wcpred/simulate.py`)

`wcpred.fixtures.parse_bracket` reads the knockout bracket (fixtures.csv
Match 73-104) from the live results feed — already-played matches carry
real winners; still-pending ones carry symbolic references like
"Winner of Match 81." `simulate.run(bracket, predictor, n=20000)` then plays
the *remaining* bracket out `n` times:

- Match numbers are processed in increasing order per simulation, so any
  "winner of match N" reference is always already resolved within that same
  simulation (`_resolve_side`).
- **Already-played matches are never re-simulated** — their real recorded
  winner is reused in every single run; only genuinely future matches draw
  from the predictor.
- **Prediction caching per (home, away) pair** (`pred_cache` in `_play`):
  many bracket slots have a fixed matchup across all 20,000 simulations
  (e.g. an already-known-but-unplayed R16 pairing), so the relatively
  expensive XGBoost+Dixon-Coles prediction for that pairing is computed once
  and reused, not recomputed 20,000 times.
- A drawn regulation result routes through the ET+shootout model from §7.
- Tallies, per still-alive team, the fraction of simulations reaching each
  milestone (`p_r16, p_qf, p_sf, p_final, p_champion`). `p_champion` sums to
  ≈1.0 across the full returned field, since every simulated champion is
  necessarily one of the returned (still-alive) teams.
- A `_FallbackPredictor` (XGBoost W/D/L only, naive independent-Poisson score
  matrix) exists purely so `simulate.py` still runs standalone before
  `ensemble.py` exists / if it fails to import — not used once the ensemble
  is available.

---

## 9. The prediction ledger + self-scoring (`wcpred/ledger.py`)

Append-only `ledger/predictions.csv` (git-tracked, per [CLAUDE.md](CLAUDE.md)),
the record that both §6 (feedback) and the web dashboard's "report card"
read from.

- **`log_prediction`**: writes one row per prediction — teams, round,
  the model's `p_home/p_draw/p_away`, its top pick, the market's odds if
  available, both models' raw component probabilities, and critically
  `result_known_at_log` — whether the real result was already knowable at
  logging time (computed via `_result_known`, a ±1-day window match against
  live results). Rewrites the whole (small) file each time rather than
  appending, which sidesteps schema-migration issues when new columns are
  added — old rows just get blank values via `reindex(columns=COLUMNS)`.
- **`log_upcoming`**: pre-registers predictions for every resolvable fixture
  in the next few days, *before* results exist — this is what lets the
  system score itself honestly later, rather than only ever logging
  after-the-fact. Idempotent per calendar day per source.
- **`score(results)`**: joins the ledger against live results, adding
  `y` (actual outcome), `correct`, `log_loss`, `brier` per row (`None` where
  unplayed).
- **`report_card(results)`**: the headline numbers — but only over
  **honest** rows (`result_known_at_log == False`), latest prediction per
  match. Compares accuracy/log-loss/Brier against a no-skill class-frequency
  baseline and, where priced, against Polymarket. A prediction logged after
  the result was already known would flatter the model, so it's excluded
  from the headline by construction, not just by convention.

---

## 10. Walk-forward backtest (`wcpred/backtest.py`)

Distinct from the ledger's live self-scoring: this evaluates the ensemble
against **already-known real outcomes**, retroactively, as a methodology
check — always using `ensemble.build` directly (no feedback layer, since
that would be leakage against matches whose results are what's being tested).

- **Sample**: every match with `tournament_weight == 4` (an actual FIFA
  World Cup finals match — see §1) since a given `since` date (default
  2022-01-01) — i.e. the 2022 World Cup plus 2026's already-played matches,
  the highest-stakes subset and the only one with real Polymarket prices to
  compare against.
- **The VAL_START floor**: `ensemble.build` needs `asof` strictly after
  `VAL_START=2023-01-01` for its validation/calibration slice to be
  non-empty, but leakage safety also needs `asof <= match_date`. Those two
  constraints are jointly impossible for any match at or before
  `VAL_START` — so the entire 2022 World Cup is detected and **skipped**
  (reported as `n_skipped_pre_val`), not silently mis-scored.
- **Monthly `asof` buckets**: re-fitting the full ensemble per match (~15s
  each) would be far too slow, so matches are grouped by calendar month and
  the ensemble is fit once per bucket at `asof = 1st of that month`. Since
  every match in a "YYYY-MM" bucket is on or after that date, this never
  trains on a match's own or later data for any match in the bucket — some
  late-month matches are scored by a slightly "stale" (start-of-month) fit,
  but the no-leakage invariant always holds.
- **Metrics**: accuracy, 3-class log-loss, Brier (`sum over classes of
  (p_c - onehot_c)^2`, averaged over matches) — reported for the ensemble,
  a no-skill baseline, and (on the subset of matches with a Polymarket
  price) the market itself, so the ensemble can be honestly compared against
  "did nothing" and "what the market already knew."

---

## 11. Market comparison (`wcpred/market.py`)

Free Polymarket Gamma API, no auth. Two entry points:
`tournament_winner()` (the single World-Cup-winner event, one Yes/No
sub-market per team) and `match(home, away)` (per-fixture events, slug
pattern `fifwc-<3-letter>-<3-letter>-2026-MM-DD`, found via full-text search
since exact slugs aren't known ahead of time). Both **de-vig**: raw Yes
prices sum to slightly over 1.0 (the market's built-in overround/vig), so
each is divided by the total to normalize to a proper probability
distribution. `match()` deliberately returns `None` rather than fabricating
a draw probability when only a binary win/lose market exists for a fixture.
Results are cached 6h, same "stale cache beats hard failure" convention as
`data.py`.

---

## 12. Design decisions evaluated and rejected

Per [CLAUDE.md](CLAUDE.md), both were implemented and walk-forward
backtested, then reverted because they regressed log-loss — recorded here so
they aren't silently retried:

- **Confederation features** (team → UEFA/CONMEBOL/... one-hot in
  `FEATURES`): backtest log-loss 0.810 → 0.815. Elo already encodes regional
  strength implicitly, so the extra features added noise, not signal.
- **A third ensemble member** — a feature-driven XGBoost-Poisson
  expected-goals model, nested into the blend as `beta·DC + (1-beta)·gx`:
  improved the goals side alone on validation (0.834 → 0.823) but regressed
  the *final* blend (0.810 → 0.815). It consumes the same underlying
  features as the W/D/L classifier, so blending it in double-counts that
  view rather than adding independent information.

---

## 13. Known limitations

- Rates **teams**, not lineups — no injuries, suspensions, xG, or tactics
  data feed into any model.
- The W/D/L classifier alone under-calls draws; Dixon-Coles's generative
  scoreline approach is specifically what compensates for this in the blend.
- Feedback (§6) is deliberately conservative — with only a handful of scored
  matches it barely moves predictions at all, by design (the shrinkage
  terms), so early-tournament noise can't swing the live model.
- The live results feed lags real kickoff by hours to days, so "live"
  self-scoring isn't instant — the report card surfaces `results_max_date`
  to make that lag visible rather than hiding it.
- No clean free *international* xG feed exists, so xG-based features remain
  out of scope for now.
