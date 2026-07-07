"""wcpred.server — Flask API for the interactive predictor front end.

Building the ensemble predictor + championship-odds simulation costs several
seconds to tens of seconds (XGBoost train + Dixon-Coles fit + a few thousand
Monte Carlo trials), so all of that happens ONCE at startup into an
in-process `State` and is only rebuilt on an explicit `POST /api/refresh` —
every other endpoint just reads the cached state. `/api/predict` (and the
startup auto-capture below) log to `wcpred.ledger` so predictions get scored
once real results land, closing the loop: predict -> result lands -> refresh
-> rescore -> recalibrate.
"""

import os
import threading
import time

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from wcpred.data import load_results, per_team_long, tournament_today
from wcpred.features import build_dataset
from wcpred.fixtures import (
    FIXTURES_PATH, find_bracket_match, find_fixture, list_team_names, parse_bracket,
    resolve_slots_for_date,
)
from wcpred.model_wdl import MATCH_NEUTRAL, MATCH_WEIGHT
from wcpred import dashboard, ensemble, feedback, ledger, market, simulate

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.results = self.dataset = self.final_elo = self.long = None
        self.bracket = self.predictor = None
        self.championship = []
        self.market_error = None
        self.built_at = None


def build_state(state=None, force_refresh=False, sim_n=5000, seed=42):
    """(Re)build every piece of server state from scratch. Pass an existing
    `state` to update it in place (used by /api/refresh)."""
    state = state or State()
    print("[server] loading results ..." + (" (forcing refresh)" if force_refresh else ""))
    results = load_results(force=force_refresh)
    dataset, final_elo = build_dataset(results)
    long = per_team_long(results)
    bracket = parse_bracket(FIXTURES_PATH, results)

    asof = tournament_today()
    print("[server] building ensemble predictor (XGBoost + Dixon-Coles) ...")
    predictor = ensemble.build(dataset, long, final_elo, asof)
    adj = feedback.apply(predictor, results)
    print("[server] " + feedback.summary_line(adj))

    print(f"[server] simulating {sim_n} tournament completions ...")
    sim_table = simulate.run(bracket, predictor, n=sim_n, seed=seed)
    championship = [
        {"team": r.team, "p_r16": dashboard._round(r.p_r16), "p_qf": dashboard._round(r.p_qf),
         "p_sf": dashboard._round(r.p_sf), "p_final": dashboard._round(r.p_final),
         "p_champion": dashboard._round(r.p_champion)}
        for r in sim_table.itertuples(index=False)
    ]

    market_error = None
    try:
        market.tournament_winner(force_refresh=force_refresh)
    except market.MarketError as exc:
        market_error = str(exc)

    state.results, state.dataset, state.final_elo = results, dataset, final_elo
    state.long, state.bracket, state.predictor = long, bracket, predictor
    state.championship, state.market_error = championship, market_error
    state.built_at = pd.Timestamp.now().isoformat()

    n_new = ledger.log_upcoming(predictor, results, source="server")
    print(f"[server] ready. auto-logged {n_new} upcoming prediction(s). built_at={state.built_at}")
    return state


def _resolve_match(state, team_a, team_b):
    """Same resolution order as the CLI (find_fixture -> bracket fallback),
    plus a free-form fallback so /api/predict works for any two valid teams,
    not just tournament fixtures."""
    m = find_fixture(team_a, team_b) or find_bracket_match(team_a, team_b, state.results)
    if m is not None:
        return m
    valid = set(state.results["home_team"]) | set(state.results["away_team"])
    home = next((t for t in valid if t.lower() == team_a.strip().lower()), None)
    away = next((t for t in valid if t.lower() == team_b.strip().lower()), None)
    if home is None or away is None:
        return None
    today_str = tournament_today().strftime("%Y-%m-%d")
    return {"match_number": "", "group": "Friendly", "stadium": "", "date": today_str,
            "home_disp": home, "away_disp": away, "home": home, "away": away}


def create_app(state):
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:path>")
    def static_files(path):
        return send_from_directory(STATIC_DIR, path)

    @app.get("/api/teams")
    def api_teams():
        valid = set(state.results["home_team"]) | set(state.results["away_team"])
        return jsonify(sorted(t for t in list_team_names() if t in valid))

    @app.get("/api/today")
    def api_today():
        date = request.args.get("date") or tournament_today().strftime("%Y-%m-%d")
        slots = resolve_slots_for_date(state.results, date)
        return jsonify([dashboard._match_payload(m, state.predictor, state.final_elo) for m in slots])

    @app.get("/api/predict")
    def api_predict():
        home_in, away_in = request.args.get("home", ""), request.args.get("away", "")
        m = _resolve_match(state, home_in, away_in)
        if m is None:
            return jsonify({"error": f"couldn't resolve a match between {home_in!r} and {away_in!r}"}), 404

        payload = dashboard._match_payload(m, state.predictor, state.final_elo)
        try:
            mkt = market.match(m["home"], m["away"])
        except market.MarketError:
            mkt = None
        if mkt:
            payload["market"] = mkt

        pred = state.predictor.predict(m["home"], m["away"], MATCH_NEUTRAL, MATCH_WEIGHT)
        try:
            ledger.log_prediction(m, pred, state.predictor, state.results, "server", market_probs=mkt)
        except Exception as exc:  # ledger logging must never break a prediction response
            print(f"[server] couldn't log prediction ({exc!r})")
        return jsonify(payload)

    @app.get("/api/bracket")
    def api_bracket():
        return jsonify(dashboard._bracket_payload(state.bracket))

    @app.get("/api/sim")
    def api_sim():
        return jsonify(state.championship)

    @app.get("/api/edge")
    def api_edge():
        model_probs = {row["team"]: row["p_champion"] for row in state.championship}
        try:
            market_probs = market.tournament_winner()
        except market.MarketError as exc:
            return jsonify({"edge": [], "error": str(exc)})
        common = sorted(set(model_probs) & set(market_probs))
        edge = sorted(
            ({"team": t, "model": dashboard._round(model_probs[t]), "market": dashboard._round(market_probs[t]),
              "edge": dashboard._round(model_probs[t] - market_probs[t])} for t in common),
            key=lambda r: abs(r["edge"]), reverse=True,
        )
        return jsonify({"edge": edge, "error": None})

    @app.get("/api/report-card")
    def api_report_card():
        card = ledger.report_card(state.results)
        card["asof"] = str(state.predictor.asof.date())
        card["alpha"] = state.predictor.alpha
        card["alpha_base"] = state.predictor.alpha_base
        card["results_max_date"] = str(state.results["date"].max().date())
        card["built_at"] = state.built_at
        card["feedback"] = state.predictor.feedback_info
        return jsonify(card)

    @app.post("/api/refresh")
    def api_refresh():
        if not state.lock.acquire(blocking=False):
            return jsonify({"error": "a refresh is already running"}), 409
        try:
            alpha_before = state.predictor.alpha
            n_scored_before = ledger.report_card(state.results)["n_scored"]
            build_state(state, force_refresh=True)
            alpha_after = state.predictor.alpha
            n_scored_after = ledger.report_card(state.results)["n_scored"]
        finally:
            state.lock.release()
        return jsonify({
            "results_max_date": str(state.results["date"].max().date()),
            "alpha_before": alpha_before, "alpha_after": alpha_after,
            "n_scored_before": n_scored_before, "n_scored_after": n_scored_after,
            "built_at": state.built_at,
        })

    return app


def start_auto_refresh(state, interval_min):
    """Spawn a daemon thread that force-refreshes results + rebuilds state
    every `interval_min` minutes, so a long-running server keeps itself
    current instead of freezing at startup state. Uses the same lock as
    /api/refresh (skips a tick if a manual refresh is in flight). interval_min
    <= 0 disables it."""
    if not interval_min or interval_min <= 0:
        return None

    def loop():
        while True:
            time.sleep(interval_min * 60)
            if not state.lock.acquire(blocking=False):
                continue  # a manual refresh is running; skip this tick
            try:
                print("[server] auto-refresh tick ...", flush=True)
                build_state(state, force_refresh=True)
                print(f"[server] auto-refresh done; built_at={state.built_at}", flush=True)
            except Exception as exc:  # a bad refresh must not kill the timer
                print(f"[server] auto-refresh failed ({exc!r}); will retry next tick.", flush=True)
            finally:
                state.lock.release()

    t = threading.Thread(target=loop, name="auto-refresh", daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    state = build_state()
    start_auto_refresh(state, interval_min=30)
    app = create_app(state)
    app.run(host="127.0.0.1", port=8000)
