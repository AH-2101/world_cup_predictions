"""wcpred.cli — argparse entry point.

Subcommands:
  match     enriched single-match prediction (XGBoost + Dixon-Coles ensemble)
  today     run `match` for every resolvable fixture on a given date
  sim       Monte Carlo the remaining knockout bracket -> championship odds
  bracket   print the resolved bracket state + per-round survival probabilities
  backtest  walk-forward accuracy/log-loss/Brier report (wcpred.backtest)
  edge      model vs. Polymarket odds disagreement table
  dashboard write a self-contained dashboard.html (no server) with all of the above

`python predict_today.py "A" "B"` (no subcommand) remains a shorthand for
`match "A" "B"` — see `main()`.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from wcpred.data import load_results, per_team_long
from wcpred.features import build_dataset, ELO_BASE
from wcpred.model_wdl import MATCH_WEIGHT, MATCH_NEUTRAL
from wcpred.fixtures import (
    FIXTURES_PATH, FIRST_KNOCKOUT_DATE,
    find_fixture, list_team_names, map_fixture_name, parse_bracket,
    fixture_meta_by_mno, resolve_slots_for_date, find_bracket_match,
)
from wcpred.viz import make_chart, tag_match, scoreline_heatmap, championship_bar, road_to_final
from wcpred import ensemble
from wcpred import simulate
from wcpred import market

COMMANDS = {"match", "today", "sim", "bracket", "backtest", "edge", "dashboard"}

# Monte Carlo trial count for `sim`/`bracket`/`edge`. The plan's default of
# 20000 is accurate but a few seconds slower than feels snappy for a CLI that
# also has to train XGBoost + fit Dixon-Coles first; 5000 keeps interactive
# runs quick while still giving stable-to-a-fraction-of-a-percent odds. Pass
# --n 20000 (or higher) for a slower, slightly more precise run.
SIM_DEFAULT_N = 5000


# ── shared helpers ──────────────────────────────────────────────────────────────
def get_teams_from_args(argv=None):
    """Two team names from the command line, or ask for them interactively."""
    argv = sys.argv if argv is None else argv
    if len(argv) >= 3:
        return argv[1], argv[2]
    print("Enter the two teams to predict (e.g. Saudi Arabia / Uruguay).")
    a = input("  Team 1: ").strip()
    b = input("  Team 2: ").strip()
    return a, b


def _out_dir(date):
    out_dir = os.path.join("predictions", str(date))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _report_match(m, predictor, final_elo):
    """Print the enriched single-match report and save both charts
    (the original bar chart + the new Dixon-Coles scoreline heatmap)."""
    out = predictor.predict(m["home"], m["away"], MATCH_NEUTRAL, MATCH_WEIGHT)
    p_home, p_draw, p_away = out["p_home"], out["p_draw"], out["p_away"]

    outcomes = [(m["home_disp"], p_home), ("Draw", p_draw), (m["away_disp"], p_away)]
    pick, conf = max(outcomes, key=lambda x: x[1])
    he, ae = final_elo.get(m["home"], ELO_BASE), final_elo.get(m["away"], ELO_BASE)
    tag = tag_match(conf, p_home, p_away, he, ae)

    out_dir = _out_dir(m["date"])
    bar_chart = make_chart(m, p_home, p_draw, p_away, m["date"], out_dir)
    heatmap = scoreline_heatmap(m["home"], m["away"], out["score_matrix"], m, out_dir)

    print("\n" + "=" * 60)
    print(f"  {m['home_disp']} vs {m['away_disp']}")
    print(f"  {m['date']}  ·  {m['group']}  ·  {m['stadium']}")
    print("=" * 60)
    print(f"  {m['home_disp']:<22} win   {p_home*100:>5.1f}%")
    print(f"  {'Draw':<22}       {p_draw*100:>5.1f}%")
    print(f"  {m['away_disp']:<22} win   {p_away*100:>5.1f}%")
    print("-" * 60)
    print(f"  PICK: {pick}  ({conf*100:.1f}%)   [{tag}]")
    print("=" * 60)
    print(f"  Bar chart saved       -> {bar_chart}")
    print(f"  Scoreline heatmap     -> {heatmap}\n")


def _fixture_meta_by_mno(fixtures_path=FIXTURES_PATH):
    return fixture_meta_by_mno(fixtures_path)


def _slots_for_date(results, date_str):
    return resolve_slots_for_date(results, date_str)


def _find_bracket_match(team_a, team_b, results):
    return find_bracket_match(team_a, team_b, results)


# ── match ────────────────────────────────────────────────────────────────────────
def run_match(team_a, team_b):
    print("\nLoading data + building features ...")
    results = load_results()
    dataset, final_elo = build_dataset(results)
    valid_teams = set(results["home_team"]) | set(results["away_team"])
    long = per_team_long(results)

    m = find_fixture(team_a, team_b) or _find_bracket_match(team_a, team_b, results)
    if m is None:
        print(f"\n  Couldn't find a World Cup match between '{team_a}' and '{team_b}'.")
        print("  Check spelling. Teams in the tournament:")
        print("   " + ", ".join(list_team_names()))
        return
    if m["home"] not in valid_teams or m["away"] not in valid_teams:
        print("\n  That match isn't predictable yet (a team is still a placeholder, e.g. a knockout slot).")
        return

    match_date = m["date"]
    print(f"Building ensemble predictor (XGBoost + Dixon-Coles, data up to {match_date}) ...")
    predictor = ensemble.build(dataset, long, final_elo, match_date)

    _report_match(m, predictor, final_elo)


def _cmd_match(args):
    if args.team_a and args.team_b:
        team_a, team_b = args.team_a, args.team_b
    else:
        team_a, team_b = get_teams_from_args(["wcpred"])
    run_match(team_a, team_b)


# ── today ────────────────────────────────────────────────────────────────────────
def run_today(date=None):
    date = date or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    print(f"\nLoading data + building features (slate date {date}) ...")
    results = load_results()
    dataset, final_elo = build_dataset(results)
    long = per_team_long(results)

    slots = _slots_for_date(results, date)
    if not slots:
        print(f"\n  No resolvable World Cup fixtures found for {date}.\n")
        return

    print(f"Building ensemble predictor (data up to {date}) ...")
    predictor = ensemble.build(dataset, long, final_elo, date)

    print(f"\n{len(slots)} match(es) on {date}:")
    for m in slots:
        _report_match(m, predictor, final_elo)


def _cmd_today(args):
    run_today(args.date)


# ── sim ──────────────────────────────────────────────────────────────────────────
def _build_predictor_today(results):
    dataset, final_elo = build_dataset(results)
    long = per_team_long(results)
    asof = pd.Timestamp.today().normalize()
    return ensemble.build(dataset, long, final_elo, asof), asof


def run_sim(n=SIM_DEFAULT_N):
    print("Loading live results feed ...")
    results = load_results()
    print("Parsing knockout bracket (Match 73-104) ...")
    bracket = parse_bracket(FIXTURES_PATH, results)
    print("Building ensemble predictor (XGBoost + Dixon-Coles, as of today) ...")
    predictor, asof = _build_predictor_today(results)

    print(f"Simulating {n} tournament completions ...")
    table = simulate.run(bracket, predictor, n=n, seed=42)

    print(f"\n=== Championship odds (top 15 of {len(table)} still-alive teams) ===\n")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.head(15).to_string(index=False))
    total = table["p_champion"].sum()
    print(f"\nSum of p_champion across all alive teams: {total:.4f}  (expect ~1.0)")

    out_dir = _out_dir(asof.date())
    team_probs = dict(zip(table["team"], table["p_champion"]))
    chart = championship_bar(team_probs, out_dir)
    print(f"Championship-odds chart saved -> {chart}\n")
    return table


def _cmd_sim(args):
    run_sim(n=args.n)


# ── bracket ──────────────────────────────────────────────────────────────────────
def run_bracket(n=SIM_DEFAULT_N):
    results = load_results()
    bracket = parse_bracket(FIXTURES_PATH, results)

    print("\n=== Knockout bracket state (Match 73-104) ===\n")
    for slot in bracket:
        home = slot["resolved_home"] or "TBD"
        away = slot["resolved_away"] or "TBD"
        if slot["played"]:
            status = f"FINAL -- winner: {slot['winner']}"
        elif slot["resolved_home"] and slot["resolved_away"]:
            status = "scheduled, not yet played"
        else:
            status = "pending (waiting on earlier results)"
        print(f"  Match {slot['match_number']:<3} [{slot['round']:<3}] {slot['date']}  "
              f"{home:<20} v {away:<20}  -- {status}")

    try:
        predictor, asof = _build_predictor_today(results)
        print(f"\nSimulating {n} tournament completions for survival probabilities ...")
        table = simulate.run(bracket, predictor, n=n, seed=42)
        out_dir = _out_dir(asof.date())
        chart = road_to_final(table, out_dir)
        print(f"Road-to-final survival chart saved -> {chart}\n")
        print("(top 8 teams by championship odds shown on the chart; full table via `wcpred.cli sim`)")
    except Exception as exc:
        print(f"\n[bracket] couldn't render the survival-probability chart ({exc!r}); "
              "bracket state above is still valid on its own.")


def _cmd_bracket(args):
    run_bracket(n=args.n)


# ── backtest ─────────────────────────────────────────────────────────────────────
def _cmd_backtest(args):
    try:
        from wcpred import backtest
    except ImportError as exc:
        print(f"\n[backtest] wcpred/backtest.py isn't available yet ({exc}). "
              "Run this again once it's landed.\n")
        return

    print("Loading data + building features ...")
    results = load_results()
    dataset, final_elo = build_dataset(results)
    long = per_team_long(results)

    print("Running walk-forward backtest ...")
    table = backtest.run(dataset, long, final_elo)

    print("\n=== Backtest metrics ===\n")
    print(table.to_string(index=False))
    print()


# ── edge ─────────────────────────────────────────────────────────────────────────
def run_edge(n=SIM_DEFAULT_N):
    print("Pulling live Polymarket tournament-winner odds ...")
    market_probs = market.tournament_winner()

    print("Loading data + building features ...")
    results = load_results()
    bracket = parse_bracket(FIXTURES_PATH, results)
    predictor, asof = _build_predictor_today(results)

    print(f"Simulating {n} tournament completions for model odds ...")
    table = simulate.run(bracket, predictor, n=n, seed=42)
    model_probs = dict(zip(table["team"], table["p_champion"]))

    common = sorted(set(model_probs) & set(market_probs))
    rows = [(t, model_probs[t], market_probs[t], model_probs[t] - market_probs[t]) for t in common]
    rows.sort(key=lambda r: abs(r[3]), reverse=True)

    print(f"\n=== Model vs. Polymarket -- tournament winner ({len(rows)} teams priced by both) ===\n")
    print(f"  {'Team':<24}{'Model':>10}{'Market':>10}{'Edge':>10}")
    print("  " + "-" * 54)
    for t, mp, kp, e in rows:
        print(f"  {t:<24}{mp*100:>9.2f}%{kp*100:>9.2f}%{e*100:>+9.2f}%")

    # optional: per-match edge for today's resolvable fixtures, if Polymarket has them
    date = asof.strftime("%Y-%m-%d")
    slots = _slots_for_date(results, date)
    match_rows = []
    for m in slots:
        mkt = market.match(m["home"], m["away"])
        if mkt is None:
            continue
        pred = predictor.predict(m["home"], m["away"], MATCH_NEUTRAL, MATCH_WEIGHT)
        match_rows.append((m["home_disp"], m["away_disp"], pred["p_home"], mkt["p_home"]))

    if match_rows:
        print(f"\n=== Model vs. Polymarket -- today's fixtures ({date}) ===\n")
        print(f"  {'Match':<34}{'Model (home win)':>18}{'Market (home win)':>19}")
        print("  " + "-" * 71)
        for home, away, mp, kp in match_rows:
            print(f"  {home + ' v ' + away:<34}{mp*100:>17.1f}%{kp*100:>18.1f}%")
    print()


def _cmd_edge(args):
    run_edge(n=args.n)


# ── dashboard ────────────────────────────────────────────────────────────────────
def run_dashboard(n=SIM_DEFAULT_N):
    from wcpred import dashboard

    print("Building dashboard data (today's matches, sim, bracket, live Polymarket odds) ...")
    data = dashboard.build_data(sim_n=n)
    date_str = data["generated_at"]
    out_path = os.path.join(_out_dir(date_str), "dashboard.html")
    path = dashboard.render(data, out_path)

    print(f"\nDashboard written -> {path}")
    print(f"  {len(data['today_matches'])} today's match(es), "
          f"{len(data['championship'])} teams in championship odds, "
          f"{len(data['edge'])} teams priced by both model and market")
    if data.get("market_error"):
        print(f"  (Polymarket odds unavailable this run: {data['market_error']})")
    print(f"\nOpen it directly in a browser -- no server needed:\n  file://{os.path.abspath(path)}\n")


def _cmd_dashboard(args):
    run_dashboard(n=args.n)


# ── argparse plumbing ──────────────────────────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(prog="wcpred", description="World Cup 2026 match predictor")
    sub = parser.add_subparsers(dest="command")

    p_match = sub.add_parser("match", help="Predict a single match (positional two teams, or interactive prompt)")
    p_match.add_argument("team_a", nargs="?", default=None)
    p_match.add_argument("team_b", nargs="?", default=None)
    p_match.set_defaults(func=_cmd_match)

    p_today = sub.add_parser("today", help="Predict every resolvable fixture on a given date (default: today)")
    p_today.add_argument("--date", default=None, help="YYYY-MM-DD (default: today's real-world date)")
    p_today.set_defaults(func=_cmd_today)

    p_sim = sub.add_parser("sim", help="Monte Carlo the remaining knockout bracket -> championship odds")
    p_sim.add_argument("--n", type=int, default=SIM_DEFAULT_N, help=f"number of simulations (default {SIM_DEFAULT_N})")
    p_sim.set_defaults(func=_cmd_sim)

    p_bracket = sub.add_parser("bracket", help="Print the resolved bracket state + survival probabilities")
    p_bracket.add_argument("--n", type=int, default=SIM_DEFAULT_N, help=f"number of simulations (default {SIM_DEFAULT_N})")
    p_bracket.set_defaults(func=_cmd_bracket)

    p_backtest = sub.add_parser("backtest", help="Walk-forward accuracy/log-loss/Brier report")
    p_backtest.set_defaults(func=_cmd_backtest)

    p_edge = sub.add_parser("edge", help="Model vs. Polymarket odds disagreement table")
    p_edge.add_argument("--n", type=int, default=SIM_DEFAULT_N, help=f"number of simulations (default {SIM_DEFAULT_N})")
    p_edge.set_defaults(func=_cmd_edge)

    p_dash = sub.add_parser("dashboard", help="Write a self-contained dashboard.html (no server)")
    p_dash.add_argument("--n", type=int, default=SIM_DEFAULT_N, help=f"number of simulations (default {SIM_DEFAULT_N})")
    p_dash.set_defaults(func=_cmd_dashboard)

    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()

    # Back-compat: `python predict_today.py "A" "B"` / no-args interactive prompt
    # should behave like `match` even without the subcommand keyword.
    if not argv or (argv and argv[0] not in COMMANDS | {"-h", "--help"}):
        team_a, team_b = get_teams_from_args(["wcpred"] + argv)
        run_match(team_a, team_b)
        return

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
