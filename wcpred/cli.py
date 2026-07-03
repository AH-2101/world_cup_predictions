"""wcpred.cli — argparse entry point.

Only the `match` subcommand is implemented in this wave (Wave 0); it reproduces
the exact behavior of the original single-file predict_today.py `main()`.
`today` / `sim` / `bracket` / `backtest` / `edge` subcommands land in later
waves (see plan.md) — the subparsers are wired below so those agents can just
add their `set_defaults(func=...)` without touching argument-parsing plumbing.
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

from wcpred.data import load_results, per_team_long
from wcpred.features import build_dataset, ELO_BASE
from wcpred.model_wdl import (
    split_by_date, train_model, predict_symmetric,
    TRAIN_START, VAL_START, MATCH_WEIGHT, MATCH_NEUTRAL,
)
from wcpred.fixtures import find_fixture, list_team_names
from wcpred.viz import make_chart, tag_match

import os


def get_teams_from_args(argv=None):
    """Two team names from the command line, or ask for them interactively."""
    argv = sys.argv if argv is None else argv
    if len(argv) >= 3:
        return argv[1], argv[2]
    print("Enter the two teams to predict (e.g. Saudi Arabia / Uruguay).")
    a = input("  Team 1: ").strip()
    b = input("  Team 2: ").strip()
    return a, b


def run_match(team_a, team_b):
    print("\nLoading data + building features ...")
    results = load_results()
    dataset, final_elo = build_dataset(results)
    valid_teams = set(results["home_team"]) | set(results["away_team"])
    long = per_team_long(results)

    m = find_fixture(team_a, team_b)
    if m is None:
        print(f"\n  Couldn't find a World Cup match between '{team_a}' and '{team_b}'.")
        print("  Check spelling. Teams in the tournament:")
        print("   " + ", ".join(list_team_names()))
        return
    if m["home"] not in valid_teams or m["away"] not in valid_teams:
        print(f"\n  That match isn't predictable yet (a team is still a placeholder, e.g. a knockout slot).")
        return

    match_date = m["date"]
    print(f"Training model (data up to {match_date} ...")
    train, val = split_by_date(dataset, TRAIN_START, VAL_START, match_date)
    model, X_val, y_val = train_model(train, val)

    p_home, p_draw, p_away = predict_symmetric(
        model, long, final_elo, m["home"], m["away"], match_date, MATCH_NEUTRAL, MATCH_WEIGHT)
    outcomes = [(m["home_disp"], p_home), ("Draw", p_draw), (m["away_disp"], p_away)]
    pick, conf = max(outcomes, key=lambda x: x[1])
    he, ae = final_elo.get(m["home"], ELO_BASE), final_elo.get(m["away"], ELO_BASE)
    tag = tag_match(conf, p_home, p_away, he, ae)

    out_dir = os.path.join("predictions", str(match_date))
    os.makedirs(out_dir, exist_ok=True)
    chart = make_chart(m, p_home, p_draw, p_away, match_date, out_dir)

    # print the single result
    print("\n" + "=" * 60)
    print(f"  {m['home_disp']} vs {m['away_disp']}")
    print(f"  {match_date}  ·  {m['group']}  ·  {m['stadium']}")
    print("=" * 60)
    print(f"  {m['home_disp']:<22} win   {p_home*100:>5.1f}%")
    print(f"  {'Draw':<22}       {p_draw*100:>5.1f}%")
    print(f"  {m['away_disp']:<22} win   {p_away*100:>5.1f}%")
    print("-" * 60)
    print(f"  PICK: {pick}  ({conf*100:.1f}%)   [{tag}]")
    print("=" * 60)
    print(f"  Chart saved -> {chart}\n")


def _cmd_match(args):
    if args.team_a and args.team_b:
        team_a, team_b = args.team_a, args.team_b
    else:
        team_a, team_b = get_teams_from_args(["wcpred"])
    run_match(team_a, team_b)


def build_parser():
    parser = argparse.ArgumentParser(prog="wcpred", description="World Cup 2026 match predictor")
    sub = parser.add_subparsers(dest="command")

    p_match = sub.add_parser("match", help="Predict a single match (positional two teams, or interactive prompt)")
    p_match.add_argument("team_a", nargs="?", default=None)
    p_match.add_argument("team_b", nargs="?", default=None)
    p_match.set_defaults(func=_cmd_match)

    # Later waves add these subcommands (see plan.md WAVE 1-3):
    #   today, sim, bracket, backtest, edge
    sub.add_parser("today", help="(later wave) predict every match on today's slate")
    sub.add_parser("sim", help="(later wave) Monte Carlo the bracket -> championship odds")
    sub.add_parser("bracket", help="(later wave) render the resolved bracket + survival probabilities")
    sub.add_parser("backtest", help="(later wave) walk-forward accuracy/log-loss/Brier report")
    sub.add_parser("edge", help="(later wave) model vs. Polymarket odds disagreement table")

    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()

    # Back-compat: `python predict_today.py "A" "B"` / no-args interactive prompt
    # should behave like `match` even without the subcommand keyword.
    if not argv or (argv and argv[0] not in {"match", "today", "sim", "bracket", "backtest", "edge", "-h", "--help"}):
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
