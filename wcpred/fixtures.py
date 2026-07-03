"""wcpred.fixtures — resolve user-typed team names against data_cache/fixtures.csv."""

import os
import re

import pandas as pd

from wcpred.data import CACHE_DIR, FIXTURE_NAME_MAP, fetch_results, normalize_country

FIXTURES_PATH = os.path.join(CACHE_DIR, "fixtures.csv")


# ── fixtures ──────────────────────────────────────────────────────────────────
def map_fixture_name(name):
    name = name.strip()
    return FIXTURE_NAME_MAP.get(name, name)


def _side_matches(user_input, raw_name):
    """True if the user's typed team matches a fixture side (by raw or mapped name)."""
    u = user_input.strip().lower()
    return u in {raw_name.strip().lower(), map_fixture_name(raw_name).strip().lower()}


def find_fixture(team_a, team_b):
    """Find the single fixture for the two named teams (order doesn't matter)."""
    fx = pd.read_csv(FIXTURES_PATH)
    for _, row in fx.iterrows():
        if " v " not in str(row["teams"]):
            continue
        left, right = [p.strip() for p in str(row["teams"]).split(" v ")]
        forward = _side_matches(team_a, left) and _side_matches(team_b, right)
        reverse = _side_matches(team_a, right) and _side_matches(team_b, left)
        if forward or reverse:
            return {"match": row.get("match_number", ""), "group": row.get("group", ""),
                    "stadium": row.get("stadium", ""), "date": row.get("date_dt", ""),
                    "home_disp": left, "away_disp": right,
                    "home": map_fixture_name(left), "away": map_fixture_name(right)}
    return None


def list_team_names():
    fx = pd.read_csv(FIXTURES_PATH)
    names = set()
    for t in fx["teams"]:
        if " v " in str(t):
            for p in str(t).split(" v "):
                p = p.strip()
                if not any(w in p.lower() for w in ["winner", "runner", "third", "place", "group"]):
                    names.add(p)
    return sorted(names)


def fixture_meta_by_mno(fixtures_path=FIXTURES_PATH):
    """{match_number(int): fixtures.csv row} for every knockout row (73-104)."""
    fx = pd.read_csv(fixtures_path)
    fx["mno"] = fx["match_number"].str.replace("Match ", "", regex=False).astype(int)
    return {int(r["mno"]): r for _, r in fx.iterrows()}


def resolve_slots_for_date(results, date_str, fixtures_path=FIXTURES_PATH):
    """Resolvable (home, away, group, stadium, date, match_number) dicts for
    every fixture on `date_str`. Knockout dates (>= FIRST_KNOCKOUT_DATE) are
    resolved via the bracket parser, since fixtures.csv itself still shows
    generic placeholder text ("Group J winners v Group H runners-up") for
    those slots even once the live results feed knows the real teams. Group
    stage dates are resolved directly from fixtures.csv (real team names
    already appear there). Shared by wcpred.cli and wcpred.dashboard."""
    date_ts = pd.Timestamp(date_str)
    meta = fixture_meta_by_mno(fixtures_path)
    out = []

    if date_ts >= pd.Timestamp(FIRST_KNOCKOUT_DATE):
        bracket = parse_bracket(fixtures_path, results)
        for slot in bracket:
            if pd.Timestamp(slot["date"]) != date_ts:
                continue
            if not slot["resolved_home"] or not slot["resolved_away"]:
                continue
            row = meta.get(slot["match_number"], {})
            group_val = row.get("group")
            group = group_val if isinstance(group_val, str) and group_val.strip() else slot["round"]
            out.append({
                "match_number": slot["match_number"], "group": group,
                "stadium": row.get("stadium", ""), "date": slot["date"],
                "home_disp": slot["resolved_home"], "away_disp": slot["resolved_away"],
                "home": slot["resolved_home"], "away": slot["resolved_away"],
            })
    else:
        valid_teams = set(results["home_team"]) | set(results["away_team"])
        fx = pd.read_csv(fixtures_path)
        day = fx[fx["date_dt"] == date_str]
        for _, row in day.iterrows():
            if " v " not in str(row["teams"]):
                continue
            left, right = [p.strip() for p in str(row["teams"]).split(" v ")]
            home, away = map_fixture_name(left), map_fixture_name(right)
            if home not in valid_teams or away not in valid_teams:
                continue
            out.append({
                "match_number": row.get("match_number", ""), "group": row.get("group", ""),
                "stadium": row.get("stadium", ""), "date": row.get("date_dt", ""),
                "home_disp": left, "away_disp": right, "home": home, "away": away,
            })
    return out


def find_bracket_match(team_a, team_b, results, fixtures_path=FIXTURES_PATH):
    """Fallback for cli.run_match: fixtures.csv still shows placeholder text
    for knockout rows once the group stage ends, so find_fixture() can't
    resolve an already-decided knockout tie like "Portugal v Spain" by name.
    The bracket parser knows the real teams for those slots — search it
    directly. Shared by wcpred.cli and wcpred.dashboard."""
    a, b = team_a.strip().lower(), team_b.strip().lower()

    def norm(name):
        return {name.strip().lower(), map_fixture_name(name).strip().lower()}

    bracket = parse_bracket(fixtures_path, results)
    for slot in bracket:
        home, away = slot["resolved_home"], slot["resolved_away"]
        if not home or not away:
            continue
        forward = a in norm(home) and b in norm(away)
        reverse = a in norm(away) and b in norm(home)
        if not (forward or reverse):
            continue
        row = fixture_meta_by_mno(fixtures_path).get(slot["match_number"], {})
        group_val = row.get("group")
        group = group_val if isinstance(group_val, str) and group_val.strip() else slot["round"]
        return {"match": slot["match_number"], "group": group,
                "stadium": row.get("stadium", ""), "date": slot["date"],
                "home_disp": home, "away_disp": away, "home": home, "away": away}
    return None


# ── knockout bracket parser (Match 73-104) ─────────────────────────────────────
# It's 2026-07-03: the group stage is finished and R32 is in progress. The live
# martj42 results feed (see wcpred.data.load_results/fetch_results) already
# carries every 2026 FIFA World Cup row from kickoff through the *resolved*
# knockout matchups (Match 73-94 -- R32 and six of the eight R16 games), with
# real scores where already played and NaN scores where not. Matches 95-104
# aren't in the feed at all yet, because their participants aren't determined
# until earlier knockout games finish -- those slots are parsed straight out of
# fixtures.csv's "Winner match N" / "Runner-up match N" text instead.
#
# Design note: rather than re-deriving FIFA's official (and fairly involved)
# best-8-of-12 third-place group -> bracket-slot lookup table, we take the live
# feed's already-resolved Match 73-94 team names as ground truth (it has
# already done that resolution correctly) and match each fixtures.csv knockout
# row to its results-feed row by date order -- verified 1:1 per date. That's
# the "simplest and correct" option this parser was told it's fine to take.
# `compute_group_standings` below is still implemented for real (points / goal
# difference / goals scored tiebreakers) and used to label group winners /
# runners-up; it's not extended to the third-place combination table.
_REF_RE = re.compile(r"(winner|runner-up)\s+match\s+(\d+)", re.IGNORECASE)

KNOCKOUT_ROUNDS = {}
KNOCKOUT_ROUNDS.update({n: "R32" for n in range(73, 89)})
KNOCKOUT_ROUNDS.update({n: "R16" for n in range(89, 97)})
KNOCKOUT_ROUNDS.update({n: "QF" for n in range(97, 101)})
KNOCKOUT_ROUNDS.update({101: "SF", 102: "SF", 103: "3rd", 104: "F"})

FIRST_KNOCKOUT_DATE = "2026-06-28"  # Match 73


class Bracket:
    """The 32 knockout slots (Match 73-104).

    `slots[match_number]` is a dict with:
      match_number, round, date, raw_text,
      home_source / away_source  -- ("team", name) if a concrete team is
          already known, else ("winner_of", N) / ("loser_of", N) referencing
          another match's outcome,
      resolved_home / resolved_away  -- concrete team name once known, else
          None (still symbolic, resolved during simulation),
      winner  -- set only once the match has *actually been played* (a fixed
          historical fact, never simulated); None while pending,
      played  -- bool.
    """

    def __init__(self, slots, standings=None):
        self.slots = slots
        self.standings = standings or {}

    def __iter__(self):
        return iter(sorted(self.slots.values(), key=lambda s: s["match_number"]))

    def __getitem__(self, match_number):
        return self.slots[match_number]

    def __len__(self):
        return len(self.slots)


def _group_letter(g):
    return str(g).replace("Group", "").strip()


def _raw_wc2026():
    """Every 2026 FIFA World Cup row (group + knockout), scores included as
    NaN for matches not yet played -- unlike `data.load_results()`, which
    drops unplayed rows entirely via its `dropna`. Needed so knockout slots
    that haven't happened yet stay visible to the bracket parser."""
    raw = fetch_results().copy()
    raw["home_team"] = raw["home_team"].map(normalize_country)
    raw["away_team"] = raw["away_team"].map(normalize_country)
    raw["date"] = pd.to_datetime(raw["date"])
    mask = (raw["tournament"] == "FIFA World Cup") & (raw["date"] >= "2026-06-01")
    return raw[mask].sort_values("date", kind="stable").reset_index(drop=True)


def compute_group_standings(results, fixtures_path=FIXTURES_PATH):
    """Standard 3/1/0-point group tables for the (now finished) 2026 group
    stage (Match 1-72). Tiebreakers: points, goal difference, goals scored --
    a reasonable simplified version of the real rules (no head-to-head
    sub-rule, no fair-play points). Returns {group_letter: [team, ...]}
    ranked 1st..4th (rank 0 = winner, 1 = runner-up, 2/3 = third/fourth)."""
    fx = pd.read_csv(fixtures_path)
    fx["mno"] = fx["match_number"].str.replace("Match ", "", regex=False).astype(int)
    group_stage = fx[fx["mno"] <= 72]

    team_group = {}
    for _, row in group_stage.iterrows():
        if " v " not in str(row["teams"]):
            continue
        g = _group_letter(row["group"])
        left, right = [map_fixture_name(p.strip()) for p in str(row["teams"]).split(" v ")]
        team_group[left] = g
        team_group[right] = g

    stats = {t: {"pts": 0, "gf": 0, "ga": 0} for t in team_group}
    r = results[(results["date"] >= "2026-06-01") &
                (results["home_team"].isin(team_group)) &
                (results["away_team"].isin(team_group)) &
                results["home_score"].notna()]
    for row in r.itertuples():
        h, a, hs, as_ = row.home_team, row.away_team, row.home_score, row.away_score
        if h not in stats or a not in stats:
            continue
        stats[h]["gf"] += hs
        stats[h]["ga"] += as_
        stats[a]["gf"] += as_
        stats[a]["ga"] += hs
        if hs > as_:
            stats[h]["pts"] += 3
        elif as_ > hs:
            stats[a]["pts"] += 3
        else:
            stats[h]["pts"] += 1
            stats[a]["pts"] += 1

    groups = {}
    for team, g in team_group.items():
        groups.setdefault(g, []).append(team)

    standings = {}
    for g, teams in groups.items():
        standings[g] = sorted(
            teams,
            key=lambda t: (-stats[t]["pts"], -(stats[t]["gf"] - stats[t]["ga"]), -stats[t]["gf"], t),
        )
    return standings


def _parse_ref(text):
    """'Winner match 74' / 'Runner-up match 101' -> ("winner_of"|"loser_of", N);
    anything else is treated as an already-concrete team name."""
    m = _REF_RE.search(str(text))
    if not m:
        return ("team", str(text).strip())
    kind = "winner_of" if m.group(1).lower() == "winner" else "loser_of"
    return (kind, int(m.group(2)))


def _try_resolve(source, slots):
    kind, ref = source
    if kind == "team":
        return ref
    target = slots.get(ref)
    if target is None:
        return None
    if kind == "winner_of":
        return target.get("winner")
    if kind == "loser_of":
        winner = target.get("winner")
        if winner is None:
            return None
        return target["resolved_away"] if winner == target["resolved_home"] else target["resolved_home"]
    return None


def parse_bracket(fixtures_path, results):
    """Build the Match 73-104 knockout `Bracket`.

    `results` is expected to be `wcpred.data.load_results()` (used here only
    for the group standings, which need no unplayed-match visibility since
    the group stage is fully complete). For the knockout rows themselves
    (which do need to see not-yet-played 2026 matches) this does its own
    lightweight raw fetch via `fetch_results()`, since `load_results()` drops
    rows with missing scores.
    """
    fx = pd.read_csv(fixtures_path)
    fx["mno"] = fx["match_number"].str.replace("Match ", "", regex=False).astype(int)
    fx = fx.sort_values("mno").reset_index(drop=True)

    standings = compute_group_standings(results, fixtures_path)

    wc = _raw_wc2026()
    knockout_wc = wc[wc["date"] >= FIRST_KNOCKOUT_DATE].sort_values("date", kind="stable").reset_index(drop=True)

    ko_fx = fx[(fx["mno"] >= 73) & (fx["mno"] <= 94)].reset_index(drop=True)
    if len(ko_fx) != len(knockout_wc):
        raise ValueError(
            f"bracket parser: fixtures.csv has {len(ko_fx)} knockout rows for Match 73-94 "
            f"but the live results feed has {len(knockout_wc)} rows from {FIRST_KNOCKOUT_DATE} on "
            "-- feed and fixtures.csv are out of sync"
        )

    slots = {}
    for i, row in ko_fx.iterrows():
        mno = int(row["mno"])
        wrow = knockout_wc.iloc[i]
        home, away = wrow["home_team"], wrow["away_team"]
        hs, as_ = wrow["home_score"], wrow["away_score"]
        played = pd.notna(hs) and pd.notna(as_)
        winner = None
        if played:
            if hs > as_:
                winner = home
            elif as_ > hs:
                winner = away
            # else: a knockout draw -- resolved below via extra-time/penalty inference
        slots[mno] = {
            "match_number": mno, "round": KNOCKOUT_ROUNDS[mno],
            "date": row["date_dt"], "raw_text": row["teams"],
            "home_source": ("team", home), "away_source": ("team", away),
            "resolved_home": home, "resolved_away": away,
            "winner": winner, "played": played,
        }

    # Knockout draws: the raw feed records the 90-minute score, so a tied
    # played match was actually decided by extra time / penalties. We don't
    # have a separate PK model, but we don't need one -- whichever of the two
    # tied teams shows up in an already-resolved later fixture is the one that
    # actually advanced. Fall back to an alphabetical (deterministic, never
    # crashes) tiebreak in the unlikely case neither shows up yet.
    later_teams = set()
    for later_mno in range(89, 97):
        s = slots.get(later_mno)
        if s is not None:
            later_teams.add(s["resolved_home"])
            later_teams.add(s["resolved_away"])
    for slot in slots.values():
        if slot["played"] and slot["winner"] is None:
            h, a = slot["resolved_home"], slot["resolved_away"]
            h_in, a_in = h in later_teams, a in later_teams
            if h_in and not a_in:
                slot["winner"] = h
            elif a_in and not h_in:
                slot["winner"] = a
            else:
                slot["winner"] = min(h, a)

    # Match 95-104: pure "Winner match N" / "Runner-up match N" references,
    # resolved to concrete teams wherever the referenced match is already decided.
    for _, row in fx[fx["mno"] >= 95].iterrows():
        mno = int(row["mno"])
        left, right = [p.strip() for p in str(row["teams"]).split(" v ")]
        h_src, a_src = _parse_ref(left), _parse_ref(right)
        slots[mno] = {
            "match_number": mno, "round": KNOCKOUT_ROUNDS[mno],
            "date": row["date_dt"], "raw_text": row["teams"],
            "home_source": h_src, "away_source": a_src,
            "resolved_home": _try_resolve(h_src, slots),
            "resolved_away": _try_resolve(a_src, slots),
            "winner": None, "played": False,
        }

    return Bracket(slots, standings)
