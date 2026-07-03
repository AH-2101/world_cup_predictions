"""wcpred.market — free Polymarket Gamma API odds (no auth) + de-vig.

Two public entry points, both returning team names normalized to the same
convention as `wcpred.data` (i.e. the names used in `results.csv`):

    tournament_winner() -> {team: p_win}          de-vigged, sums to ~1.0
    match(home, away)   -> {p_home,p_draw,p_away} | None

Polymarket structures the FIFA World Cup 2026 markets as:
  - a single "World Cup Winner" event (slug `world-cup-winner`) with one
    binary Yes/No sub-market per team ("Will <team> win the 2026 FIFA World
    Cup?"); some slots are unfilled placeholders (groupItemTitle "Other" or
    "Team AG"/"Team AH"/... ) and are skipped.
  - one event per fixture, slug pattern `fifwc-<3-letter>-<3-letter>-2026-MM-DD`
    (e.g. `fifwc-prt-esp-2026-07-06`), with a binary sub-market per team
    ("Will <team> win on <date>?") plus, when applicable, a binary "Draw"
    sub-market ("Will <A> vs <B> end in a draw?"). We find these via the
    public full-text search endpoint since exact slugs/dates aren't known
    ahead of time.

De-vig: raw Yes-prices sum to slightly over 1.0 (the overround/vig); we
normalize by dividing each by the sum so the set sums to exactly 1.0.
"""

import json
import os
import re
import time

import requests

from . import data as _data

BASE_URL = "https://gamma-api.polymarket.com"
CACHE_DIR = "data_cache"
WINNER_CACHE = os.path.join(CACHE_DIR, "polymarket_winner.json")
MATCH_CACHE = os.path.join(CACHE_DIR, "polymarket_matches.json")
STALE_SECONDS = 6 * 3600  # refetch cache older than 6 hours
WINNER_EVENT_SLUG = "world-cup-winner"

# Polymarket-specific spellings not already covered by wcpred.data.NAME_MAP.
# Applied on TOP of data.normalize_country (fixes cases where Polymarket's
# spelling isn't the results.csv raw spelling either).
EXTRA_NAME_MAP = {
    "Turkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}

# Reverse aliases: normalized (results.csv) name -> a Polymarket-style
# spelling worth trying when building a search query / matching a title.
# Only needed where Polymarket's spelling differs from our normalized name.
SEARCH_ALIASES = {
    "United States": "USA",
    "DR Congo": "Congo DR",
    "Cabo Verde": "Cape Verde",
    "Czech Republic": "Czechia",
    "Turkey": "Turkiye",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Curacao": "Curaçao",
    "Ireland": "Republic of Ireland",
}

_PLACEHOLDER_TEAM_RE = re.compile(r"^Team [A-Z]+$")


class MarketError(RuntimeError):
    """Raised when the Polymarket Gamma API can't be reached or returns
    something we don't know how to parse — never silently swallowed."""


# ── name normalization ──────────────────────────────────────────────────────
def normalize_team(name):
    """Normalize a Polymarket team/country spelling to the results.csv
    convention used throughout wcpred (extends data.NAME_MAP)."""
    if name in EXTRA_NAME_MAP:
        return EXTRA_NAME_MAP[name]
    return _data.normalize_country(name)


def _search_variants(name):
    """Candidate spellings to try when searching/matching Polymarket titles."""
    variants = {name}
    if name in SEARCH_ALIASES:
        variants.add(SEARCH_ALIASES[name])
    return variants


# ── tiny JSON cache helpers ──────────────────────────────────────────────────
def _load_cache(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            blob = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    return blob


def _cache_is_fresh(blob):
    return bool(blob) and (time.time() - blob.get("fetched_at", 0)) < STALE_SECONDS


def _save_cache(path, payload):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh)


# ── low-level Gamma API access ──────────────────────────────────────────────
def _get(path, params=None):
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise MarketError(f"Polymarket Gamma API request failed: {url} params={params}: {exc}") from exc
    except ValueError as exc:
        raise MarketError(f"Polymarket Gamma API returned non-JSON from {url}: {exc}") from exc


def _yes_price(market):
    """Extract the de-serialized (raw, still-vigged) Yes-price of a market."""
    outcomes = json.loads(market["outcomes"])
    prices = json.loads(market["outcomePrices"])
    try:
        idx = outcomes.index("Yes")
    except ValueError:
        return None
    try:
        return float(prices[idx])
    except (IndexError, TypeError, ValueError):
        return None


# ── tournament winner market ────────────────────────────────────────────────
def _fetch_winner_event():
    events = _get("/events", params={"slug": WINNER_EVENT_SLUG})
    if not isinstance(events, list) or not events:
        raise MarketError(f"Polymarket returned no event for slug={WINNER_EVENT_SLUG!r}")
    return events[0]


def tournament_winner(force_refresh=False):
    """FIFA World Cup 2026 tournament-winner market, de-vigged.

    Returns {team_name: probability} with team_name normalized to the
    wcpred.data convention. Probabilities sum to ~1.0.
    """
    cache = _load_cache(WINNER_CACHE)
    event = None
    if not force_refresh and _cache_is_fresh(cache):
        event = cache["event"]
    else:
        try:
            event = _fetch_winner_event()
            _save_cache(WINNER_CACHE, {"fetched_at": time.time(), "event": event})
        except MarketError:
            if cache and "event" in cache:
                event = cache["event"]  # stale cache beats a hard failure
            else:
                raise

    markets = event.get("markets", [])
    if not markets:
        raise MarketError("Polymarket world-cup-winner event has no sub-markets")

    raw = {}
    for m in markets:
        title = m.get("groupItemTitle")
        if not title or title == "Other" or _PLACEHOLDER_TEAM_RE.match(title):
            continue
        if "outcomePrices" not in m or "outcomes" not in m:
            continue
        price = _yes_price(m)
        if price is None:
            continue
        team = normalize_team(title)
        raw[team] = raw.get(team, 0.0) + price

    total = sum(raw.values())
    if total <= 0:
        raise MarketError("Polymarket world-cup-winner market parsed to zero total probability")

    return {team: p / total for team, p in raw.items()}


# ── per-match market ─────────────────────────────────────────────────────────
def _candidate_queries(home, away):
    queries = []
    for h in _search_variants(home):
        for a in _search_variants(away):
            queries.append(f"{h} vs {a}")
            queries.append(f"{a} vs {h}")
    # de-dupe while preserving order
    seen = set()
    out = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _title_matches(title, home, away):
    t = title.lower()
    home_ok = any(v.lower() in t for v in _search_variants(home))
    away_ok = any(v.lower() in t for v in _search_variants(away))
    return home_ok and away_ok


def _find_match_event(home, away):
    for q in _candidate_queries(home, away):
        try:
            result = _get("/public-search", params={"q": q, "limit_per_type": 10})
        except MarketError:
            continue
        for ev in result.get("events", []):
            slug = ev.get("slug", "")
            title = ev.get("title", "")
            if slug.startswith("fifwc-") and _title_matches(title, home, away):
                return ev
    return None


def _match_cache_key(home, away):
    return "|".join(sorted([home, away]))


def match(home, away, force_refresh=False):
    """Polymarket odds for a specific World Cup 2026 fixture, de-vigged.

    Returns {"p_home":.., "p_draw":.., "p_away":..} or None if no findable
    market exists for this pairing, or if only a binary (no-draw) market
    exists (we never fabricate a draw probability).
    """
    key = _match_cache_key(home, away)
    cache = _load_cache(MATCH_CACHE) or {"fetched_at": 0, "events": {}}
    entry = cache.get("events", {}).get(key)

    event = None
    have_fresh_entry = entry is not None and (time.time() - entry.get("fetched_at", 0)) < STALE_SECONDS
    if not force_refresh and have_fresh_entry:
        event = entry["event"]  # may legitimately be None (cached "not found")
    else:
        try:
            event = _find_match_event(home, away)
            cache.setdefault("events", {})[key] = {"fetched_at": time.time(), "event": event}
            cache["fetched_at"] = time.time()
            _save_cache(MATCH_CACHE, cache)
        except MarketError:
            if entry is not None:
                event = entry["event"]
            else:
                raise

    if not event:
        return None

    home_price = away_price = draw_price = None
    for m in event.get("markets", []):
        title = (m.get("groupItemTitle") or "")
        if "outcomePrices" not in m or "outcomes" not in m:
            continue
        price = _yes_price(m)
        if price is None:
            continue
        if title.lower().startswith("draw"):
            draw_price = price
        elif any(v.lower() == title.lower() for v in _search_variants(home)):
            home_price = price
        elif any(v.lower() == title.lower() for v in _search_variants(away)):
            away_price = price

    if home_price is None or away_price is None:
        return None
    if draw_price is None:
        # Only a binary win/lose market exists for this fixture — don't
        # fabricate a draw probability.
        return None

    total = home_price + draw_price + away_price
    if total <= 0:
        return None
    return {
        "p_home": home_price / total,
        "p_draw": draw_price / total,
        "p_away": away_price / total,
    }


if __name__ == "__main__":
    winner = tournament_winner()
    ranked = sorted(winner.items(), key=lambda kv: kv[1], reverse=True)
    print(f"{'Team':<28} {'P(win)':>8}")
    print("-" * 37)
    for team, p in ranked:
        print(f"{team:<28} {p:>8.4f}")
    print("-" * 37)
    print(f"{'sum':<28} {sum(winner.values()):>8.4f}")
