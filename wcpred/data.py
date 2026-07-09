"""wcpred.data — fetch/load/normalize historical results + per-team reshaping."""

import os
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

CACHE_DIR = "data_cache"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
RESULTS_STALE_SECONDS = 6 * 3600  # refetch cache older than 6 hours, mirrors wcpred.market's TTL

# The 2026 World Cup is played across North American venues (US Eastern to
# Pacific). Anchor "today" to the westmost tournament timezone so the current
# match-day stays aligned with when games actually finish. Using UTC instead
# rolls the date to "tomorrow" the instant it passes midnight UTC — which is
# mid-evening in the Americas, so a game still in progress (e.g. a 07-06 R16
# tie) would wrongly drop off "today's" slate and the app would look a day ahead.
TOURNAMENT_TZ = ZoneInfo("America/Los_Angeles")


def tournament_today():
    """Normalized (midnight) current date in the tournament's local timezone,
    returned tz-naive to match the rest of the pipeline's timestamps."""
    return pd.Timestamp.now(TOURNAMENT_TZ).normalize().tz_localize(None)

# normalizes the historical results.csv team names
NAME_MAP = {
    "USA": "United States", "Korea Republic": "South Korea",
    "Republic of Ireland": "Ireland", "Türkiye": "Turkey",
    "Cape Verde": "Cabo Verde", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao",
    "Congo DR": "DR Congo", "Congo": "Republic of the Congo",
}

# maps fixtures.csv team names -> the normalized results.csv names
FIXTURE_NAME_MAP = {
    "IR Iran": "Iran", "Korea Republic": "South Korea", "Türkiye": "Turkey",
    "Congo DR": "DR Congo", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao", "USA": "United States",
    "Cape Verde": "Cabo Verde",
}


# ── data loading ────────────────────────────────────────────────────────────────
def fetch_results(force=False):
    """Download results.csv if missing or stale (mtime older than
    RESULTS_STALE_SECONDS), else reuse the cache. A download failure falls
    back to a stale cache with a printed warning rather than a hard error,
    same "stale cache beats hard failure" convention as wcpred.market."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "results.csv")
    exists = os.path.exists(path)
    stale = not exists or force or (time.time() - os.path.getmtime(path)) > RESULTS_STALE_SECONDS
    if stale:
        try:
            resp = requests.get(RESULTS_URL, timeout=120)
            resp.raise_for_status()
            with open(path, "wb") as fh:
                fh.write(resp.content)
        except requests.RequestException as exc:
            if not exists:
                raise
            print(f"[data] results.csv refresh failed ({exc}); using stale cache.")
    return pd.read_csv(path)


def normalize_country(name):
    return NAME_MAP.get(name, name) if isinstance(name, str) else name


def load_results(force=False):
    r = fetch_results(force=force)
    r["home_team"] = r["home_team"].map(normalize_country)
    r["away_team"] = r["away_team"].map(normalize_country)
    r["date"] = pd.to_datetime(r["date"])
    r = r.dropna(subset=["home_score", "away_score"]).copy()
    r["home_score"] = r["home_score"].astype(int)
    r["away_score"] = r["away_score"].astype(int)
    r["neutral"] = r["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    return r.sort_values("date").reset_index(drop=True)


def tournament_weight(name):
    t = str(name).lower()
    if "fifa world cup" in t and "qualif" not in t:
        return 4
    if "qualif" in t:
        return 3
    big = ["uefa nations", "copa america", "afc asian cup", "africa cup",
           "concacaf", "uefa euro", "confederations"]
    if any(tok in t for tok in big):
        return 3
    if "friendly" in t:
        return 1
    return 2


def add_label_and_context(r):
    r = r.copy()
    r["label"] = np.where(r["home_score"] > r["away_score"], 0,
                          np.where(r["home_score"] == r["away_score"], 1, 2))
    r["tournament_weight"] = r["tournament"].map(tournament_weight)
    return r


def per_team_long(r):
    home = pd.DataFrame({"date": r["date"].values, "team": r["home_team"].values,
                         "opp": r["away_team"].values, "gf": r["home_score"].values,
                         "ga": r["away_score"].values, "neutral": r["neutral"].values})
    away = pd.DataFrame({"date": r["date"].values, "team": r["away_team"].values,
                         "opp": r["home_team"].values, "gf": r["away_score"].values,
                         "ga": r["home_score"].values, "neutral": r["neutral"].values})
    long = pd.concat([home, away], ignore_index=True)
    long["result"] = np.where(long["gf"] > long["ga"], 1.0,
                              np.where(long["gf"] == long["ga"], 0.5, 0.0))
    long["gd"] = long["gf"] - long["ga"]
    return long
