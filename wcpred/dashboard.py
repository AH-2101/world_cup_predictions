"""wcpred.dashboard — a single self-contained dashboard.html, no server.

Runs the full pipeline once (today's matches, the tournament simulator, the
bracket, and live Polymarket odds), bakes the results into JSON, and writes
one static HTML file with the charts rendered client-side in plain JS. Open
it in any browser; re-run `wcpred.cli dashboard` to refresh the snapshot.
"""

import json
import os

import numpy as np
import pandas as pd

from wcpred.data import load_results, per_team_long, tournament_today
from wcpred.features import build_dataset, ELO_BASE
from wcpred.model_wdl import MATCH_NEUTRAL, MATCH_WEIGHT
from wcpred.fixtures import FIXTURES_PATH, parse_bracket, resolve_slots_for_date
from wcpred.viz import tag_match
from wcpred import ensemble, simulate, market, feedback

ROAD_TOP_N = 8


def _round(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def _match_payload(m, predictor, final_elo):
    out = predictor.predict(m["home"], m["away"], MATCH_NEUTRAL, MATCH_WEIGHT)
    p_home, p_draw, p_away = out["p_home"], out["p_draw"], out["p_away"]
    outcomes = [(m["home_disp"], p_home), ("Draw", p_draw), (m["away_disp"], p_away)]
    pick, conf = max(outcomes, key=lambda x: x[1])
    he, ae = final_elo.get(m["home"], ELO_BASE), final_elo.get(m["away"], ELO_BASE)
    tag = tag_match(conf, p_home, p_away, he, ae)
    matrix = np.asarray(out["score_matrix"])
    return {
        "home": m["home_disp"], "away": m["away_disp"], "group": m["group"],
        "stadium": m["stadium"], "date": str(m["date"]),
        "p_home": _round(p_home), "p_draw": _round(p_draw), "p_away": _round(p_away),
        "pick": pick, "pick_prob": _round(conf), "tag": tag,
        "matrix": [[_round(v) for v in row] for row in matrix.tolist()],
    }


def _bracket_payload(bracket):
    rows = []
    for slot in bracket:
        if slot["played"]:
            status = "final"
        elif slot["resolved_home"] and slot["resolved_away"]:
            status = "scheduled"
        else:
            status = "pending"
        rows.append({
            "match_number": slot["match_number"], "round": slot["round"], "date": str(slot["date"]),
            "home": slot["resolved_home"] or "TBD", "away": slot["resolved_away"] or "TBD",
            "status": status, "winner": slot.get("winner"),
        })
    return rows


def build_data(sim_n=5000, seed=42):
    """Assemble every dashboard section into one JSON-serializable dict."""
    results = load_results()
    dataset, final_elo = build_dataset(results)
    long = per_team_long(results)
    bracket = parse_bracket(FIXTURES_PATH, results)

    asof = tournament_today()
    date_str = asof.strftime("%Y-%m-%d")
    predictor = ensemble.build(dataset, long, final_elo, asof)
    feedback.apply(predictor, results)  # learn from the ledger's track record

    today_matches = [_match_payload(m, predictor, final_elo)
                      for m in resolve_slots_for_date(results, date_str)]

    sim_table = simulate.run(bracket, predictor, n=sim_n, seed=seed)
    championship = [
        {"team": r.team, "p_r16": _round(r.p_r16), "p_qf": _round(r.p_qf),
         "p_sf": _round(r.p_sf), "p_final": _round(r.p_final), "p_champion": _round(r.p_champion)}
        for r in sim_table.itertuples(index=False)
    ]

    market_error = None
    try:
        market_probs = market.tournament_winner()
    except Exception as exc:  # live API call — don't let a network hiccup kill the whole dashboard
        market_probs, market_error = {}, str(exc)

    model_probs = {row["team"]: row["p_champion"] for row in championship}
    common = sorted(set(model_probs) & set(market_probs))
    edge = sorted(
        ({"team": t, "model": _round(model_probs[t]), "market": _round(market_probs[t]),
          "edge": _round(model_probs[t] - market_probs[t])} for t in common),
        key=lambda r: abs(r["edge"]), reverse=True,
    )

    return {
        "generated_at": date_str,
        "today_matches": today_matches,
        "championship": championship,
        "road_to_final": championship[:ROAD_TOP_N],
        "bracket": _bracket_payload(bracket),
        "edge": edge,
        "market_error": market_error,
    }


# ── HTML rendering ──────────────────────────────────────────────────────────────
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>World Cup 2026 -- Prediction Dashboard</title>
<style>
:root {
  --surface-1: #fcfcfb; --surface-2: #ffffff; --border: #e8e8ee;
  --text-primary: #1a1a2e; --text-secondary: #52514e; --text-muted: #8a8a9e;
  --orange: #ff6b18; --blue: #1f6feb; --gray: #9aa0a6;
  --orange-wash: rgba(255,107,24,0.12); --blue-wash: rgba(31,111,235,0.12);
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #121218; --surface-2: #1a1a24; --border: #2c2c3a;
    --text-primary: #f2f2f7; --text-secondary: #c3c2cf; --text-muted: #8a8a9e;
    --orange: #ff8a45; --blue: #5b9bf5; --gray: #9aa0a6;
    --orange-wash: rgba(255,138,69,0.16); --blue-wash: rgba(91,155,245,0.16);
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px; background: var(--surface-1); color: var(--text-primary);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 14px; }
.subtitle { color: var(--text-secondary); margin: 0 0 28px; font-size: 14px; }
.card {
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
  padding: 20px; margin-bottom: 20px;
}
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.match-card { padding: 16px; }
.match-title { font-weight: 600; font-size: 15px; margin-bottom: 2px; }
.match-meta { color: var(--text-muted); font-size: 12.5px; margin-bottom: 12px; }
.wdl-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 13px; }
.wdl-label { width: 92px; flex-shrink: 0; color: var(--text-secondary); text-align: right; }
.wdl-track { flex: 1; height: 20px; background: var(--surface-1); border-radius: 4px; overflow: hidden; position: relative; }
.wdl-fill { height: 100%; border-radius: 4px; }
.wdl-fill.home { background: var(--orange); }
.wdl-fill.draw { background: var(--gray); }
.wdl-fill.away { background: var(--blue); }
.wdl-val { width: 46px; flex-shrink: 0; font-variant-numeric: tabular-nums; font-weight: 600; }
.pick-line { margin-top: 10px; font-size: 13px; color: var(--text-secondary); }
.pick-line b { color: var(--text-primary); }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
th { color: var(--text-muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }
td.num, th.num { text-align: right; }
.bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }
.bar-name { width: 110px; flex-shrink: 0; font-size: 13px; text-align: right; }
.bar-track { flex: 1; height: 18px; background: var(--surface-1); border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--orange); border-radius: 4px; }
.bar-val { width: 52px; flex-shrink: 0; font-size: 12.5px; font-variant-numeric: tabular-nums; color: var(--text-secondary); }
.heat-wrap { overflow-x: auto; }
.heat-grid { display: grid; gap: 1px; background: var(--border); border-radius: 6px; overflow: hidden; width: max-content; }
.heat-cell {
  width: 26px; height: 22px; display: flex; align-items: center; justify-content: center;
  font-size: 9.5px; color: var(--text-primary); position: relative; cursor: default;
}
.heat-cell.axis { background: var(--surface-2); color: var(--text-muted); font-weight: 600; }
.heat-cell[data-tip]:hover::after {
  content: attr(data-tip); position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%);
  background: var(--text-primary); color: var(--surface-2); padding: 3px 7px; border-radius: 4px;
  font-size: 11px; white-space: nowrap; z-index: 5; margin-bottom: 4px;
}
.road-team { margin-bottom: 14px; }
.road-name { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
.road-stages { display: flex; gap: 3px; }
.road-stage { flex: 1; text-align: center; }
.road-stage-track { height: 22px; background: var(--surface-1); border-radius: 3px; position: relative; overflow: hidden; }
.road-stage-fill { position: absolute; inset: 0; background: var(--orange); border-radius: 3px; }
.road-stage-label { font-size: 10px; color: var(--text-muted); margin-top: 2px; }
.edge-bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; font-size: 13px; }
.edge-team { width: 110px; flex-shrink: 0; text-align: right; }
.edge-track { flex: 1; height: 18px; position: relative; background: var(--surface-1); border-radius: 4px; }
.edge-zero { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--border); }
.edge-fill { position: absolute; top: 2px; bottom: 2px; border-radius: 3px; }
.edge-fill.pos { background: var(--orange); left: 50%; }
.edge-fill.neg { background: var(--blue); right: 50%; }
.edge-val { width: 60px; flex-shrink: 0; font-variant-numeric: tabular-nums; font-size: 12px; }
.legend { display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary); margin-bottom: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.status-final { color: var(--text-muted); }
.status-scheduled { color: var(--orange); font-weight: 600; }
.status-pending { color: var(--text-muted); }
details.tableview { margin-top: 10px; }
details.tableview summary { cursor: pointer; font-size: 12px; color: var(--text-muted); }
.note { font-size: 12.5px; color: var(--text-muted); }
</style>
</head>
<body>
<h1>World Cup 2026 -- Prediction Dashboard</h1>
<p class="subtitle">Generated <span id="gen-date"></span> &middot; XGBoost + Dixon-Coles ensemble, Monte Carlo bracket simulation, live Polymarket odds</p>

<div class="card">
  <h2>Today's matches</h2>
  <div class="grid" id="today-grid"></div>
</div>

<div class="card">
  <h2>Championship odds</h2>
  <div id="champ-chart"></div>
  <details class="tableview"><summary>View as table</summary><div id="champ-table"></div></details>
</div>

<div class="card">
  <h2>Road to the final</h2>
  <p class="note">Top 8 teams by championship odds &middot; survival probability by round</p>
  <div id="road-view"></div>
</div>

<div class="card">
  <h2>Model vs. Polymarket -- tournament winner</h2>
  <p class="note" id="edge-note"></p>
  <div id="edge-chart"></div>
  <details class="tableview"><summary>View as table</summary><div id="edge-table"></div></details>
</div>

<div class="card">
  <h2>Knockout bracket</h2>
  <div id="bracket-table"></div>
</div>

<script>
const DATA = __DATA_JSON__;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function pct(x) { return x === null || x === undefined ? "--" : (x * 100).toFixed(1) + "%"; }

document.getElementById("gen-date").textContent = DATA.generated_at;

// ── today's matches ──────────────────────────────────────────────────────
const grid = document.getElementById("today-grid");
if (DATA.today_matches.length === 0) {
  grid.appendChild(el("p", "note", "No resolvable fixtures today."));
}
DATA.today_matches.forEach(m => {
  const card = el("div", "card match-card");
  card.appendChild(el("div", "match-title", `${m.home} vs ${m.away}`));
  card.appendChild(el("div", "match-meta", `${m.date} · ${m.group} · ${m.stadium}`));

  [["home", m.home, m.p_home], ["draw", "Draw", m.p_draw], ["away", m.away, m.p_away]].forEach(([cls, label, p]) => {
    const row = el("div", "wdl-row");
    row.appendChild(el("div", "wdl-label", label));
    const track = el("div", "wdl-track");
    const fill = el("div", `wdl-fill ${cls}`);
    fill.style.width = (p * 100) + "%";
    fill.title = `${label}: ${pct(p)}`;
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("div", "wdl-val", pct(p)));
    card.appendChild(row);
  });

  const pickLine = el("div", "pick-line");
  const b = el("b", null, `PICK: ${m.pick} (${pct(m.pick_prob)})`);
  pickLine.appendChild(document.createTextNode(""));
  pickLine.appendChild(b);
  pickLine.appendChild(document.createTextNode(`  [${m.tag}]`));
  card.appendChild(pickLine);

  // scoreline heatmap (most likely ~6x6 corner of the matrix, real football scores)
  const heatWrap = el("div", "heat-wrap");
  const dim = 6;
  const gridEl = el("div", "heat-grid");
  gridEl.style.gridTemplateColumns = `repeat(${dim + 1}, 26px)`;
  gridEl.appendChild(el("div", "heat-cell axis", ""));
  for (let a = 0; a < dim; a++) gridEl.appendChild(el("div", "heat-cell axis", String(a)));
  let maxP = 0;
  for (let h = 0; h < dim; h++) for (let a = 0; a < dim; a++) maxP = Math.max(maxP, m.matrix[h][a]);
  for (let h = 0; h < dim; h++) {
    gridEl.appendChild(el("div", "heat-cell axis", String(h)));
    for (let a = 0; a < dim; a++) {
      const p = m.matrix[h][a];
      const t = maxP > 0 ? p / maxP : 0;
      const cell = el("div", "heat-cell", (p * 100).toFixed(1));
      cell.style.background = `rgba(255,107,24,${(0.06 + 0.85 * t).toFixed(3)})`;
      cell.dataset.tip = `${m.home} ${h} - ${a} ${m.away}: ${pct(p)}`;
      gridEl.appendChild(cell);
    }
  }
  heatWrap.appendChild(gridEl);
  card.appendChild(el("div", "match-meta", "Most likely scorelines (Dixon-Coles)"));
  card.appendChild(heatWrap);

  grid.appendChild(card);
});

// ── championship odds bar chart ──────────────────────────────────────────
const champChart = document.getElementById("champ-chart");
const champMax = DATA.championship.length ? DATA.championship[0].p_champion : 1;
DATA.championship.slice(0, 15).forEach(r => {
  const row = el("div", "bar-row");
  row.appendChild(el("div", "bar-name", r.team));
  const track = el("div", "bar-track");
  const fill = el("div", "bar-fill");
  fill.style.width = (champMax > 0 ? (r.p_champion / champMax) * 100 : 0) + "%";
  fill.title = `${r.team}: ${pct(r.p_champion)}`;
  track.appendChild(fill);
  row.appendChild(track);
  row.appendChild(el("div", "bar-val", pct(r.p_champion)));
  champChart.appendChild(row);
});
const champTable = document.getElementById("champ-table");
{
  const t = el("table");
  const thead = el("tr");
  ["Team", "R16", "QF", "SF", "Final", "Champion"].forEach(h => thead.appendChild(el("th", h === "Team" ? "" : "num", h)));
  t.appendChild(thead);
  DATA.championship.forEach(r => {
    const tr = el("tr");
    tr.appendChild(el("td", "", r.team));
    [r.p_r16, r.p_qf, r.p_sf, r.p_final, r.p_champion].forEach(v => tr.appendChild(el("td", "num", pct(v))));
    t.appendChild(tr);
  });
  champTable.appendChild(t);
}

// ── road to the final (small multiples) ──────────────────────────────────
const roadView = document.getElementById("road-view");
const stages = [["p_r16", "R16"], ["p_qf", "QF"], ["p_sf", "SF"], ["p_final", "Final"], ["p_champion", "Champion"]];
DATA.road_to_final.forEach(r => {
  const wrap = el("div", "road-team");
  wrap.appendChild(el("div", "road-name", r.team));
  const stagesEl = el("div", "road-stages");
  stages.forEach(([key, label]) => {
    const p = r[key];
    const stage = el("div", "road-stage");
    const track = el("div", "road-stage-track");
    const fill = el("div", "road-stage-fill");
    fill.style.width = Math.max(2, p * 100) + "%";
    fill.title = `${r.team} -- ${label}: ${pct(p)}`;
    track.appendChild(fill);
    stage.appendChild(track);
    stage.appendChild(el("div", "road-stage-label", `${label} ${pct(p)}`));
    stagesEl.appendChild(stage);
  });
  wrap.appendChild(stagesEl);
  roadView.appendChild(wrap);
});

// ── edge: model vs Polymarket (diverging) ────────────────────────────────
const edgeNote = document.getElementById("edge-note");
if (DATA.market_error) {
  edgeNote.textContent = `Polymarket odds unavailable this run (${DATA.market_error}).`;
} else if (DATA.edge.length === 0) {
  edgeNote.textContent = "No teams currently priced by both the model and Polymarket.";
} else {
  edgeNote.textContent = `${DATA.edge.length} teams priced by both. Orange = model favors more than the market; blue = market favors more than the model.`;
  const legend = el("div", "legend");
  legend.innerHTML = "";
  const l1 = el("span"); l1.appendChild(el("span", "swatch")); l1.lastChild.style.background = "var(--orange)"; l1.appendChild(document.createTextNode("Model > Market"));
  const l2 = el("span"); l2.appendChild(el("span", "swatch")); l2.lastChild.style.background = "var(--blue)"; l2.appendChild(document.createTextNode("Market > Model"));
  legend.appendChild(l1); legend.appendChild(l2);
  document.getElementById("edge-chart").appendChild(legend);
}
const edgeMax = Math.max(0.01, ...DATA.edge.map(r => Math.abs(r.edge)));
const edgeChart = document.getElementById("edge-chart");
DATA.edge.slice(0, 15).forEach(r => {
  const row = el("div", "edge-bar-row");
  row.appendChild(el("div", "edge-team", r.team));
  const track = el("div", "edge-track");
  track.appendChild(el("div", "edge-zero"));
  const fill = el("div", `edge-fill ${r.edge >= 0 ? "pos" : "neg"}`);
  const w = (Math.abs(r.edge) / edgeMax) * 50;
  fill.style.width = w + "%";
  fill.title = `${r.team}: model ${pct(r.model)} vs market ${pct(r.market)} (edge ${r.edge >= 0 ? "+" : ""}${(r.edge * 100).toFixed(1)}%)`;
  track.appendChild(fill);
  row.appendChild(track);
  row.appendChild(el("div", "edge-val", `${r.edge >= 0 ? "+" : ""}${(r.edge * 100).toFixed(1)}%`));
  edgeChart.appendChild(row);
});
const edgeTable = document.getElementById("edge-table");
{
  const t = el("table");
  const thead = el("tr");
  ["Team", "Model", "Market", "Edge"].forEach(h => thead.appendChild(el("th", h === "Team" ? "" : "num", h)));
  t.appendChild(thead);
  DATA.edge.forEach(r => {
    const tr = el("tr");
    tr.appendChild(el("td", "", r.team));
    tr.appendChild(el("td", "num", pct(r.model)));
    tr.appendChild(el("td", "num", pct(r.market)));
    tr.appendChild(el("td", "num", `${r.edge >= 0 ? "+" : ""}${(r.edge * 100).toFixed(1)}%`));
    t.appendChild(tr);
  });
  edgeTable.appendChild(t);
}

// ── bracket table ─────────────────────────────────────────────────────────
{
  const t = el("table");
  const thead = el("tr");
  ["Match", "Round", "Date", "Home", "Away", "Status"].forEach(h => thead.appendChild(el("th", "", h)));
  t.appendChild(thead);
  DATA.bracket.forEach(s => {
    const tr = el("tr");
    tr.appendChild(el("td", "", "#" + s.match_number));
    tr.appendChild(el("td", "", s.round));
    tr.appendChild(el("td", "", s.date));
    tr.appendChild(el("td", "", s.home));
    tr.appendChild(el("td", "", s.away));
    let statusText = s.status === "final" ? `Final -- ${s.winner} won` : s.status === "scheduled" ? "Scheduled" : "Pending";
    tr.appendChild(el("td", `status-${s.status}`, statusText));
    t.appendChild(tr);
  });
  document.getElementById("bracket-table").appendChild(t);
}
</script>
</body>
</html>
"""


def render(data, out_path):
    html = _TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path


if __name__ == "__main__":
    data = build_data()
    date_str = data["generated_at"]
    out_path = os.path.join("predictions", date_str, "dashboard.html")
    path = render(data, out_path)
    print(f"Dashboard written -> {path}")
    print(f"  {len(data['today_matches'])} today's matches, "
          f"{len(data['championship'])} teams in championship odds, "
          f"{len(data['edge'])} teams priced by both model and market")
