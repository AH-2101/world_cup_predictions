// wcpred interactive front end -- vanilla JS, no build step.
// Forked from wcpred/dashboard.py's inline renderers (same look/behavior),
// adapted to fetch('/api/...') instead of reading a baked-in JSON blob.

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function pct(x) { return x === null || x === undefined ? "--" : (x * 100).toFixed(1) + "%"; }
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

async function getJSON(url, opts) {
  const resp = await fetch(url, opts);
  const body = await resp.json();
  if (!resp.ok) throw new Error(body.error || `${url} -> HTTP ${resp.status}`);
  return body;
}

// ── match card (predict-a-match result + today's slate) ────────────────────
function renderMatchCard(m) {
  const card = el("div", "card match-card");
  card.appendChild(el("div", "match-title", `${m.home} vs ${m.away}`));
  card.appendChild(el("div", "match-meta", `${m.date} · ${m.group} · ${m.stadium || ""}`));

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
  pickLine.appendChild(b);
  pickLine.appendChild(document.createTextNode(`  [${m.tag}]`));
  card.appendChild(pickLine);

  if (m.market) {
    const mkt = el("div", "market-line",
      `Polymarket: ${m.home} ${pct(m.market.p_home)} · Draw ${pct(m.market.p_draw)} · ${m.away} ${pct(m.market.p_away)}`);
    card.appendChild(mkt);
  }

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
  return card;
}

// ── predict a match ─────────────────────────────────────────────────────────
async function loadTeams() {
  const teams = await getJSON("/api/teams");
  const homeSel = document.getElementById("pick-home");
  const awaySel = document.getElementById("pick-away");
  teams.forEach(t => {
    homeSel.appendChild(el("option", null, t)).value = t;
    awaySel.appendChild(el("option", null, t)).value = t;
  });
}

async function runPredict() {
  const home = document.getElementById("pick-home").value;
  const away = document.getElementById("pick-away").value;
  const out = document.getElementById("predict-result");
  clear(out);
  if (!home || !away || home === away) {
    out.appendChild(el("div", "error-msg", "Pick two different teams."));
    return;
  }
  const btn = document.getElementById("predict-btn");
  btn.disabled = true;
  out.appendChild(el("div", "spinner", "Predicting ..."));
  try {
    const m = await getJSON(`/api/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`);
    clear(out);
    out.appendChild(renderMatchCard(m));
    loadReportCard(); // this predict just got logged to the ledger
  } catch (e) {
    clear(out);
    out.appendChild(el("div", "error-msg", e.message));
  } finally {
    btn.disabled = false;
  }
}

// ── today's slate ────────────────────────────────────────────────────────────
async function loadToday() {
  const grid = document.getElementById("today-grid");
  clear(grid);
  const matches = await getJSON("/api/today");
  if (matches.length === 0) {
    grid.appendChild(el("p", "note", "No resolvable fixtures today."));
    return;
  }
  matches.forEach(m => grid.appendChild(renderMatchCard(m)));
}

// ── championship odds + road to final ────────────────────────────────────────
async function loadSim() {
  const championship = await getJSON("/api/sim");

  const champChart = document.getElementById("champ-chart");
  clear(champChart);
  const champMax = championship.length ? championship[0].p_champion : 1;
  championship.slice(0, 15).forEach(r => {
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
  clear(champTable);
  {
    const t = el("table");
    const thead = el("tr");
    ["Team", "R16", "QF", "SF", "Final", "Champion"].forEach(h => thead.appendChild(el("th", h === "Team" ? "" : "num", h)));
    t.appendChild(thead);
    championship.forEach(r => {
      const tr = el("tr");
      tr.appendChild(el("td", "", r.team));
      [r.p_r16, r.p_qf, r.p_sf, r.p_final, r.p_champion].forEach(v => tr.appendChild(el("td", "num", pct(v))));
      t.appendChild(tr);
    });
    champTable.appendChild(t);
  }

  const roadView = document.getElementById("road-view");
  clear(roadView);
  const stages = [["p_r16", "R16"], ["p_qf", "QF"], ["p_sf", "SF"], ["p_final", "Final"], ["p_champion", "Champion"]];
  championship.slice(0, 8).forEach(r => {
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
}

// ── edge: model vs Polymarket ────────────────────────────────────────────────
async function loadEdge() {
  const { edge, error } = await getJSON("/api/edge");
  const edgeNote = document.getElementById("edge-note");
  const edgeChart = document.getElementById("edge-chart");
  const edgeTable = document.getElementById("edge-table");
  clear(edgeChart);
  clear(edgeTable);

  if (error) {
    edgeNote.textContent = `Polymarket odds unavailable this run (${error}).`;
    return;
  }
  if (edge.length === 0) {
    edgeNote.textContent = "No teams currently priced by both the model and Polymarket.";
    return;
  }
  edgeNote.textContent = `${edge.length} teams priced by both. Orange = model favors more than the market; blue = market favors more than the model.`;

  const legend = el("div", "legend");
  const l1 = el("span"); l1.appendChild(el("span", "swatch")); l1.lastChild.style.background = "var(--orange)"; l1.appendChild(document.createTextNode("Model > Market"));
  const l2 = el("span"); l2.appendChild(el("span", "swatch")); l2.lastChild.style.background = "var(--blue)"; l2.appendChild(document.createTextNode("Market > Model"));
  legend.appendChild(l1); legend.appendChild(l2);
  edgeChart.appendChild(legend);

  const edgeMax = Math.max(0.01, ...edge.map(r => Math.abs(r.edge)));
  edge.slice(0, 15).forEach(r => {
    const row = el("div", "edge-bar-row");
    row.appendChild(el("div", "edge-team", r.team));
    const track = el("div", "edge-track");
    track.appendChild(el("div", "edge-zero"));
    const fill = el("div", `edge-fill ${r.edge >= 0 ? "pos" : "neg"}`);
    fill.style.width = (Math.abs(r.edge) / edgeMax) * 50 + "%";
    fill.title = `${r.team}: model ${pct(r.model)} vs market ${pct(r.market)} (edge ${r.edge >= 0 ? "+" : ""}${(r.edge * 100).toFixed(1)}%)`;
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("div", "edge-val", `${r.edge >= 0 ? "+" : ""}${(r.edge * 100).toFixed(1)}%`));
    edgeChart.appendChild(row);
  });

  const t = el("table");
  const thead = el("tr");
  ["Team", "Model", "Market", "Edge"].forEach(h => thead.appendChild(el("th", h === "Team" ? "" : "num", h)));
  t.appendChild(thead);
  edge.forEach(r => {
    const tr = el("tr");
    tr.appendChild(el("td", "", r.team));
    tr.appendChild(el("td", "num", pct(r.model)));
    tr.appendChild(el("td", "num", pct(r.market)));
    tr.appendChild(el("td", "num", `${r.edge >= 0 ? "+" : ""}${(r.edge * 100).toFixed(1)}%`));
    t.appendChild(tr);
  });
  edgeTable.appendChild(t);
}

// ── bracket ──────────────────────────────────────────────────────────────────
async function loadBracket() {
  const bracket = await getJSON("/api/bracket");
  const container = document.getElementById("bracket-table");
  clear(container);
  const t = el("table");
  const thead = el("tr");
  ["Match", "Round", "Date", "Home", "Away", "Status"].forEach(h => thead.appendChild(el("th", "", h)));
  t.appendChild(thead);
  bracket.forEach(s => {
    const tr = el("tr");
    tr.appendChild(el("td", "", "#" + s.match_number));
    tr.appendChild(el("td", "", s.round));
    tr.appendChild(el("td", "", s.date));
    tr.appendChild(el("td", "", s.home));
    tr.appendChild(el("td", "", s.away));
    const statusText = s.status === "final" ? `Final -- ${s.winner} won` : s.status === "scheduled" ? "Scheduled" : "Pending";
    tr.appendChild(el("td", `status-${s.status}`, statusText));
    t.appendChild(tr);
  });
  container.appendChild(t);
}

// ── model report card ────────────────────────────────────────────────────────
function tile(label, value, sub, cls) {
  const t = el("div", "tile");
  t.appendChild(el("div", "tile-label", label));
  t.appendChild(el("div", `tile-value ${cls || ""}`, value));
  if (sub) t.appendChild(el("div", "tile-sub", sub));
  return t;
}

function renderTrend(perMatch) {
  const container = document.getElementById("report-trend");
  clear(container);
  if (perMatch.length < 2) return;
  const w = 640, h = 120, pad = 8;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", h);
  const n = perMatch.length;
  const pts = perMatch.map((r, i) => {
    const x = pad + (i / (n - 1)) * (w - 2 * pad);
    const y = h - pad - r.cum_accuracy * (h - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.setAttribute("points", pts);
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "var(--orange)");
  line.setAttribute("stroke-width", "2");
  svg.appendChild(line);
  container.appendChild(svg);
  container.appendChild(el("div", "note", "Cumulative accuracy over scored predictions, in match order"));
}

async function loadReportCard() {
  const card = await getJSON("/api/report-card");
  const tiles = document.getElementById("report-tiles");
  clear(tiles);

  tiles.appendChild(tile("Logged", card.n_logged));
  tiles.appendChild(tile("Scored (honest)", card.n_scored));
  if (card.n_scored) {
    tiles.appendChild(tile("Accuracy", pct(card.accuracy)));
    const llCls = card.log_loss < card.baseline_log_loss ? "win" : "lose";
    tiles.appendChild(tile("Log-loss", card.log_loss.toFixed(3), `vs ${card.baseline_log_loss.toFixed(3)} baseline`, llCls));
    const brCls = card.brier < card.baseline_brier ? "win" : "lose";
    tiles.appendChild(tile("Brier", card.brier.toFixed(3), `vs ${card.baseline_brier.toFixed(3)} baseline`, brCls));
    if (card.market_n) {
      const mCls = card.log_loss < card.market_log_loss ? "win" : "lose";
      tiles.appendChild(tile("vs Polymarket", card.log_loss.toFixed(3), `market ${card.market_log_loss.toFixed(3)} (n=${card.market_n})`, mCls));
    }
  }
  tiles.appendChild(tile("Blend alpha", card.alpha.toFixed(3), `unweighted ${card.alpha_base.toFixed(3)}`));
  tiles.appendChild(tile("Results as of", card.results_max_date));

  renderTrend(card.per_match || []);

  const table = document.getElementById("report-table");
  clear(table);
  if (card.per_match && card.per_match.length) {
    const t = el("table");
    const thead = el("tr");
    ["Date", "Home", "Away", "Pick", "P(pick)", "Correct", "Log-loss"].forEach(h => thead.appendChild(el("th", "", h)));
    t.appendChild(thead);
    card.per_match.slice().reverse().forEach(r => {
      const tr = el("tr");
      tr.appendChild(el("td", "", r.date));
      tr.appendChild(el("td", "", r.home));
      tr.appendChild(el("td", "", r.away));
      tr.appendChild(el("td", "", r.pick));
      tr.appendChild(el("td", "num", pct(r.pick_prob)));
      tr.appendChild(el("td", r.correct ? "status-scheduled" : "error-msg", r.correct ? "Y" : "n"));
      tr.appendChild(el("td", "num", r.log_loss.toFixed(3)));
      t.appendChild(tr);
    });
    table.appendChild(t);
  } else {
    table.appendChild(el("p", "note", "No honestly-scored predictions yet."));
  }
}

async function runRefresh() {
  const btn = document.getElementById("refresh-btn");
  const status = document.getElementById("refresh-status");
  btn.disabled = true;
  status.textContent = "Refreshing + rescoring (can take ~20s) ...";
  try {
    const r = await getJSON("/api/refresh", { method: "POST" });
    status.textContent = `Done. Results as of ${r.results_max_date}. ` +
      `Alpha ${r.alpha_before.toFixed(3)} -> ${r.alpha_after.toFixed(3)}. ` +
      `Scored ${r.n_scored_before} -> ${r.n_scored_after}.`;
    await Promise.all([loadToday(), loadSim(), loadEdge(), loadBracket(), loadReportCard()]);
  } catch (e) {
    status.textContent = `Refresh failed: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

// ── boot ─────────────────────────────────────────────────────────────────────
document.getElementById("predict-btn").addEventListener("click", runPredict);
document.getElementById("refresh-btn").addEventListener("click", runRefresh);
loadTeams();
loadToday();
loadReportCard();
loadSim();
loadEdge();
loadBracket();
