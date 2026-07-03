"""wcpred.viz — branded charts. Reuses the palette across all viz functions."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette (matches the existing reel charts)
INK, MUTE, GRID = "#1a1a2e", "#8a8a9e", "#e8e8ee"
ORANGE, BLUE, GRAY = "#ff6b18", "#1f6feb", "#9aa0a6"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


# ── chart ───────────────────────────────────────────────────────────────────────
def make_chart(m, p_home, p_draw, p_away, slate_date, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [f"{m['home_disp']}\nwin", "Draw", f"{m['away_disp']}\nwin"]
    vals = [p_home, p_draw, p_away]
    colors = [ORANGE, GRAY, BLUE]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.2, f"{v*100:.1f}%",
                ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(vals) * 100 + 12)
    ax.set_ylabel("Win probability (%)")
    sub = f"{slate_date}  ·  {m['group']}  ·  {m['stadium']}"
    ax.set_title(f"{m['home_disp']} vs {m['away_disp']}\n{sub}", fontsize=13)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    safe = f"{m['home_disp']}_vs_{m['away_disp']}".replace(" ", "_").replace("/", "-")
    path = os.path.join(out_dir, f"viz_{safe}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def tag_match(top_prob, p_home, p_away, home_elo, away_elo):
    favorite_is_home = p_home >= p_away
    fav_elo_is_home = home_elo >= away_elo
    upset = (favorite_is_home != fav_elo_is_home)
    if top_prob >= 0.60:
        strength = "LOCK"
    elif top_prob >= 0.45:
        strength = "LEAN"
    else:
        strength = "TOSS-UP"
    return strength + ("  ⚠️ UPSET PICK" if upset else "")


# ── new shareable charts (Wave 1 / Agent B3) ─────────────────────────────────────
# All three reuse the palette/rcParams defined above. Append-only: nothing above
# this line is modified.

def scoreline_heatmap(home_name, away_name, score_matrix, match_meta, out_dir):
    """Heatmap of P(home scores i, away scores j) for i,j in 0..10.

    score_matrix: 11x11 array-like, M[i,j] = P(home=i, away=j)  (wcpred.model_goals.score_matrix shape).
    match_meta: dict with home_disp/away_disp/group/stadium, same convention as make_chart's `m`.
    Returns the saved PNG path.
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    M = np.asarray(score_matrix, dtype=float)
    n_home, n_away = M.shape

    # sequential ramp built from the brand's ORANGE (white -> ORANGE), not a stock rainbow map
    cmap = LinearSegmentedColormap.from_list("brand_sequential", ["#ffffff", ORANGE])

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(M, origin="lower", cmap=cmap, aspect="auto",
                   extent=(-0.5, n_away - 0.5, -0.5, n_home - 0.5), zorder=2)

    i_max, j_max = np.unravel_index(np.argmax(M), M.shape)
    p_max = M[i_max, j_max]
    ax.add_patch(plt.Rectangle((j_max - 0.5, i_max - 0.5), 1, 1, fill=False,
                                edgecolor=INK, linewidth=2.2, zorder=4))
    text_color = "white" if p_max >= 0.55 * M.max() else INK
    ax.text(j_max, i_max, f"{i_max}-{j_max}\n{p_max*100:.1f}%", ha="center", va="center",
            fontsize=11, fontweight="bold", color=text_color, zorder=5)

    ax.set_xticks(range(n_away))
    ax.set_yticks(range(n_home))
    ax.set_xlabel(f"{match_meta['away_disp']} goals")
    ax.set_ylabel(f"{match_meta['home_disp']} goals")
    sub = f"{match_meta['group']}  ·  {match_meta['stadium']}"
    ax.set_title(f"{match_meta['home_disp']} vs {match_meta['away_disp']}\n{sub}", fontsize=13)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Scoreline probability")
    cbar.outline.set_visible(False)

    fig.tight_layout()
    safe = f"{home_name}_vs_{away_name}".replace(" ", "_").replace("/", "-")
    path = os.path.join(out_dir, f"viz_heatmap_{safe}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def championship_bar(team_probs, out_dir, top_n=16):
    """Horizontal bar chart of top_n teams by P(champion), descending.

    team_probs: dict-like (or pandas Series) of team -> probability.
    Returns the saved PNG path.
    """
    items = team_probs.to_dict() if hasattr(team_probs, "to_dict") else dict(team_probs)
    ranked = sorted(items.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    plot_order = list(reversed(ranked))  # barh draws bottom-up; reverse so #1 lands on top
    teams = [t for t, _ in plot_order]
    vals = [v for _, v in plot_order]

    fig_h = max(4.0, 0.42 * len(teams) + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    bars = ax.barh(teams, [v * 100 for v in vals], color=ORANGE, height=0.62, zorder=3)
    top_val = max(vals) if vals else 0.0
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + top_val * 100 * 0.015, b.get_y() + b.get_height() / 2,
                f"{v*100:.1f}%", ha="left", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, top_val * 100 * 1.18 if top_val > 0 else 1)
    ax.set_xlabel("Championship probability (%)")
    ax.set_title(f"World Cup 2026 — Championship Odds\nTop {len(teams)} teams", fontsize=13)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(out_dir, "viz_championship_odds.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _bracket_rows(bracket_probs):
    """Normalize a DataFrame / dict-of-dicts / dict-of-columns into a list of row dicts."""
    if hasattr(bracket_probs, "to_dict"):
        try:
            return bracket_probs.to_dict("records")
        except TypeError:
            pass
    if isinstance(bracket_probs, dict):
        first_val = next(iter(bracket_probs.values()), None)
        if isinstance(first_val, dict):
            rows = []
            for team, stats in bracket_probs.items():
                row = {"team": team}
                row.update(stats)
                rows.append(row)
            return rows
        keys = list(bracket_probs.keys())
        n = len(bracket_probs[keys[0]]) if keys else 0
        return [{k: bracket_probs[k][i] for k in keys} for i in range(n)]
    return list(bracket_probs)


def road_to_final(bracket_probs, out_dir):
    """Slopegraph of each team's survival probability declining round by round.

    bracket_probs: DataFrame/dict shaped like wcpred.simulate.run()'s output —
    team, p_r16, p_qf, p_sf, p_final, p_champion.
    Returns the saved PNG path.
    """
    rows = _bracket_rows(bracket_probs)
    stage_cols = ["p_r16", "p_qf", "p_sf", "p_final", "p_champion"]
    stage_labels = ["R16", "QF", "SF", "Final", "Champion"]
    rows = sorted(rows, key=lambda r: r.get("p_champion", 0.0), reverse=True)
    top = rows[:8]

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    x = list(range(len(stage_cols)))
    highlight_colors = [ORANGE, BLUE]
    for idx, row in enumerate(top):
        y = [row.get(c, 0.0) * 100 for c in stage_cols]
        if idx < len(highlight_colors):
            color, lw, z, alpha, weight = highlight_colors[idx], 2.6, 4, 1.0, "bold"
        else:
            color, lw, z, alpha, weight = GRAY, 1.6, 3, 0.75, "normal"
        ax.plot(x, y, color=color, linewidth=lw, marker="o", markersize=5, zorder=z, alpha=alpha)
        label_color = color if idx < len(highlight_colors) else MUTE
        ax.text(x[-1] + 0.08, y[-1], row.get("team", "?"), color=label_color,
                fontsize=10, fontweight=weight, va="center", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_xlim(-0.3, len(stage_cols) - 0.3 + 1.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Survival probability (%)")
    ax.set_title("Road to the Final — survival probability by round", fontsize=13)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(out_dir, "viz_road_to_final.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
