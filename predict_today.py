"""
predict_today.py — World Cup 2026 daily match predictor (back-compat shim)
===========================================================================

The implementation now lives in the `wcpred` package (see wcpred/cli.py).
This file is kept as a thin drop-in entry point so existing usage keeps working:

    python3 predict_today.py "Saudi Arabia" "Uruguay"
    python3 predict_today.py Spain "Cabo Verde"
    python3 predict_today.py            # then type the two teams when prompted

Team order doesn't matter. The match date, group and stadium are looked up
automatically from data_cache/fixtures.csv, so you only type the two teams.

It prints the win / draw / win probabilities, the pick, and a tag
(LOCK / LEAN / TOSS-UP, plus UPSET PICK), and saves one branded reel chart to:
    predictions/<date>/viz_<Home>_vs_<Away>.png
"""

from wcpred.cli import main

if __name__ == "__main__":
    main()
