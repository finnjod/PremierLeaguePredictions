# Premier League Predictor

Predicts Premier League standings using historical team statistics, Random Forest, Gradient Boosting, Ridge regression, and backtest-derived feature weights.

## Setup
Requires Python 3.12.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

## Run
Keep the historical season CSVs in the locations expected by the source code.

    python epl_predictorv4.py

## Outputs
Projected standings, backtest results, feature importance reports, and charts.

## Data source
https://www.football-data.co.uk/englandm.php
