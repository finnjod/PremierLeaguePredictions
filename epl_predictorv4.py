"""
EPL Season Predictor v5
=======================

Backtests every added Premier League season from 2020-21 through 2025-26,
derives hyperweighted feature importance from those holdouts, and predicts
the 2026-27 table.

DATA SOURCE: https://www.football-data.co.uk/englandm.php
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE, ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# E0(6).csv is the 2019-20 context season. It is needed so 2020-21 has a
# prior Premier League season to be predicted from.
SEASON_FILES = {
    "2019-20": os.path.join(BASE, "E0(6).csv"),
    "2020-21": os.path.join(BASE, "E0 (5).csv"),
    "2021-22": os.path.join(BASE, "E0 (4).csv"),
    "2022-23": os.path.join(BASE, "E0 (3).csv"),
    "2023-24": os.path.join(BASE, "E0 (2).csv"),
    "2024-25": os.path.join(BASE, "E0 (1).csv"),
    "2025-26": os.path.join(BASE, "E0.csv"),
}

BACKTEST_TARGET_SEASONS = [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

PREDICTION_INPUT_SEASON = "2025-26"
PREDICTION_SEASON = "2026-27"

# Current fixture reports list Coventry, Ipswich, and Hull as the promoted
# clubs for 2026-27. If your source differs, edit this list.
PROMOTED_TEAMS_2627 = ["Coventry", "Ipswich", "Hull"]

# Amplifies reliable feature importance. 1.0 = plain normalized weights.
HYPERWEIGHT_POWER = 1.75
PERMUTATION_REPEATS = 12
RANDOM_SEED = 42

OUT_DIR = BASE

CORE_COLS = [
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]

FEATURE_COLS = [
    "points",
    "goal_diff",
    "goals_for",
    "goals_against",
    "ppg",
    "wins",
    "draws",
    "losses",
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
    "shot_conversion",
    "sot_pct",
    "shots_pg",
    "sot_pg",
    "yellows",
    "reds",
    "fouls",
    "corners",
    "home_ppg",
    "away_ppg",
    "home_away_delta",
]

BG, PANEL = "#0d1117", "#161b22"
GOLD, GREEN, BLUE, RED, WHITE, GREY = (
    "#FFD700",
    "#2ea043",
    "#58a6ff",
    "#f85149",
    "#e6edf3",
    "#8b949e",
)


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
def season_start_year(label):
    return int(label.split("-")[0])


def sorted_seasons(labels):
    return sorted(labels, key=season_start_year)


def load_season(path, label):
    available = pd.read_csv(path, nrows=0).columns.tolist()
    use = [col for col in CORE_COLS if col in available]
    df = pd.read_csv(path, usecols=use)

    required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    df = df.dropna(subset=[col for col in required if col in df.columns]).copy()
    df = df[df["FTR"].isin(["H", "D", "A"])].copy()

    for col in CORE_COLS[5:]:
        if col not in df.columns:
            df[col] = 0

    numeric_cols = ["FTHG", "FTAG"] + CORE_COLS[5:]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["season"] = label
    return df


def load_all(season_files):
    frames = []
    for label in sorted_seasons(season_files):
        path = season_files[label]
        if os.path.exists(path):
            frame = load_season(path, label)
            frames.append(frame)
            print(f"  loaded {label:7s}  matches={len(frame):3d}  file={os.path.basename(path)}")
        else:
            print(f"  missing {label:7s}  file={path}")

    if not frames:
        raise FileNotFoundError("No season files found.")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# FEATURES
# ---------------------------------------------------------------------------
def team_season_features(matches):
    rows = []

    for season in sorted_seasons(matches["season"].unique()):
        season_df = matches[matches["season"] == season]
        teams = sorted(set(season_df["HomeTeam"]) | set(season_df["AwayTeam"]))

        for team in teams:
            home = season_df[season_df["HomeTeam"] == team]
            away = season_df[season_df["AwayTeam"] == team]

            home_pts = (home["FTR"] == "H").sum() * 3 + (home["FTR"] == "D").sum()
            away_pts = (away["FTR"] == "A").sum() * 3 + (away["FTR"] == "D").sum()
            total_pts = home_pts + away_pts

            gf = home["FTHG"].sum() + away["FTAG"].sum()
            ga = home["FTAG"].sum() + away["FTHG"].sum()

            wins = (home["FTR"] == "H").sum() + (away["FTR"] == "A").sum()
            draws = (home["FTR"] == "D").sum() + (away["FTR"] == "D").sum()
            losses = (home["FTR"] == "A").sum() + (away["FTR"] == "H").sum()
            games = wins + draws + losses

            sf = home["HS"].sum() + away["AS"].sum()
            sa = home["AS"].sum() + away["HS"].sum()
            sof = home["HST"].sum() + away["AST"].sum()
            soa = home["AST"].sum() + away["HST"].sum()

            home_ppg = home_pts / len(home) if len(home) else 0
            away_ppg = away_pts / len(away) if len(away) else 0

            rows.append(
                {
                    "season": season,
                    "team": team,
                    "points": total_pts,
                    "goal_diff": gf - ga,
                    "goals_for": gf,
                    "goals_against": ga,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "ppg": total_pts / games if games else 0,
                    "shots_for": sf,
                    "shots_against": sa,
                    "sot_for": sof,
                    "sot_against": soa,
                    "shot_conversion": gf / sf if sf else 0,
                    "sot_pct": sof / sf if sf else 0,
                    "shots_pg": sf / games if games else 0,
                    "sot_pg": sof / games if games else 0,
                    "yellows": home["HY"].sum() + away["AY"].sum(),
                    "reds": home["HR"].sum() + away["AR"].sum(),
                    "fouls": home["HF"].sum() + away["AF"].sum(),
                    "corners": home["HC"].sum() + away["AC"].sum(),
                    "home_ppg": home_ppg,
                    "away_ppg": away_ppg,
                    "home_away_delta": home_ppg - away_ppg,
                }
            )

    return pd.DataFrame(rows)


def rank_table(table, points_col="points", gd_col="goal_diff", gf_col="goals_for"):
    ranked = table.sort_values(
        [points_col, gd_col, gf_col, "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def current_table(feats, season):
    season_rows = feats[feats["season"] == season].copy()
    return rank_table(season_rows)


# ---------------------------------------------------------------------------
# LAG DATASET: season N -> season N + 1
# ---------------------------------------------------------------------------
def iter_lag_transitions(feats, allowed_target_seasons=None, exclude_seasons=None):
    seasons = sorted_seasons(feats["season"].unique())
    allowed = set(allowed_target_seasons) if allowed_target_seasons else None
    excluded = set(exclude_seasons) if exclude_seasons else set()

    for idx in range(1, len(seasons)):
        input_season = seasons[idx - 1]
        target_season = seasons[idx]

        if allowed is not None and target_season not in allowed:
            continue
        if input_season in excluded or target_season in excluded:
            continue

        yield input_season, target_season


def build_lag_dataset(feats, allowed_target_seasons=None, exclude_seasons=None):
    x_rows, y_rows, meta_rows = [], [], []

    for input_season, target_season in iter_lag_transitions(
        feats,
        allowed_target_seasons=allowed_target_seasons,
        exclude_seasons=exclude_seasons,
    ):
        input_rows = feats[feats["season"] == input_season].set_index("team")
        target_rows = feats[feats["season"] == target_season].set_index("team")
        common_teams = sorted(input_rows.index.intersection(target_rows.index))

        for team in common_teams:
            x_rows.append(input_rows.loc[team, FEATURE_COLS].astype(float).values)
            y_rows.append(float(target_rows.loc[team, "points"]))
            meta_rows.append(
                {
                    "team": team,
                    "input_season": input_season,
                    "target_season": target_season,
                }
            )

    X = pd.DataFrame(x_rows, columns=FEATURE_COLS)
    y = pd.Series(y_rows, name="next_points")
    meta = pd.DataFrame(meta_rows)
    groups = meta["input_season"] if len(meta) else pd.Series(dtype=object)
    return X, y, groups, meta


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
def make_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.045,
            random_state=RANDOM_SEED,
        ),
        "Ridge": Ridge(alpha=8.0),
    }


def weight_array(feature_weights):
    if feature_weights is None:
        return None
    if isinstance(feature_weights, pd.Series):
        return feature_weights.reindex(FEATURE_COLS).fillna(1.0).values.astype(float)
    return np.asarray(feature_weights, dtype=float)


def transform_features(scaler, X, feature_weights=None):
    X_clean = X[FEATURE_COLS].astype(float).fillna(0)
    Xs = scaler.transform(X_clean)
    weights = weight_array(feature_weights)
    if weights is not None:
        Xs = Xs * weights
    return Xs


def train_models(X, y, feature_weights=None):
    X_clean = X[FEATURE_COLS].astype(float).fillna(0)
    scaler = StandardScaler()
    scaler.fit(X_clean)
    Xs = transform_features(scaler, X_clean, feature_weights)

    trained = {}
    for name, model in make_models().items():
        model.fit(Xs, y)
        trained[name] = model

    return trained, scaler


def ensemble_predict(trained, scaler, X, feature_weights=None):
    Xs = transform_features(scaler, X, feature_weights)
    preds = np.array([model.predict(Xs) for model in trained.values()])
    return preds.mean(axis=0)


def safe_spearman(a, b):
    rho, _ = spearmanr(a, b)
    return 0.0 if pd.isna(rho) else float(rho)


# ---------------------------------------------------------------------------
# BACKTESTS AND IMPORTANCE
# ---------------------------------------------------------------------------
def backtest_one_season(feats, target_season):
    seasons = sorted_seasons(feats["season"].unique())
    target_idx = seasons.index(target_season)
    if target_idx == 0:
        raise ValueError(f"{target_season} has no prior season to predict from.")

    input_season = seasons[target_idx - 1]
    X_train, y_train, _, train_meta = build_lag_dataset(
        feats,
        allowed_target_seasons=BACKTEST_TARGET_SEASONS,
        exclude_seasons={target_season},
    )

    if len(X_train) < 20:
        raise ValueError(f"{target_season} has too little training data: {len(X_train)} rows")

    trained, scaler = train_models(X_train, y_train)

    input_rows = feats[feats["season"] == input_season].set_index("team")
    target_rows = feats[feats["season"] == target_season].set_index("team")
    common_teams = sorted(input_rows.index.intersection(target_rows.index))

    X_eval = input_rows.loc[common_teams, FEATURE_COLS].reset_index(drop=True)
    preds = ensemble_predict(trained, scaler, X_eval)

    actual = target_rows.loc[common_teams, ["points", "goal_diff", "goals_for"]].reset_index()
    actual = actual.rename(
        columns={
            "points": "actual_pts",
            "goal_diff": "actual_goal_diff",
            "goals_for": "actual_goals_for",
        }
    )

    result = input_rows.loc[common_teams, FEATURE_COLS].reset_index()
    result["predicted_pts"] = preds
    result = result.merge(actual, on="team")
    result["error"] = result["predicted_pts"] - result["actual_pts"]
    result["abs_error"] = result["error"].abs()
    result["input_season"] = input_season
    result["target_season"] = target_season

    result = result.sort_values("predicted_pts", ascending=False).reset_index(drop=True)
    result["predicted_rank"] = np.arange(1, len(result) + 1)

    actual_ranked = result[["team", "actual_pts", "actual_goal_diff", "actual_goals_for"]].copy()
    actual_ranked = actual_ranked.rename(
        columns={
            "actual_pts": "points",
            "actual_goal_diff": "goal_diff",
            "actual_goals_for": "goals_for",
        }
    )
    actual_ranked = rank_table(actual_ranked)
    result = result.merge(actual_ranked[["team", "rank"]], on="team")
    result = result.rename(columns={"rank": "actual_rank"})

    pred_champ = result.loc[result["predicted_rank"].idxmin(), "team"]
    actual_champ = result.loc[result["actual_rank"].idxmin(), "team"]

    summary = {
        "target_season": target_season,
        "input_season": input_season,
        "teams_predicted": len(result),
        "training_rows": len(X_train),
        "training_transitions": train_meta[["input_season", "target_season"]]
        .drop_duplicates()
        .shape[0],
        "mae": result["abs_error"].mean(),
        "rmse": np.sqrt(np.mean(result["error"] ** 2)),
        "rank_rho": safe_spearman(result["predicted_rank"], result["actual_rank"]),
        "pred_champ": pred_champ,
        "actual_champ": actual_champ,
        "champ_correct": pred_champ == actual_champ,
    }

    feature_error_corr = {}
    for feature in FEATURE_COLS:
        if result[feature].std() > 0 and result["abs_error"].std() > 0:
            feature_error_corr[feature] = result[feature].corr(result["abs_error"])
        else:
            feature_error_corr[feature] = 0.0
    feature_error_corr["target_season"] = target_season

    permutation_importance = permutation_importance_for_backtest(
        trained,
        scaler,
        X_eval,
        result["actual_pts"].values,
    )
    permutation_importance["target_season"] = target_season

    return result, summary, feature_error_corr, permutation_importance


def permutation_importance_for_backtest(trained, scaler, X_eval, y_actual):
    rng = np.random.default_rng(RANDOM_SEED)
    baseline_pred = ensemble_predict(trained, scaler, X_eval)
    baseline_mae = np.abs(baseline_pred - y_actual).mean()

    importances = {}
    for feature in FEATURE_COLS:
        deltas = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = X_eval.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].values)
            shuffled_pred = ensemble_predict(trained, scaler, shuffled)
            shuffled_mae = np.abs(shuffled_pred - y_actual).mean()
            deltas.append(shuffled_mae - baseline_mae)
        importances[feature] = max(float(np.mean(deltas)), 0.0)

    return importances


def run_all_backtests(feats):
    all_results = []
    summary_rows = []
    error_corr_rows = []
    importance_rows = []

    print(f"\n  Backtesting targets: {', '.join(BACKTEST_TARGET_SEASONS)}")
    print("  Method: leave the whole target season out of training")

    for target_season in BACKTEST_TARGET_SEASONS:
        if target_season not in set(feats["season"]):
            print(f"  skip {target_season}: season is not loaded")
            continue

        result, summary, error_corr, importance = backtest_one_season(feats, target_season)
        all_results.append(result)
        summary_rows.append(summary)
        error_corr_rows.append(error_corr)
        importance_rows.append(importance)

        status = "correct" if summary["champ_correct"] else "missed"
        print(
            f"  {target_season}: MAE={summary['mae']:.1f} pts, "
            f"rank_rho={summary['rank_rho']:.2f}, champ {status} "
            f"({summary['pred_champ']} vs {summary['actual_champ']})"
        )

    return (
        all_results,
        pd.DataFrame(summary_rows),
        pd.DataFrame(error_corr_rows),
        pd.DataFrame(importance_rows),
    )


def normalize_mean_one(series):
    values = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    values = values.clip(lower=0)
    if values.mean() <= 0:
        return pd.Series(1.0, index=series.index)
    return values / values.mean()


def derive_hyperweighted_feature_importance(feats, error_corr_df, importance_df):
    X_all, y_all, _, _ = build_lag_dataset(
        feats,
        allowed_target_seasons=BACKTEST_TARGET_SEASONS,
    )

    perm_importance = importance_df[FEATURE_COLS].mean().fillna(0)
    next_season_signal = X_all.corrwith(y_all).abs().fillna(0)
    mean_error_corr = error_corr_df[FEATURE_COLS].mean().fillna(0)

    importance_score = normalize_mean_one(perm_importance)
    signal_score = normalize_mean_one(next_season_signal)

    # Positive error correlation means the feature was associated with larger
    # misses; negative correlation means the feature was associated with
    # smaller misses. Keep this moderate before the hyperweight exponent.
    penalty = mean_error_corr.clip(lower=0)
    reward = (-mean_error_corr).clip(lower=0)
    reliability = (1 - 0.65 * penalty).clip(lower=0.25) * (1 + 0.35 * reward)

    raw_weight = (0.70 * importance_score + 0.30 * signal_score) * reliability
    base_weight = normalize_mean_one(raw_weight)
    hyperweight = normalize_mean_one(base_weight.pow(HYPERWEIGHT_POWER).clip(0.20, 4.00))

    report = pd.DataFrame(
        {
            "Permutation Importance": perm_importance,
            "Next-Season Signal": next_season_signal,
            "Error Corr": mean_error_corr,
            "Reliability": reliability,
            "Base Weight": base_weight,
            "Hyperweight": hyperweight,
        }
    )
    report = report.sort_values("Hyperweight", ascending=False)

    return hyperweight.reindex(FEATURE_COLS), report


# ---------------------------------------------------------------------------
# 2026-27 PREDICTION
# ---------------------------------------------------------------------------
def promoted_points_baseline(feats):
    promoted_points = []
    for input_season, target_season in iter_lag_transitions(
        feats,
        allowed_target_seasons=BACKTEST_TARGET_SEASONS,
    ):
        input_teams = set(feats.loc[feats["season"] == input_season, "team"])
        target_rows = feats[feats["season"] == target_season]
        promoted = target_rows[~target_rows["team"].isin(input_teams)]
        promoted_points.extend(promoted["points"].astype(float).tolist())

    if not promoted_points:
        return 34.0

    return float(np.median(promoted_points))


def build_prediction_pool(feats):
    latest_table = current_table(feats, PREDICTION_INPUT_SEASON)
    returning = latest_table.head(17)["team"].tolist()
    relegated = latest_table.tail(3)["team"].tolist()

    latest_rows = feats[feats["season"] == PREDICTION_INPUT_SEASON].set_index("team")
    prediction_rows = latest_rows.loc[returning, FEATURE_COLS].copy()
    prediction_rows.insert(0, "team", returning)
    prediction_rows["source"] = f"{PREDICTION_INPUT_SEASON} EPL stats"

    return prediction_rows.reset_index(drop=True), returning, relegated


def predict_2627(feats, feature_weights):
    X, y, _, _ = build_lag_dataset(feats, allowed_target_seasons=BACKTEST_TARGET_SEASONS)
    trained, scaler = train_models(X, y, feature_weights=feature_weights)

    prediction_rows, returning, relegated = build_prediction_pool(feats)
    returning_preds = ensemble_predict(
        trained,
        scaler,
        prediction_rows[FEATURE_COLS],
        feature_weights=feature_weights,
    )

    results = prediction_rows[["team", "source"]].copy()
    results["Proj. Pts"] = returning_preds

    promoted_baseline = promoted_points_baseline(feats)
    promoted_rows = []
    for idx, team in enumerate(PROMOTED_TEAMS_2627):
        # The repo has no Championship feature rows. Use historical promoted
        # club outcomes, with a small ordering adjustment so automatic/playoff
        # promotions do not all tie by construction.
        ordering_adjustment = [2.0, 0.5, -1.5][idx] if idx < 3 else 0.0
        promoted_rows.append(
            {
                "team": team,
                "source": "promoted baseline",
                "Proj. Pts": promoted_baseline + ordering_adjustment,
            }
        )

    if promoted_rows:
        results = pd.concat([results, pd.DataFrame(promoted_rows)], ignore_index=True)

    results["Proj. Pts"] = results["Proj. Pts"].round(1)
    results = results.sort_values(["Proj. Pts", "team"], ascending=[False, True]).reset_index(drop=True)
    results["Rank"] = np.arange(1, len(results) + 1)

    shifted = results["Proj. Pts"].values - results["Proj. Pts"].min()
    prob = shifted**2
    if prob.sum() == 0:
        win_prob = np.repeat(100 / len(results), len(results))
    else:
        win_prob = prob / prob.sum() * 100
    results["Win Prob (%)"] = np.round(win_prob, 1)

    return results[["Rank", "team", "Proj. Pts", "Win Prob (%)", "source"]], returning, relegated


# ---------------------------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------------------------
def save_csv_outputs(all_results, summary_df, feature_report, prediction):
    backtest_rows = []
    for result in all_results:
        backtest_rows.append(
            result[
                [
                    "target_season",
                    "input_season",
                    "team",
                    "predicted_rank",
                    "predicted_pts",
                    "actual_rank",
                    "actual_pts",
                    "error",
                    "abs_error",
                ]
            ].copy()
        )

    backtests = pd.concat(backtest_rows, ignore_index=True)
    paths = {
        "backtests": os.path.join(OUT_DIR, "epl_backtest_results.csv"),
        "summary": os.path.join(OUT_DIR, "epl_backtest_summary.csv"),
        "features": os.path.join(OUT_DIR, "epl_hyperweighted_feature_importance.csv"),
        "prediction": os.path.join(OUT_DIR, "epl_2026_27_prediction.csv"),
    }

    backtests.to_csv(paths["backtests"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    feature_report.to_csv(paths["features"])
    prediction.to_csv(paths["prediction"], index=False)

    return paths


def plot_backtest_summary(summary_df, importance_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    ax = axes[0]
    colors = [GOLD if ok else RED for ok in summary_df["champ_correct"]]
    bars = ax.bar(summary_df["target_season"], summary_df["mae"], color=colors, width=0.58)
    for bar, mae in zip(bars, summary_df["mae"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{mae:.1f}",
            color=WHITE,
            ha="center",
            fontsize=9,
        )
    ax.set_title("Backtest MAE by Target Season", color=WHITE, fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean absolute error, points", color=GREY)
    ax.tick_params(colors=WHITE, labelrotation=25)
    ax.spines[:].set_visible(False)
    ax.set_ylim(0, max(summary_df["mae"]) * 1.25)

    ax = axes[1]
    mean_importance = importance_df[FEATURE_COLS].mean().sort_values().tail(12)
    ax.barh(mean_importance.index, mean_importance.values, color=BLUE)
    ax.set_title("Top Backtest Permutation Importance", color=WHITE, fontsize=12, fontweight="bold")
    ax.set_xlabel("MAE increase when shuffled", color=GREY)
    ax.tick_params(colors=WHITE)
    ax.spines[:].set_visible(False)

    fig.suptitle("Multi-Season Backtest: 2020-21 through 2025-26", color=WHITE, fontsize=14)
    plt.tight_layout()
    out = os.path.join(out_dir, "epl_backtest_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  chart -> {out}")


def plot_feature_weights(feature_report, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    weights = feature_report["Hyperweight"].sort_values()
    colors = [GREEN if val >= 1 else RED for val in weights.values]

    ax = axes[0]
    ax.barh(weights.index, weights.values, color=colors)
    ax.axvline(1.0, color=GREY, linestyle="--", linewidth=1)
    for y, val in enumerate(weights.values):
        ax.text(val + 0.02, y, f"{val:.2f}x", color=WHITE, va="center", fontsize=8)
    ax.set_title("Hyperweighted Feature Importance", color=WHITE, fontsize=12, fontweight="bold")
    ax.set_xlabel("Final multiplier", color=GREY)
    ax.tick_params(colors=WHITE)
    ax.spines[:].set_visible(False)
    ax.set_xlim(0, max(weights.max() * 1.22, 1.2))

    ax = axes[1]
    x = np.arange(len(feature_report))
    ordered = feature_report.sort_values("Hyperweight", ascending=False)
    ax.bar(x, ordered["Permutation Importance"], color=BLUE, alpha=0.8, label="Permutation")
    ax.plot(x, ordered["Next-Season Signal"], color=GOLD, linewidth=2, marker="o", label="Signal")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered.index, rotation=55, ha="right", fontsize=8, color=WHITE)
    ax.set_title("Importance vs Raw Next-Season Signal", color=WHITE, fontsize=12, fontweight="bold")
    ax.tick_params(colors=WHITE)
    ax.spines[:].set_visible(False)
    ax.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)

    fig.suptitle("Backtest-Derived Feature Calibration", color=WHITE, fontsize=14)
    plt.tight_layout()
    out = os.path.join(out_dir, "epl_feature_weights.png")
    legacy_out = os.path.join(out_dir, "epl_feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.savefig(legacy_out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  chart -> {out}")
    print(f"  chart -> {legacy_out}")


def plot_final_prediction(prediction, out_dir):
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    plot_df = prediction.sort_values("Rank", ascending=False)
    colors = [
        GOLD if rank == 1 else (GREY if source == "promoted baseline" else BLUE)
        for rank, source in zip(plot_df["Rank"], plot_df["source"])
    ]

    ax.barh(plot_df["team"], plot_df["Proj. Pts"], color=colors)
    for y, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(row["Proj. Pts"] + 0.4, y, f"{row['Proj. Pts']:.1f}", color=WHITE, va="center")

    ax.set_title(f"{PREDICTION_SEASON} Predicted Premier League Table", color=WHITE, fontsize=14)
    ax.set_xlabel("Projected points", color=GREY)
    ax.tick_params(colors=WHITE)
    ax.spines[:].set_visible(False)
    ax.set_xlim(0, max(prediction["Proj. Pts"]) * 1.12)

    plt.tight_layout()
    out = os.path.join(out_dir, "epl_prediction.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  chart -> {out}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 68)
    print("  EPL PREDICTOR v5 - multi-season hyperweighted backtest")
    print("=" * 68)

    print("\n[1/7] Loading seasons")
    raw = load_all(SEASON_FILES)
    print(f"  total matches loaded: {len(raw)}")

    print("\n[2/7] Engineering team-season features")
    feats = team_season_features(raw)
    print(f"  seasons: {', '.join(sorted_seasons(feats['season'].unique()))}")
    print(f"  team-seasons: {len(feats)}")

    print("\n[3/7] Running season-holdout backtests")
    all_results, summary_df, error_corr_df, importance_df = run_all_backtests(feats)

    if summary_df.empty:
        raise RuntimeError("No backtests ran. Check season files and target seasons.")

    print("\nBacktest summary")
    display_summary = summary_df[
        [
            "target_season",
            "input_season",
            "teams_predicted",
            "mae",
            "rank_rho",
            "pred_champ",
            "actual_champ",
            "champ_correct",
        ]
    ].copy()
    display_summary["mae"] = display_summary["mae"].round(1)
    display_summary["rank_rho"] = display_summary["rank_rho"].round(2)
    print(display_summary.to_string(index=False))
    print(f"\n  overall MAE:       {summary_df['mae'].mean():.1f} pts")
    print(f"  overall rank rho:  {summary_df['rank_rho'].mean():.2f}")
    print(f"  champion accuracy: {summary_df['champ_correct'].mean() * 100:.0f}%")

    print("\n[4/7] Deriving hyperweighted feature importance")
    feature_weights, feature_report = derive_hyperweighted_feature_importance(
        feats,
        error_corr_df,
        importance_df,
    )

    print("\nTop hyperweighted features")
    for feature, row in feature_report.head(12).iterrows():
        print(
            f"  {feature:22s}  weight={row['Hyperweight']:.2f}x  "
            f"perm={row['Permutation Importance']:.3f}  "
            f"signal={row['Next-Season Signal']:.3f}"
        )

    print(f"\n[5/7] Predicting {PREDICTION_SEASON}")
    prediction, returning, relegated = predict_2627(feats, feature_weights)
    print(f"  returning clubs from {PREDICTION_INPUT_SEASON}: {len(returning)}")
    print(f"  relegated by local {PREDICTION_INPUT_SEASON} table: {', '.join(relegated)}")
    print(f"  promoted baseline clubs: {', '.join(PROMOTED_TEAMS_2627)}")

    print("\n" + "=" * 68)
    print(f"  {PREDICTION_SEASON} PREDICTED TABLE")
    print("=" * 68)
    print(prediction.to_string(index=False))

    print("\n[6/7] Saving CSV outputs")
    paths = save_csv_outputs(all_results, summary_df, feature_report, prediction)
    for label, path in paths.items():
        print(f"  {label:10s} -> {path}")

    print("\n[7/7] Generating charts")
    plot_backtest_summary(summary_df, importance_df, OUT_DIR)
    plot_feature_weights(feature_report, OUT_DIR)
    plot_final_prediction(prediction, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
