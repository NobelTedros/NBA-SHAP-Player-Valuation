"""
NBA SHAP Project — Phase 3 v2: Trade Compatibility Model (Patched)
===================================================================
Fixes three issues from v1:
  1. Added role-specific features (usage proxy, efficiency, role score)
  2. Added roster gap penalty — penalizes teams that already have
     a player filling the same role as the candidate
  3. Replaced TEAM_ARCHETYPE (which had 0 importance) with a
     pace/ball-movement score built from existing team features

Run:
    python trade_model_v2.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

INPUT_FILE    = "nba_feature_matrix_labeled.csv"
MODEL_FILE    = "trade_model_v2.pkl"
RANKINGS_FILE = "trade_rankings_v2.csv"
TARGET        = "LABEL_WS48_CHG"

TEAM_NAMES = {
    1610612737: "Atlanta Hawks",       1610612738: "Boston Celtics",
    1610612739: "Cleveland Cavaliers", 1610612740: "New Orleans Pelicans",
    1610612741: "Chicago Bulls",       1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",      1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",     1610612746: "Los Angeles Clippers",
    1610612747: "Los Angeles Lakers",  1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",     1610612750: "Minnesota Timberwolves",
    1610612751: "Brooklyn Nets",       1610612752: "New York Knicks",
    1610612753: "Orlando Magic",       1610612754: "Indiana Pacers",
    1610612755: "Philadelphia 76ers",  1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers", 1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",   1610612760: "Oklahoma City Thunder",
    1610612761: "Toronto Raptors",     1610612762: "Utah Jazz",
    1610612763: "Memphis Grizzlies",   1610612764: "Washington Wizards",
    1610612765: "Detroit Pistons",     1610612766: "Charlotte Hornets",
}


# ── Fix 1: Engineer role-aware features ──────────────────────────────────────

def add_player_role_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    USAGE_PROXY:    PTS + AST — ball dominance, continuous
    EFFICIENCY:     PTS / TOV — scoring efficiency
    DEFENSIVE_ROLE: STL + BLK — defensive identity
                    Replaces ROLE_SCORE which was redundant with USAGE_PROXY
    """
    df = df.copy()
    df["USAGE_PROXY"]    = df["PTS"] + df["AST"]
    df["EFFICIENCY"]     = df["PTS"] / (df["TOV"].clip(lower=0.1))
    df["DEFENSIVE_ROLE"] = df["STL"] + df["BLK"]
    return df


# ── Fix 2: Roster gap penalty ────────────────────────────────────────────────

def compute_roster_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates roster gap using USAGE_PROXY (ball dominance)
    instead of the removed ROLE_SCORE.

    ROSTER_GAP = candidate USAGE_PROXY - team average USAGE_PROXY
    Positive = candidate is more ball-dominant than team average (fills need)
    Negative = team already has ball-dominant players (redundant)
    """
    df = df.copy()

    team_usage_avg = (
        df.groupby(["TEAM_ID", "SEASON"])["USAGE_PROXY"]
        .mean()
        .reset_index()
        .rename(columns={"USAGE_PROXY": "TEAM_AVG_USAGE"})
    )
    df = df.merge(team_usage_avg, on=["TEAM_ID", "SEASON"], how="left")
    df["ROSTER_GAP"] = df["USAGE_PROXY"] - df["TEAM_AVG_USAGE"]
    return df

# ── Fix 3: Replace broken TEAM_ARCHETYPE with pace score ─────────────────────

def add_team_system_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TEAM_ARCHETYPE had 0.000 importance because K-Means clusters
    didn't correlate with trade outcomes.

    We replace it with two interpretable system features:

    TEAM_PACE_SCORE:   TEAM_PTS + TEAM_AST normalized
                       High = up-tempo ball-movement system
                       Low  = slow/isolation-heavy system

    TEAM_DEF_SCORE:    TEAM_STL + TEAM_BLK normalized
                       High = defensive-identity team
                       Low  = offense-first team

    These are meaningful for fit: a pass-first point guard
    fits better on a high TEAM_PACE_SCORE team. A rim protector
    fits better on a high TEAM_DEF_SCORE team.
    """
    df = df.copy()
    df["TEAM_PACE_SCORE"] = df["TEAM_PTS"] + df["TEAM_AST"]
    df["TEAM_DEF_SCORE"]  = df["TEAM_STL"] + df["TEAM_BLK"]
    return df


# ── Updated feature set ───────────────────────────────────────────────────────

PLAYER_FEATURES = [
    "PLUS_MINUS", "PTS", "AST", "REB", "STL", "BLK", "TOV", "MIN",
    "USAGE_PROXY", "EFFICIENCY", "DEFENSIVE_ROLE"
]

TEAM_FEATURES = [
    "TEAM_W_PCT", "TEAM_PLUS_MINUS",
    "TEAM_PTS", "TEAM_AST", "TEAM_REB", "TEAM_STL", "TEAM_BLK",
    # New system features (replaces broken TEAM_ARCHETYPE)
    "TEAM_PACE_SCORE", "TEAM_DEF_SCORE",
    # Roster gap (Fix 2)
    "ROSTER_GAP"
]

ALL_FEATURES = PLAYER_FEATURES + TEAM_FEATURES


# ── Pipeline ──────────────────────────────────────────────────────────────────

def prepare_data(path: str):
    print("Loading and engineering features...")
    df = pd.read_csv(path)

    # Apply all three fixes
    df = add_player_role_features(df)
    df = compute_roster_gap(df)
    df = add_team_system_features(df)

    # Filter to traded players only
    trade_df = df[df[TARGET].notna()].copy()
    print(f"  Trade rows: {len(trade_df):,}")

    present  = [f for f in ALL_FEATURES if f in trade_df.columns]
    missing  = [f for f in ALL_FEATURES if f not in trade_df.columns]
    if missing:
        print(f"  Missing features: {missing}")

    X = trade_df[present]
    y = trade_df[TARGET]
    print(f"  Feature matrix: {X.shape}")
    return df, trade_df, X, y


def temporal_split(trade_df, X, y):
    print("\nTemporal split (train: 2018-2022, test: 2022-23)...")
    test_mask  = trade_df["SEASON"] == "2022-23"
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    print("\nTraining XGBoost v2...")
    model = xgb.XGBRegressor(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.6,
        colsample_bytree=0.6,
        min_child_weight=8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    print("  Done")
    return model


def evaluate(model, X_train, X_test, y_train, y_test):
    print("\nModel evaluation:")
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)
    print(f"  Train  MAE: {mean_absolute_error(y_train, train_preds):.3f}  "
          f"R²: {r2_score(y_train, train_preds):.3f}")
    print(f"  Test   MAE: {mean_absolute_error(y_test, test_preds):.3f}  "
          f"R²: {r2_score(y_test, test_preds):.3f}")

    gap = r2_score(y_train, train_preds) - r2_score(y_test, test_preds)
    if gap > 0.3:
        print(f"  Note: still some overfitting — gap is {gap:.2f}")
    else:
        print("  Train/test gap looks healthy.")


def show_importance(model, feature_names):
    print("\nFeature importance:")
    pairs = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for name, score in pairs:
        bar = "█" * int(score * 40)
        print(f"  {name:<22} {bar} {score:.3f}")


def rank_teams(model, full_df: pd.DataFrame, player_name: str):
    print(f"\nRanking teams for: {player_name}")

    player_rows = full_df[
        full_df["PLAYER_NAME"].str.contains(player_name, case=False, na=False)
    ]
    if len(player_rows) == 0:
        print(f"  '{player_name}' not found.")
        return None

    player_row = player_rows.sort_values("SEASON").iloc[-1]
    print(f"  Stats from season: {player_row['SEASON']}")
    print(f"  Defensive role: {player_row.get('DEFENSIVE_ROLE', 'N/A'):.2f}  "
          f"| Usage proxy: {player_row.get('USAGE_PROXY', 'N/A'):.2f}")

    # Get most recent team system vector per team
    team_df = full_df.drop_duplicates(["TEAM_ID", "SEASON"])
    team_df = team_df.sort_values("SEASON").drop_duplicates("TEAM_ID", keep="last")

    present_p = [f for f in PLAYER_FEATURES if f in full_df.columns]
    present_t = [f for f in TEAM_FEATURES   if f in full_df.columns]

    rows = []
    for _, team_row in team_df.iterrows():
        row = {f: player_row[f] for f in present_p if f in player_row}
        row.update({f: team_row[f] for f in present_t if f in team_row})

        # Compute roster gap for this specific team
        row["ROSTER_GAP"] = (
            player_row.get("USAGE_PROXY", 0) - team_row.get("TEAM_AVG_USAGE", 0)
        )
        row["TEAM_ID_REF"] = team_row["TEAM_ID"]
        rows.append(row)

    scoring_df = pd.DataFrame(rows)
    feat_cols  = [f for f in ALL_FEATURES if f in scoring_df.columns]
    scores     = model.predict(scoring_df[feat_cols])

    scoring_df["COMPATIBILITY_SCORE"] = scores
    min_s, max_s = scores.min(), scores.max()
    scoring_df["SCORE_0_100"] = (
        ((scores - min_s) / (max_s - min_s) * 100)
        if max_s > min_s else np.full(len(scores), 50.0)
    )

    ranked = scoring_df.sort_values("COMPATIBILITY_SCORE", ascending=False)
    ranked = ranked.reset_index(drop=True)
    ranked.index += 1

    print(f"\n  Top 10 team fits for {player_name}:")
    print(f"  {'Rank':<6} {'Team':<28} {'Score':<10} {'Pred gain':<12} {'Roster gap'}")
    print(f"  {'-'*66}")
    for i, row in ranked.head(10).iterrows():
        team_name  = TEAM_NAMES.get(int(row["TEAM_ID_REF"]), str(int(row["TEAM_ID_REF"])))
        score      = round(row["SCORE_0_100"], 1)
        gain       = row["COMPATIBILITY_SCORE"]
        roster_gap = row.get("ROSTER_GAP", 0)
        gap_label  = f"+{roster_gap:.1f} (fills need)" if roster_gap > 0 else f"{roster_gap:.1f} (redundant)"
        print(f"  {i:<6} {team_name:<28} {score:<10} {gain:<+12.3f} {gap_label}")

    return ranked


def main():
    print("=== NBA SHAP Project — Phase 3 v2: Trade Model (Patched) ===\n")

    full_df, trade_df, X, y = prepare_data(INPUT_FILE)
    X_train, X_test, y_train, y_test = temporal_split(trade_df, X, y)
    model = train_model(X_train, y_train)
    evaluate(model, X_train, X_test, y_train, y_test)

    present_features = [f for f in ALL_FEATURES if f in X.columns]
    show_importance(model, present_features)

    # Try multiple players to show differentiation
    for player in ["Kyrie Irving", "Markieff Morris", "Russell Westbrook"]:
        rankings = rank_teams(model, full_df, player)
        if rankings is not None:
            rankings.to_csv(
                f"trade_rankings_{player.split()[0].lower()}.csv", index=True
            )

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {MODEL_FILE}")
    print("\n=== Done. Ready for Phase 4 — SHAP development model ===")


if __name__ == "__main__":
    main()