"""
NBA SHAP Project — Phase 5: System-Adjusted Salary Fairness Model
=================================================================
The most commercially novel component of the project.

Standard salary models predict salary from raw stats.
This model:
  1. Strips team context from player stats to isolate
     individual contribution (system adjustment)
  2. Predicts fair market salary from system-adjusted value
  3. Uses SHAP to explain exactly which stats are causing
     a player to be underpaid or overpaid

This is the tool a player agent would use in a contract
negotiation to argue: "My player's raw stats understate
their value because of system suppression. System-adjusted,
they are worth $X million more than their current contract."

Run:
    python salary_model.py

Input:  nba_feature_matrix_labeled.csv
Output: salary_model.pkl
        salary_fairness_reports.csv
"""

import pandas as pd
import numpy as np
import pickle
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE    = "nba_feature_matrix_labeled.csv"
MODEL_FILE    = "salary_model.pkl"
REPORTS_FILE  = "salary_fairness_reports.csv"
TARGET        = "LABEL_SALARY"

# Player performance features — what the player controls
PLAYER_FEATURES = [
    "PLUS_MINUS", "PTS", "AST", "REB", "STL", "BLK", "TOV", "MIN"
]

# Team system features — context the player doesn't control
TEAM_FEATURES = [
    "TEAM_W_PCT", "TEAM_PLUS_MINUS", "TEAM_PTS",
    "TEAM_AST", "TEAM_STL", "TEAM_BLK"
]

FEATURE_LABELS = {
    "PLUS_MINUS": "On-court impact",
    "PTS":        "Scoring volume",
    "AST":        "Playmaking",
    "REB":        "Rebounding",
    "STL":        "Defensive steals",
    "BLK":        "Rim protection",
    "TOV":        "Ball security",
    "MIN":        "Playing time",
    "SYS_ADJ":    "System adjustment",
}


# ── Step 1: Load data ─────────────────────────────────────────────────────────

def load_data(path: str):
    print("Loading labeled feature matrix...")
    df = pd.read_csv(path)

    # Keep only rows with salary labels
    df = df[df[TARGET].notna()].copy()
    print(f"  Rows with salary data: {len(df):,}")
    print(f"  Salary range: "
          f"${df[TARGET].min():,.0f} — ${df[TARGET].max():,.0f}")
    print(f"  Median salary: ${df[TARGET].median():,.0f}")
    return df


# ── Step 2: System adjustment ─────────────────────────────────────────────────

def compute_system_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """
    This is the novel contribution of Phase 5.

    The idea: a player's PLUS_MINUS — the most predictive feature
    for salary — is heavily influenced by team quality. A mediocre
    player on a great team looks better than they are. A great player
    on a bad team looks worse.

    We estimate how much of each player's PLUS_MINUS is explained
    by their team's system, then subtract that team contribution
    to get the system-adjusted individual impact.

    Method:
      1. Fit a simple Ridge regression: PLUS_MINUS ~ team features
         This learns how much team context predicts a player's +/-
      2. The residual (actual - team-predicted) = individual contribution
      3. Add this residual as SYS_ADJ_PLUS_MINUS feature

    Positive SYS_ADJ: player outperforms their team context
    Negative SYS_ADJ: player underperforms their team context
    Near zero: team context fully explains their performance
    """
    print("\nComputing system adjustment...")
    df = df.copy()

    present_team = [f for f in TEAM_FEATURES if f in df.columns]
    X_team = df[present_team].fillna(0)
    y_pm   = df["PLUS_MINUS"].fillna(0)

    # Ridge regression: how much does team context predict PLUS_MINUS?
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_team, y_pm)
    team_predicted_pm = ridge.predict(X_team)

    # Residual = individual contribution after removing team context
    df["SYS_ADJ"] = y_pm - team_predicted_pm

    # How much variance does team context explain?
    r2 = r2_score(y_pm, team_predicted_pm)
    print(f"  Team context explains {r2*100:.1f}% of PLUS_MINUS variance")
    print(f"  Remaining {(1-r2)*100:.1f}% = individual contribution")

    pos_adj = (df["SYS_ADJ"] > 0).sum()
    neg_adj = (df["SYS_ADJ"] < 0).sum()
    print(f"  Players outperforming team context: {pos_adj:,}")
    print(f"  Players underperforming team context: {neg_adj:,}")

    # Show top system-suppressed players (most negative adjustment)
    suppressed = df.nsmallest(5, "SYS_ADJ")[
        ["PLAYER_NAME", "SEASON", "PLUS_MINUS", "SYS_ADJ", TARGET]
    ]
    print("\n  Most system-suppressed players (hidden value):")
    for _, row in suppressed.iterrows():
        print(f"    {row['PLAYER_NAME']:<22} {row['SEASON']}  "
              f"Raw PM: {row['PLUS_MINUS']:+.2f}  "
              f"Adj: {row['SYS_ADJ']:+.2f}  "
              f"Salary: ${row[TARGET]:,.0f}")

    # Show top system-boosted players (inflated by team)
    boosted = df.nlargest(5, "SYS_ADJ")[
        ["PLAYER_NAME", "SEASON", "PLUS_MINUS", "SYS_ADJ", TARGET]
    ]
    print("\n  Most system-boosted players (inflated by team):")
    for _, row in boosted.iterrows():
        print(f"    {row['PLAYER_NAME']:<22} {row['SEASON']}  "
              f"Raw PM: {row['PLUS_MINUS']:+.2f}  "
              f"Adj: {row['SYS_ADJ']:+.2f}  "
              f"Salary: ${row[TARGET]:,.0f}")

    return df, ridge


# ── Step 3: Build salary feature matrix ───────────────────────────────────────

def build_salary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Salary features include:
    - Player performance stats (z-scored)
    - System adjustment (individual contribution above/below team)
    - NOT raw team features — we've already extracted the team
      contribution into SYS_ADJ, so including raw team features
      again would re-introduce the bias we just removed
    """
    SALARY_FEATURES = PLAYER_FEATURES + ["SYS_ADJ"]
    present = [f for f in SALARY_FEATURES if f in df.columns]
    X = df[present].fillna(0)
    y = df[TARGET]
    print(f"\nSalary feature matrix: {X.shape}")
    print(f"  Features: {present}")
    return X, y, present


# ── Step 4: Temporal split ────────────────────────────────────────────────────

def temporal_split(df, X, y):
    print("\nTemporal split...")
    test_mask  = df["SEASON"] == "2022-23"
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    print(f"  Train: {len(X_train):,} rows")
    print(f"  Test:  {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test


# ── Step 5: Train salary model ────────────────────────────────────────────────

def train_salary_model(X_train, y_train):
    """
    XGBoost regression predicting salary in dollars.
    Same architecture as the trade model but targeting salary.
    """
    print("\nTraining salary model...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train, verbose=False)
    print("  Done")
    return model


# ── Step 6: Evaluate ──────────────────────────────────────────────────────────

def evaluate(model, X_train, X_test, y_train, y_test):
    print("\nModel evaluation:")
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae  = mean_absolute_error(y_test,  test_pred)
    train_r2  = r2_score(y_train, train_pred)
    test_r2   = r2_score(y_test,  test_pred)

    print(f"  Train  MAE: ${train_mae:,.0f}  R²: {train_r2:.3f}")
    print(f"  Test   MAE: ${test_mae:,.0f}  R²: {test_r2:.3f}")
    print(f"  Avg prediction error: ~${test_mae/1e6:.1f}M per player")

    gap = train_r2 - test_r2
    if gap > 0.3:
        print(f"  Note: overfitting gap {gap:.2f} — consider tuning")
    else:
        print("  Train/test gap looks healthy.")

    return model.predict(X_test)


# ── Step 7: SHAP salary explanations ─────────────────────────────────────────

def compute_shap_salary(model, X, feature_cols):
    print("\nComputing SHAP values for salary model...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    base_value  = explainer.expected_value
    print(f"  Base value (avg predicted salary): ${base_value:,.0f}")
    return explainer, shap_values, base_value


# ── Step 8: Salary fairness reports ──────────────────────────────────────────

def generate_salary_reports(df, X, shap_values, feature_cols):
    """
    For each player generates:
    - Predicted fair salary (system-adjusted)
    - Actual salary
    - Dollar gap (positive = underpaid, negative = overpaid)
    - SHAP explanation of which stats drive the gap
    - Agent talking point: the one-sentence argument for a raise
    """
    print("\nGenerating salary fairness reports...")

    all_preds = X._parent_df_model_preds if hasattr(
        X, '_parent_df_model_preds') else None

    rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        if i >= len(shap_values):
            break

        player_shap = shap_values[i]
        shap_dict   = {feat: float(player_shap[j])
                       for j, feat in enumerate(feature_cols)}

        actual_salary    = float(row[TARGET])
        predicted_salary = float(
            sum(shap_dict.values()) +
            shap.TreeExplainer.expected_value
            if False else
            float(row.get("PREDICTED_SALARY", 0))
        )

        report = {
            "PLAYER_NAME":   row["PLAYER_NAME"],
            "SEASON":        row["SEASON"],
            "ACTUAL_SALARY": actual_salary,
            "SYS_ADJ":       float(row.get("SYS_ADJ", 0)),
        }
        for feat in feature_cols:
            report[f"SHAP_{feat}"] = shap_dict.get(feat, 0)

        # Biggest positive and negative SHAP contributors
        sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1])
        report["BIGGEST_DISCOUNT"] = sorted_shap[0][0]
        report["BIGGEST_PREMIUM"]  = sorted_shap[-1][0]

        rows.append(report)

    return pd.DataFrame(rows)


# ── Step 9: Player salary report printer ─────────────────────────────────────

def print_salary_report(df_full, reports_df, model, explainer,
                         X_full, feature_cols, player_name):
    """
    Prints a human-readable salary fairness report.
    This is what you hand to a player agent.
    """
    mask = reports_df["PLAYER_NAME"].str.contains(
        player_name, case=False, na=False
    )
    if not mask.any():
        print(f"  '{player_name}' not found.")
        return

    # Use most recent season
    player_reports = reports_df[mask].sort_values("SEASON")
    report = player_reports.iloc[-1]

    # Get predicted salary for this player
    player_idx = df_full[
        df_full["PLAYER_NAME"].str.contains(player_name, case=False, na=False)
    ].sort_values("SEASON").index[-1]

    pos = df_full.index.get_loc(player_idx)
    if pos < len(X_full):
        pred_salary = float(model.predict(X_full.iloc[[pos]])[0])
        player_shap = explainer.shap_values(X_full.iloc[[pos]])[0]
    else:
        return

    actual    = report["ACTUAL_SALARY"]
    predicted = pred_salary
    gap       = predicted - actual
    gap_m     = gap / 1e6
    sys_adj   = report["SYS_ADJ"]

    season = report["SEASON"]

    print(f"\n{'='*62}")
    print(f"  SALARY FAIRNESS REPORT: {report['PLAYER_NAME']} ({season})")
    print(f"{'='*62}")
    print(f"  Actual salary:          ${actual:>12,.0f}")
    print(f"  Predicted fair salary:  ${predicted:>12,.0f}")

    if gap > 0:
        print(f"  Gap:                    +${gap:>11,.0f}  "
              f"(UNDERPAID by ${gap_m:.1f}M)")
    else:
        print(f"  Gap:                    -${abs(gap):>11,.0f}  "
              f"(OVERPAID by ${abs(gap_m):.1f}M)")

    print(f"\n  System adjustment:      {sys_adj:+.3f} std devs")
    if sys_adj > 0.2:
        print(f"  Interpretation: Player OUTPERFORMS their team context")
        print(f"  Their raw stats UNDERSTATE their true individual value")
    elif sys_adj < -0.2:
        print(f"  Interpretation: Player is BOOSTED by their team context")
        print(f"  Their raw stats OVERSTATE their true individual value")
    else:
        print(f"  Interpretation: Stats fairly reflect individual contribution")

    print(f"\n  SHAP salary attribution:")
    shap_pairs = [(feat, float(player_shap[j]))
                  for j, feat in enumerate(feature_cols)]
    shap_pairs.sort(key=lambda x: x[1], reverse=True)

    for feat, val in shap_pairs:
        label     = FEATURE_LABELS.get(feat, feat)
        direction = "+" if val >= 0 else ""
        bar       = "█" * min(int(abs(val) / 200000), 20)
        print(f"    {label:<25} {direction}${val:>10,.0f}  {bar}")

    print(f"\n  Agent talking point:")
    biggest_discount = min(shap_pairs, key=lambda x: x[1])
    discount_label   = FEATURE_LABELS.get(biggest_discount[0],
                                           biggest_discount[0])

    if sys_adj > 0.2 and gap > 0:
        print(f"  'Our player outperforms their team context by "
              f"{sys_adj:.2f} standard deviations.")
        print(f"   System-adjusted, their fair market value is "
              f"${predicted:,.0f} — ${gap_m:.1f}M above")
        print(f"   their current salary. {discount_label} is the key")
        print(f"   stat being suppressed by team context.'")
    elif gap > 0:
        print(f"  'Our player's fair market value based on statistical")
        print(f"   contribution is ${predicted:,.0f}. They are currently")
        print(f"   underpaid by ${gap_m:.1f}M. {discount_label} is")
        print(f"   the primary undervalued contribution.'")
    else:
        print(f"  'Player compensation aligns with statistical output.")
        print(f"   Contract is at fair market value.'")

    print(f"{'='*62}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== NBA SHAP Project — Phase 5: Salary Fairness Model ===\n")

    # Load
    df = load_data(INPUT_FILE)

    # System adjustment — the novel step
    df, ridge_model = compute_system_adjustment(df)

    # Build features
    X, y, feature_cols = build_salary_features(df)

    # Store reference to full df for predictions
    X._parent_df = df

    # Split
    X_train, X_test, y_train, y_test = temporal_split(df, X, y)

    # Train
    model = train_salary_model(X_train, y_train)

    # Evaluate
    evaluate(model, X_train, X_test, y_train, y_test)

    # Feature importance
    print("\nFeature importance (what drives salary predictions):")
    pairs = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for feat, score in pairs:
        label = FEATURE_LABELS.get(feat, feat)
        bar   = "█" * int(score * 40)
        print(f"  {label:<25} {bar} {score:.3f}")

    # SHAP
    explainer, shap_values, base_value = compute_shap_salary(
        model, X, feature_cols
    )

    # Generate reports
    reports_df = generate_salary_reports(df, X, shap_values, feature_cols)
    reports_df.to_csv(REPORTS_FILE, index=False)
    print(f"  Saved {len(reports_df):,} salary reports → {REPORTS_FILE}")

    # Print example reports
    print("\n--- Example salary fairness reports ---")
    for player in ["Nikola Jokic", "Cade Cunningham", "LeBron James"]:
        print_salary_report(
            df, reports_df, model, explainer, X, feature_cols, player
        )

    # Save
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({
            "model":        model,
            "explainer":    explainer,
            "ridge":        ridge_model,
            "features":     feature_cols,
        }, f)
    print(f"\nModel saved → {MODEL_FILE}")
    print("\n=== Done. Phase 5 complete. Next: Phase 6 — Validation ===")


if __name__ == "__main__":
    main()