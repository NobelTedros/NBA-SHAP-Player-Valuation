"""
NBA SHAP Project — Phase 4: SHAP Development Model
====================================================
Trains an XGBoost classifier to predict All-Star selection,
then uses SHAP to explain which features are holding each
player back from All-Star or MVP classification.

This is the forward-looking development feedback component —
the novel research contribution of this project.

Run:
    pip install shap
    python shap_model.py

Input:  nba_feature_matrix_labeled.csv
Output: shap_model.pkl
        shap_development_reports.csv  (one row per player with SHAP values)
"""

import pandas as pd
import numpy as np
import pickle
import shap
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE   = "nba_feature_matrix_labeled.csv"
MODEL_FILE   = "shap_model.pkl"
REPORTS_FILE = "shap_development_reports.csv"
TARGET       = "LABEL_ALLSTAR"

# ── Features ──────────────────────────────────────────────────────────────────
# We use the same player features as the trade model
# but NOT team features — the SHAP model measures individual
# player skill gaps, independent of team context

FEATURES = [
    "PLUS_MINUS",  # overall on-court impact
    "PTS",         # scoring
    "AST",         # playmaking
    "REB",         # rebounding
    "STL",         # defensive activity
    "BLK",         # rim protection
    "TOV",         # ball security (negative = more turnovers)
    "MIN",         # playing time / coach trust
]

FEATURE_LABELS = {
    "PLUS_MINUS": "On-court impact (Plus/Minus)",
    "PTS":        "Scoring volume",
    "AST":        "Playmaking / assists",
    "REB":        "Rebounding",
    "STL":        "Defensive activity (steals)",
    "BLK":        "Rim protection (blocks)",
    "TOV":        "Ball security (turnovers)",
    "MIN":        "Playing time",
}


# ── Step 1: Load data ─────────────────────────────────────────────────────────

def load_data(path: str):
    print("Loading labeled feature matrix...")
    df = pd.read_csv(path)
    print(f"  Total rows: {len(df):,}")

    present = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  Warning: missing features: {missing}")

    X = df[present].copy()
    y = df[TARGET].fillna(0).astype(int)

    allstar_count = y.sum()
    print(f"  All-Stars: {int(allstar_count):,} / {len(y):,} "
          f"({allstar_count/len(y)*100:.1f}%)")
    return df, X, y


# ── Step 2: Temporal split ────────────────────────────────────────────────────

def temporal_split(df, X, y):
    print("\nTemporal split...")
    test_mask  = df["SEASON"] == "2022-23"
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    print(f"  Train: {len(X_train):,} rows  "
          f"({int(y_train.sum())} All-Stars)")
    print(f"  Test:  {len(X_test):,} rows  "
          f"({int(y_test.sum())} All-Stars)")
    return X_train, X_test, y_train, y_test


# ── Step 3: Train with class weighting ───────────────────────────────────────

def train_model(X_train, y_train):
    """
    Class imbalance: only ~5% of players are All-Stars.
    Without correction, the model would predict 'not All-Star'
    for everyone and be 95% accurate — but completely useless.

    scale_pos_weight tells XGBoost to treat each All-Star row
    as if it were worth N times a non-All-Star row.
    We set it to the ratio of negatives to positives.
    """
    print("\nTraining XGBoost classifier...")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos
    print(f"  Class ratio — non-All-Star: {neg}, All-Star: {pos}")
    print(f"  scale_pos_weight: {scale:.1f}x")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        scale_pos_weight=scale,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        verbosity=0,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train, verbose=False)
    print("  Done")
    return model


# ── Step 4: Evaluate ──────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    """
    For a classification model predicting rare events (All-Stars),
    accuracy is misleading. We use:

    Precision — of players we predicted as All-Star, what % actually were?
    Recall    — of actual All-Stars, what % did we correctly identify?
    ROC-AUC   — overall ability to rank All-Stars above non-All-Stars
                 0.5 = random, 1.0 = perfect
    """
    print("\nModel evaluation:")
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.3).astype(int)  # lower threshold for rare class

    precision = precision_score(y_test, preds, zero_division=0)
    recall    = recall_score(y_test, preds, zero_division=0)
    f1        = f1_score(y_test, preds, zero_division=0)
    auc       = roc_auc_score(y_test, probs)

    print(f"  Precision: {precision:.3f}  "
          f"(of players we called All-Star, {precision*100:.0f}% actually were)")
    print(f"  Recall:    {recall:.3f}  "
          f"(we correctly identified {recall*100:.0f}% of actual All-Stars)")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  ROC-AUC:   {auc:.3f}  "
          f"(ability to rank All-Stars above non-All-Stars)")

    if auc > 0.85:
        print("  Excellent discrimination — model clearly separates "
              "All-Stars from non-All-Stars")
    elif auc > 0.75:
        print("  Good discrimination — model reliably identifies "
              "All-Star profiles")
    else:
        print("  Moderate discrimination — consider adding more features")

    return probs


# ── Step 5: SHAP values ───────────────────────────────────────────────────────

def compute_shap_values(model, X):
    """
    TreeExplainer is the SHAP method for tree-based models like XGBoost.
    It is exact (not approximate) and fast.

    shap_values shape: (n_players, n_features)
    Each value represents how much that feature pushed the prediction
    above or below the base value (average All-Star probability).

    Positive SHAP = pushed toward All-Star
    Negative SHAP = pushed away from All-Star
    """
    print("\nComputing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    base_value  = explainer.expected_value
    print(f"  Base value (avg All-Star prob): {base_value:.4f}")
    print(f"  SHAP matrix shape: {shap_values.shape}")
    return explainer, shap_values, base_value


# ── Step 6: Development reports ───────────────────────────────────────────────

def generate_development_reports(df, X, shap_values, probs, feature_cols):
    """
    For each player, creates a development report that identifies:
    1. Their predicted All-Star probability
    2. Which features are helping them (positive SHAP)
    3. Which features are hurting them (negative SHAP)
    4. The #1 skill gap — the single biggest thing to work on

    This is the forward-looking SHAP application that is novel
    in the sports analytics literature.
    """
    print("\nGenerating player development reports...")

    report_rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        if i >= len(shap_values):
            break

        player_shap = shap_values[i]
        report = {
            "PLAYER_NAME":    row["PLAYER_NAME"],
            "SEASON":         row["SEASON"],
            "LABEL_ALLSTAR":  int(row[TARGET]) if pd.notna(row[TARGET]) else 0,
            "ALLSTAR_PROB":   round(float(probs[i]), 4),
        }

        # Add SHAP values for each feature
        for j, feat in enumerate(feature_cols):
            report[f"SHAP_{feat}"] = round(float(player_shap[j]), 4)

        # Find biggest gap (most negative SHAP = biggest drag)
        shap_dict = {feat: player_shap[j]
                     for j, feat in enumerate(feature_cols)}
        sorted_by_shap = sorted(shap_dict.items(), key=lambda x: x[1])

        biggest_gap    = sorted_by_shap[0]
        second_gap     = sorted_by_shap[1] if len(sorted_by_shap) > 1 else None
        biggest_strength = sorted(shap_dict.items(),
                                  key=lambda x: x[1], reverse=True)[0]

        report["BIGGEST_GAP"]      = biggest_gap[0]
        report["BIGGEST_GAP_SHAP"] = round(float(biggest_gap[1]), 4)
        report["SECOND_GAP"]       = second_gap[0] if second_gap else ""
        report["BIGGEST_STRENGTH"] = biggest_strength[0]

        report_rows.append(report)

    reports_df = pd.DataFrame(report_rows)
    return reports_df


# ── Step 7: Print example reports ─────────────────────────────────────────────

def print_player_report(reports_df, player_name, feature_cols):
    """
    Prints a human-readable development report for one player.
    This is what you would show to a coach or player agent.
    """
    rows = reports_df[
        reports_df["PLAYER_NAME"].str.contains(
            player_name, case=False, na=False
        )
    ].sort_values("SEASON")

    if len(rows) == 0:
        print(f"  Player '{player_name}' not found in reports.")
        return

    row = rows.iloc[-1]
    prob  = row["ALLSTAR_PROB"] * 100
    actual = "All-Star" if row["LABEL_ALLSTAR"] == 1 else "Not All-Star"

    print(f"\n{'='*58}")
    print(f"  DEVELOPMENT REPORT: {row['PLAYER_NAME']} ({row['SEASON']})")
    print(f"{'='*58}")
    print(f"  Actual:    {actual}")
    print(f"  Predicted: {prob:.1f}% All-Star probability")
    print()

    # Feature contributions sorted by SHAP
    shap_pairs = []
    for feat in feature_cols:
        shap_val = row[f"SHAP_{feat}"]
        label    = FEATURE_LABELS.get(feat, feat)
        shap_pairs.append((label, shap_val))

    shap_pairs.sort(key=lambda x: x[1], reverse=True)

    print("  Feature contributions (positive = helps, negative = hurts):")
    for label, val in shap_pairs:
        direction = "+" if val >= 0 else ""
        bar_len   = int(abs(val) * 60)
        bar       = "█" * min(bar_len, 20)
        side      = f"  {direction}{val:.3f}  {bar}"
        if val < 0:
            side = f"  {val:.3f}  {bar}"
        print(f"    {label:<35} {direction}{val:.3f}")

    print()
    gap  = row["BIGGEST_GAP"]
    gap2 = row["SECOND_GAP"]
    strength = row["BIGGEST_STRENGTH"]
    gap_label  = FEATURE_LABELS.get(gap, gap)
    gap2_label = FEATURE_LABELS.get(gap2, gap2)
    str_label  = FEATURE_LABELS.get(strength, strength)

    print(f"  Biggest strength:  {str_label}")
    print(f"  #1 gap to close:   {gap_label}")
    if gap2:
        print(f"  #2 gap to close:   {gap2_label}")
    print()

    if row["LABEL_ALLSTAR"] == 1:
        print("  Coaching note: Confirmed All-Star. SHAP profile shows")
        print(f"  {FEATURE_LABELS.get(gap_label, gap_label)} as the only area")
        print("  to monitor to maintain this classification level.")
    elif prob < 20:
        print("  Coaching note: This player is significantly below All-Star")
        print("  threshold. Focus on the top gap first before addressing others.")
    elif prob < 50:
        print("  Coaching note: This player is on the fringe. Improving the")
        print("  top gap could realistically push them into All-Star range.")
    else:
        print("  Coaching note: Strong All-Star candidate. Small improvements")
        print("  in the top gap may be the deciding factor.")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== NBA SHAP Project — Phase 4: SHAP Development Model ===\n")

    # Load
    df, X, y = load_data(INPUT_FILE)
    feature_cols = [f for f in FEATURES if f in X.columns]

    # Split
    X_train, X_test, y_train, y_test = temporal_split(df, X, y)

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    test_probs = evaluate(model, X_test, y_test)

    # SHAP on full dataset (we want reports for all players)
    all_probs = model.predict_proba(X)[:, 1]
    explainer, shap_values, base_value = compute_shap_values(model, X)

    # Generate reports
    reports_df = generate_development_reports(
        df, X, shap_values, all_probs, feature_cols
    )
    reports_df.to_csv(REPORTS_FILE, index=False)
    print(f"  Saved {len(reports_df):,} player reports → {REPORTS_FILE}")

    # Print example reports for three different player types
    print("\n--- Example development reports ---")
    for player in ["Jayson Tatum", "Cade Cunningham", "Gary Harris"]:
        print_player_report(reports_df, player, feature_cols)

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "explainer": explainer,
                     "features": feature_cols}, f)
    print(f"\nModel saved → {MODEL_FILE}")
    print("\n=== Done. Phase 4 complete. Next: Phase 5 — Salary model ===")


if __name__ == "__main__":
    main()