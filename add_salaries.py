"""
NBA SHAP Project — Salary Label from Kaggle Dataset
====================================================
Uses the NBA_Player_Stats_and_Salaries_2010-2025.csv file.

Place that CSV in your NBA_SHAP folder, then run:
    python add_salaries.py
"""

import pandas as pd
import numpy as np

LABELED_FILE  = "nba_feature_matrix_labeled.csv"
SALARY_FILE   = "NBA_Player_Stats_and_Salaries_2010-2025.csv"

# Map: end year (Kaggle format) → season string (our format)
YEAR_TO_SEASON = {
    2019: "2018-19",
    2020: "2019-20",
    2021: "2020-21",
    2022: "2021-22",
    2023: "2022-23",
}

def main():
    print("=== Salary Label Expansion (Kaggle dataset) ===\n")

    # ── Load labeled matrix ─────────────────────────────────────────────────
    df = pd.read_csv(LABELED_FILE)
    print(f"Labeled matrix loaded:  {len(df):,} rows")
    print(f"Salary filled before:   {df['LABEL_SALARY'].notna().sum():,} rows")

    # ── Load Kaggle salary data ──────────────────────────────────────────────
    sal = pd.read_csv(SALARY_FILE)
    print(f"\nKaggle dataset loaded:  {len(sal):,} rows")

    # Keep only our 5 seasons
    sal = sal[sal["Year"].isin(YEAR_TO_SEASON.keys())].copy()
    print(f"Rows for our seasons:   {len(sal):,}")

    # Convert year to season string
    sal["SEASON"] = sal["Year"].map(YEAR_TO_SEASON)

    # Clean up player name column to match our matrix
    # Normalize special characters so Serbian/European names match
    import unicodedata

    def normalize_name(name):
        return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").strip()

    sal["PLAYER_NAME"] = sal["Player"].apply(normalize_name)
    df["PLAYER_NAME"]  = df["PLAYER_NAME"].apply(normalize_name)

    # Keep just the columns we need
    sal = sal[["PLAYER_NAME", "SEASON", "Salary"]].rename(columns={"Salary": "SALARY_NEW"})
    sal = sal.dropna(subset=["SALARY_NEW"])
    sal = sal[sal["SALARY_NEW"] > 0]

    print(f"Clean salary rows:      {len(sal):,}")

    # ── Merge into labeled matrix ────────────────────────────────────────────
    df = df.merge(sal, on=["PLAYER_NAME", "SEASON"], how="left")

    # Fill salary label — Kaggle data takes priority, hardcoded as fallback
    df["LABEL_SALARY"] = df["SALARY_NEW"].combine_first(df["LABEL_SALARY"])
    df = df.drop(columns=["SALARY_NEW"])

    filled = df["LABEL_SALARY"].notna().sum()
    print(f"\nSalary filled after:    {filled:,} / {len(df):,} rows")

    # ── Spot-check a few known players ──────────────────────────────────────
    print("\nSpot-check (should match known salaries):")
    checks = [
        ("Stephen Curry",  "2022-23"),
        ("LeBron James",   "2021-22"),
        ("Nikola Jokic",   "2020-21"),
        ("Ja Morant",      "2022-23"),
    ]
    for name, season in checks:
        row = df[(df["PLAYER_NAME"] == name) & (df["SEASON"] == season)]
        if len(row) > 0:
            sal_val = row["LABEL_SALARY"].values[0]
            print(f"  {name} ({season}): ${sal_val:,.0f}" if pd.notna(sal_val) else f"  {name} ({season}): not found")
        else:
            print(f"  {name} ({season}): not in matrix")

    # ── Save ────────────────────────────────────────────────────────────────
    df.to_csv(LABELED_FILE, index=False)
    print(f"\nSaved → {LABELED_FILE}")
    print("\n=== Done. All three labels now populated. Ready for Phase 3. ===")

if __name__ == "__main__":
    main()