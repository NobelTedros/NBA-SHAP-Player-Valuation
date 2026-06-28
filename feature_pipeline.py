"""
NBA Player Valuation — Feature Engineering Pipeline
====================================================
Pulls raw player and team stats via nba_api, engineers the
player vector and team system vector, normalizes, and outputs
a clean feature matrix ready for model training.

Covers all three models:
  - Trade compatibility model  (label: WS/48 change post-trade)
  - SHAP development model     (label: All-Star selection 0/1)
  - Salary fairness model      (label: salary vs predicted value)

Run:
    pip install nba_api pandas scikit-learn
    python feature_pipeline.py
"""

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashteamstats,
)

# ── Config ──────────────────────────────────────────────────────────────
SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
SLEEP    = 1.2   # seconds between API calls (avoid rate-limiting)
N_TEAM_CLUSTERS = 6  # number of play-style archetypes for teams

# ── Stage 1: Raw data ingestion ─────────────────────────────────────────

def pull_player_stats(seasons: list[str]) -> pd.DataFrame:
    """
    Pull per-game player stats for each season via nba_api.
    Returns a combined DataFrame with a 'season' column.
    """
    frames = []
    for season in seasons:
        print(f"  Pulling player stats: {season}...")
        endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
        )
        df = endpoint.get_data_frames()[0]
        df["SEASON"] = season
        frames.append(df)
        time.sleep(SLEEP)
    return pd.concat(frames, ignore_index=True)


def pull_team_stats(seasons: list[str]) -> pd.DataFrame:
    """
    Pull team-level stats for pace, ratings, and shot profile.
    """
    frames = []
    for season in seasons:
        print(f"  Pulling team stats:   {season}...")
        endpoint = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed="PerGame",
        )
        df = endpoint.get_data_frames()[0]
        df["SEASON"] = season
        frames.append(df)
        time.sleep(SLEEP)
    return pd.concat(frames, ignore_index=True)


# ── Stage 2: Feature selection ──────────────────────────────────────────

PLAYER_FEATURES = [
    "USG_PCT",       # Usage rate          — how often player is involved
    "AST_PCT",       # Assist percentage   — playmaking
    "REB_PCT",       # Rebound percentage  — rebounding ability (if available)
    "TS_PCT",        # True shooting %     — scoring efficiency
    "PLUS_MINUS",    # Plus/minus          — on-court impact
    "PTS",           # Points per game     — scoring volume
    "AST",           # Assists per game
    "REB",           # Rebounds per game
    "STL",           # Steals per game     — defensive activity
    "BLK",           # Blocks per game     — rim protection
    "TOV",           # Turnovers per game  — ball security
    "MIN",           # Minutes per game    — playing time / trust
    "GP",            # Games played        — durability
]

TEAM_FEATURES = [
    "W_PCT",         # Win percentage      — team quality context
    "PTS",           # Points per game     — offensive tempo indicator
    "OPP_PTS",       # Opponent points     — defensive quality
    "AST",           # Team assists        — ball movement style
    "REB",           # Team rebounds
    "STL",           # Team steals         — defensive aggression
    "BLK",           # Team blocks         — rim protection emphasis
    "PLUS_MINUS",    # Net rating proxy
]


def build_player_vector(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Select and clean the player feature columns.
    Drops players with fewer than 20 games (small sample).
    """
    cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "SEASON"] + PLAYER_FEATURES
    available = [c for c in cols if c in raw.columns]
    df = raw[available].copy()
    df = df[df["GP"] >= 20]
    df = df.dropna(subset=[c for c in PLAYER_FEATURES if c in df.columns])
    return df.reset_index(drop=True)


def build_team_vector(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Select and clean team system feature columns.
    """
    cols = ["TEAM_ID", "TEAM_NAME", "SEASON"] + TEAM_FEATURES
    available = [c for c in cols if c in raw.columns]
    df = raw[available].copy()
    df = df.dropna(subset=[c for c in TEAM_FEATURES if c in df.columns])
    return df.reset_index(drop=True)


# ── Stage 3a: Normalize player stats across seasons ─────────────────────

def normalize_by_season(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Z-score normalize each stat within each season.
    This removes era and pace-of-play bias so that, for example,
    a 25-PPG season in 2019 is comparable to a 25-PPG season in 2023.
    """
    df = df.copy()
    present = [c for c in feature_cols if c in df.columns]
    # Convert to float so z-scores (decimals) can be stored back
    df[present] = df[present].astype(float)
    for season in df["SEASON"].unique():
        mask = df["SEASON"] == season
        scaler = StandardScaler()
        df.loc[mask, present] = scaler.fit_transform(df.loc[mask, present])
    return df


# ── Stage 3b: Cluster teams into play-style archetypes ──────────────────

def cluster_team_styles(
    team_df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = N_TEAM_CLUSTERS,
) -> pd.DataFrame:
    """
    K-means clustering groups teams into play-style archetypes.
    Examples: "pace-and-space", "defensive-physical", "ball-movement", etc.

    Adds a TEAM_ARCHETYPE column (0..n_clusters-1) to the team DataFrame.
    """
    df = team_df.copy()
    present = [c for c in feature_cols if c in df.columns]
    scaler  = StandardScaler()
    X       = scaler.fit_transform(df[present])
    km      = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["TEAM_ARCHETYPE"] = km.fit_predict(X)
    print(f"  Team archetypes: {df.groupby('TEAM_ARCHETYPE')['TEAM_NAME'].count().to_dict()}")
    return df


# ── Stage 4: Merge into one training row per player-team-season ─────────

def build_feature_matrix(
    player_df: pd.DataFrame,
    team_df:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Join normalized player stats with team system vectors.
    Each row = one player on one team in one season.
    This is the final feature matrix fed into all three models.
    """
    player_feat_cols = [c for c in PLAYER_FEATURES if c in player_df.columns]
    team_feat_cols   = [c for c in TEAM_FEATURES   if c in team_df.columns]

    # Rename team features to avoid collision with player features
    team_renamed = team_df.rename(
        columns={c: f"TEAM_{c}" for c in team_feat_cols}
    )

    merged = player_df.merge(
        team_renamed[["TEAM_ID", "SEASON", "TEAM_ARCHETYPE"]
                     + [f"TEAM_{c}" for c in team_feat_cols]],
        on=["TEAM_ID", "SEASON"],
        how="left",
    )
    return merged.reset_index(drop=True)


# ── Stage 5: Add placeholder labels ────────────────────────────────────

def add_placeholder_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder label columns.

    In the real pipeline you would:
      - Join salary data for SALARY_LABEL
      - Join All-Star rosters for ALLSTAR_LABEL
      - For TRADE_LABEL: join post-trade WS/48 for players who changed teams

    These are set to NaN here so the schema is clear.
    """
    df = df.copy()
    df["LABEL_ALLSTAR"]  = np.nan  # 1 = made All-Star, 0 = did not
    df["LABEL_WS48_CHG"] = np.nan  # WS/48 change post trade
    df["LABEL_SALARY"]   = np.nan  # actual salary (for fairness model)
    return df


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=== NBA Feature Engineering Pipeline ===\n")

    # Stage 1: Pull raw data
    print("Stage 1: Pulling data from nba_api...")
    raw_players = pull_player_stats(SEASONS)
    raw_teams   = pull_team_stats(SEASONS)
    print(f"  Raw player rows: {len(raw_players):,}")
    print(f"  Raw team rows:   {len(raw_teams):,}\n")

    # Stage 2: Select features
    print("Stage 2: Building feature vectors...")
    player_df = build_player_vector(raw_players)
    team_df   = build_team_vector(raw_teams)
    print(f"  Player rows after filtering: {len(player_df):,}")
    print(f"  Team rows:                   {len(team_df):,}\n")

    # Stage 3: Normalize + cluster
    print("Stage 3: Normalizing and clustering...")
    player_feat_cols = [c for c in PLAYER_FEATURES if c in player_df.columns]
    team_feat_cols   = [c for c in TEAM_FEATURES   if c in team_df.columns]

    player_df = normalize_by_season(player_df, player_feat_cols)
    team_df   = cluster_team_styles(team_df, team_feat_cols)
    print()

    # Stage 4: Merge into training matrix
    print("Stage 4: Building feature matrix...")
    feature_matrix = build_feature_matrix(player_df, team_df)
    print(f"  Feature matrix shape: {feature_matrix.shape}\n")

    # Stage 5: Add labels
    feature_matrix = add_placeholder_labels(feature_matrix)

    # Save
    out_path = "nba_feature_matrix.csv"
    feature_matrix.to_csv(out_path, index=False)
    print(f"Stage 5: Saved feature matrix → {out_path}")
    print(f"\nColumns ({len(feature_matrix.columns)}):")
    for col in feature_matrix.columns:
        print(f"  {col}")

    print("\n=== Done. Next step: join salary + All-Star labels, then train models. ===")


if __name__ == "__main__":
    main()
