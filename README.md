# NBA SHAP Player Valuation System

A three-component machine learning framework for NBA player 
valuation using XGBoost and SHAP (SHapley Additive exPlanations).

## What This Does

### Component 1 — Trade Compatibility Model
Ranks all 30 NBA teams by predicted fit for any player using 
system-aware features including roster gap analysis and 
play-style scoring. Built with XGBoost regression.

### Component 2 — SHAP Development Model  
Predicts All-Star probability for every NBA player and uses 
TreeSHAP to identify exactly which statistical gaps are 
holding each player below All-Star threshold. ROC-AUC: 0.980.

### Component 3 — System-Adjusted Salary Fairness Model
Strips team context from player statistics using Ridge 
regression residuals to estimate true individual contribution, 
then predicts fair market salary and uses SHAP to explain 
underpayment or overpayment gaps. No published research 
currently does this.

## Key Results
| Model | Metric | Value |
|---|---|---|
| Trade compatibility | Test R² | 0.212 |
| Trade compatibility | Direction accuracy | 88% |
| SHAP development | ROC-AUC | 0.980 |
| SHAP development | Recall | 0.909 |
| Salary fairness | Test R² | 0.551 |
| Salary fairness | Avg error | $5.0M |
| System adjustment | PLUS_MINUS variance explained by team | 44.4% |

## Novel Contribution
System-adjusted salary modeling with per-player SHAP 
explanations does not exist in published sports analytics 
literature. This framework is the first to isolate individual 
statistical contribution from team system context before 
computing fair market salary.

## Commercial Application
Built for the player agent use case — agents currently lack 
dedicated quantitative tools while every NBA team has a full 
analytics department. Analogous to Analytics FC's work with 
Kevin De Bruyne in soccer (£104M contract).

## Example Output

**Salary Fairness Report: Nikola Jokic (2022-23)**
- Actual salary: $32,400,000
- Predicted fair salary: $37,403,276
- Gap: UNDERPAID by $5.0M
- System adjustment: +3.02 std devs above team context

**SHAP Development Report: Cade Cunningham (2021-22)**
- All-Star probability: 24.9%
- Biggest strength: Ball security
- #1 gap to close: On-court impact (suppressed by team context)
- Agent argument: individual skills are All-Star trajectory, 
  PLUS_MINUS suppressed by Detroit's losing system

## Data Sources
- NBA statistics: nba_api (free, open source)
- Salary data: Basketball-Reference / Kaggle
- Seasons covered: 2018-19 through 2022-23
- Players: 2,146 player-season observations

## Tech Stack
- Python, pandas, scikit-learn
- XGBoost (all three models)
- SHAP (TreeExplainer)
- Ridge regression (system adjustment)

## Repository Structure

### Python Scripts (run in this order)

| File | Phase | Description |
|---|---|---|
| `feature_pipeline.py` | 1 | Pulls 5 seasons of NBA player and team stats via nba_api. Engineers player feature vectors (PLUS_MINUS, PTS, AST, REB, STL, BLK, TOV, MIN) and team system vectors (pace, defensive score, win percentage). Applies z-score normalization by season to remove era bias. Clusters teams into 6 play-style archetypes using K-Means. Outputs nba_feature_matrix.csv |
| `label_pipeline.py` | 2 | Loads the feature matrix and fills three label columns. LABEL_ALLSTAR: matches players against hardcoded All-Star rosters (2018-2023). LABEL_WS48_CHG: identifies players who changed teams between seasons and calculates performance change. LABEL_SALARY: seeds salary data for key players. Outputs nba_feature_matrix_labeled.csv |
| `add_salaries.py` | 2 | Expands salary coverage from 67 rows to 1,930 rows using the Kaggle NBA Player Stats and Salaries 2010-2025 dataset. Applies Unicode normalization to fix European player name mismatches (e.g. Jokić → Jokic). Requires NBA_Player_Stats_and_Salaries_2010-2025.csv in the project folder |
| `trade_model_v2.py` | 3 | Trains an XGBoost regression model on 561 real trade observations to predict post-trade performance change. Engineers three role-aware features: USAGE_PROXY (ball dominance), EFFICIENCY (scoring per turnover), DEFENSIVE_ROLE (steals + blocks). Adds roster gap penalty to discourage redundant signings. Replaces broken K-Means archetype with interpretable TEAM_PACE_SCORE and TEAM_DEF_SCORE. Ranks all 30 teams for any player. Outputs trade_model_v2.pkl and trade_rankings CSV files |
| `shap_model.py` | 4 | Trains an XGBoost classifier to predict All-Star selection using scale_pos_weight to handle 5% class imbalance. Achieves ROC-AUC 0.980 and Recall 0.909 on held-out 2022-23 season. Applies TreeSHAP to decompose every prediction into per-feature contributions. Generates a development report for all 2,146 players identifying their biggest skill gap and strongest attribute. This forward-looking SHAP application is the novel research contribution. Outputs shap_model.pkl and shap_development_reports.csv |
| `salary_model.py` | 5 | The most commercially novel component. Fits Ridge regression on team features to estimate how much of each player's PLUS_MINUS is explained by team context (44.4%). Computes system-adjusted residual (SYS_ADJ) as a new feature representing true individual contribution. Trains XGBoost salary regression achieving Test R² 0.551. Applies SHAP to explain which stats drive underpayment or overpayment gaps. Generates agent-ready salary fairness reports with dollar gap and one-paragraph negotiation talking point. Outputs salary_model.pkl and salary_fairness_reports.csv |
| `validation.py` | 6 | Four-part validation suite. Part A backtests the trade model against 8 real 2022-23 trades — achieves 88% direction accuracy. Part B validates SHAP model prospectively — high-probability players become All-Stars at 6.5% vs 0.2% for low-probability players (32x lift). Part C analyzes system adjustment distribution across 1,930 salary rows. Part D prints the full MIT Sloan research paper outline with actual results filled in |

### Data Files (not tracked in Git — download separately)

| File | Source | Description |
|---|---|---|
| `NBA_Player_Stats_and_Salaries_2010-2025.csv` | Kaggle | 7,298 rows of player stats and salary data covering 2010-2025. Required by add_salaries.py. Download from Kaggle and place in project folder |
| `nba_feature_matrix.csv` | Generated | Output of feature_pipeline.py. 2,146 player-season rows with normalized features and empty label columns |
| `nba_feature_matrix_labeled.csv` | Generated | Output of label_pipeline.py and add_salaries.py. Same 2,146 rows with all three labels filled |
| `shap_development_reports.csv` | Generated | Output of shap_model.py. One row per player with All-Star probability, per-feature SHAP values, biggest gap, and biggest strength |
| `salary_fairness_reports.csv` | Generated | Output of salary_model.py. One row per player with actual salary, system adjustment score, per-feature SHAP values, and agent talking point |

### Model Files (not tracked in Git — generated locally)

| File | Description |
|---|---|
| `trade_model_v2.pkl` | Trained XGBoost trade compatibility model |
| `shap_model.pkl` | Trained XGBoost All-Star classifier with TreeExplainer |
| `salary_model.pkl` | Trained XGBoost salary regression with Ridge system adjuster |

## Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/NBA-SHAP-Player-Valuation.git
cd NBA-SHAP-Player-Valuation
```

**2. Install dependencies**
```bash
pip install nba_api pandas scikit-learn xgboost shap "numpy<2"
```

**3. Download the Kaggle salary dataset**
Search "NBA Player Stats and Salaries 2010-2025" on Kaggle.
Download and place `NBA_Player_Stats_and_Salaries_2010-2025.csv` 
in the project folder.

**4. Run the pipeline in order**
```bash
python feature_pipeline.py      # ~3 minutes (API calls)
python label_pipeline.py        # ~2 minutes (API calls)
python add_salaries.py          # instant
python trade_model_v2.py        # instant
python shap_model.py            # instant
python salary_model.py          # instant
python validation.py            # instant
```
```
