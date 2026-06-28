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

## Pipeline
