# Formula 1 Podium Prediction — Milestone 1 (Team 106)

Advanced Computer Lab, GUC · Milestone 1

**Team:** John Magdy Fouad Soliman · Abdelrahman Mohamed Abdelrahman Omar Elaswar · Omar Ehab Mohamed Salama Nasr · Mohamed Hany Elghamry Youssef

**Kaggle notebook:** https://www.kaggle.com/code/johnsolimann/project-acl-2026

## Overview

For every driver in every Formula 1 race, the project predicts whether the driver finishes on the podium (`positionOrder ≤ 3`). It uses only information available **after qualifying and before the race starts**. The whole pipeline lives in one run-all Jupyter notebook:

- data audit and cleaning
- the three data-engineering questions
- leakage-safe feature engineering
- modeling and ablation
- explainability (XAI)
- an inference function

## Repository contents

| File | Description |
|---|---|
| `project-acl-2026_MS1_finalSubmission.ipynb` | The run-all notebook with all outputs (293 cells, 55 figures) |
| `ACL_Milestone1_Report_Team106.pdf` | Analytical report: answers to the data-engineering questions with figures, the selected features and why each is valid at the pre-race cutoff, and known limitations |
| `README.md` | This file |

## Dataset

[Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020), Kaggle / Ergast. It has 14 relational CSV tables (`results`, `races`, `qualifying`, `drivers`, `constructors`, `circuits`, `status`, standings, …). The dataset is **not** included in this repository.

## Pipeline

1. **EDA and cleaning.**
   - `\N` sentinels are parsed as missing, and dates and lap times are converted to real types.
   - Primary and foreign keys are verified; the raw CSVs are never modified.
   - The Indianapolis 500 and non-starters are excluded.
   - 1950s shared-drive duplicates are resolved (keep the car the driver started in, then the best `positionOrder`).
   - Result: a modeling table of **24,551 rows, unique on `(raceId, driverId)`**, with a 13.6% podium rate.
   - The key EDA plots are repeated before and after cleaning.
2. **Data-engineering questions** (multi-hop joins, each answered with plots plus notes on grain, missing data and the pre-race cutoff):
   - **Q1:** front-row conversion by circuit, 2003–2013 vs 2014–2024, comparing the grid and qualifying definitions of "front row".
   - **Q2:** home advantage after controlling for grid position (Cochran–Mantel–Haenszel and logistic regression). **No home effect.**
   - **Q3:** constructor mechanical-retirement rates, 2014–2021 vs 2022–2024, using an explicit mapping of all 139 status values.
3. **Leakage-safe features.** 24 features in 6 groups:
   - Grid
   - Qualifying
   - Driver form (`shift(1)` before a 20-race rolling window)
   - Constructor form (aggregated per constructor and race before shifting, so a teammate's result cannot leak)
   - Championship standings after the previous round
   - Context

   Six assertion-based leakage checks verify them. The rolling window (N = 20) and the training window (full history) were chosen by validation experiments.
4. **Split.** Chronological by season: train 1950–2019, validation 2020–2021, test 2022–2024.
5. **Pre-processing.** Median imputation then standard scaling, fitted on the training rows only. Three pre-processing-order experiments are included.
6. **Models.**
   - Logistic Regression (baseline), Random Forest, HistGradientBoosting, and a shallow neural network (scikit-learn `MLPClassifier`, 128→64, early stopping).
   - 52 configurations in total, all tuned on validation.
   - Primary metric is **PR-AUC**, secondary **ROC-AUC**. F1, precision and recall are also reported, at thresholds chosen on validation.
7. **Ablation and XAI.**
   - Feature-group ablation.
   - Permutation importance and SHAP (model-agnostic permutation explainer) for global explanations.
   - A SHAP waterfall plot and LIME for single predictions.
   - Feature importance is reported as what the model relies on, **not** as a causal effect.
8. **Inference.**
   - `predict_podium(record)` builds the features from a raw pre-race record and applies the frozen pipeline and threshold. It handles complete and partially missing records.
   - It rejects post-race fields.
   - The trained model is saved to `model/podium_model.joblib`.

## Results

| Model | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1 |
|---|---|---|---|---|
| Grid-only rule (no ML) | 0.648 | 0.647 | 0.913 | 0.649 |
| Logistic Regression | 0.742 | 0.716 | 0.934 | 0.654 |
| Random Forest | 0.743 | 0.718 | 0.940 | 0.680 |
| **Gradient Boosting (final)** | **0.749** | **0.709** | **0.938** | **0.673** |
| Shallow NN | 0.746 | 0.705 | 0.932 | 0.650 |

The final model was chosen on validation only. All models beat the grid-only rule. Their differences are within the bootstrap uncertainty (test 95% CI for Gradient Boosting: 0.66–0.77).

## How to run

**On Kaggle (recommended).**

1. Open the Kaggle notebook.
2. Make sure the dataset is attached (*Add Input → Formula 1 World Championship (1950 – 2024)*).
3. Turn *Settings → Internet* on, then choose *Run All*.

The first cell pins `scikit-learn==1.9.1` and installs `shap` and `lime` if they are missing.

**Locally.**

1. Download the dataset from Kaggle and extract the CSVs into any folder under the notebook's directory (e.g. `data/`). The notebook finds `results.csv` automatically.
2. Install the dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn==1.9.1 shap lime joblib
   ```
3. Run all cells.

The notebook writes the cleaned tables to `clean/`, all figures to `figures/`, the result tables to `tables/` and the trained model to `model/`. A full run takes about 7–9 minutes on a CPU.
