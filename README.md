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

### Final model: Gradient Boosting

The final model is a **Gradient Boosting** classifier (`HistGradientBoostingClassifier`, learning rate 0.03, 15 leaves per tree, at least 20 rows per leaf, 300 trees). It uses a podium threshold of **0.233**: a driver is predicted to finish on the podium when the predicted probability is at least 0.233. This threshold was chosen to maximise F1 on the validation set.

**Why this model.** The selection rule was fixed in advance: the model with the highest validation PR-AUC wins, and models within 0.005 of it count as a tie, in which case the simpler one wins. Gradient Boosting scored highest (0.749). The neural network (0.746) tied with it but is more complex. The test set was used only once, after this choice was made.

**Performance of the final model**

| Split | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Train (1950–2019) | 0.933 | 0.673 | 0.626 | 0.505 | 0.823 |
| Validation (2020–2021) | 0.931 | 0.749 | 0.703 | 0.641 | 0.778 |
| **Test (2022–2024, unseen)** | **0.938** | **0.709** | **0.673** | **0.567** | **0.828** |

On the unseen test seasons the model finds 83% of the actual podium finishers (recall 0.828). When it predicts a podium it is right 57% of the time (precision 0.567). Most of its wrong podium predictions are fast cars that finished 4th–6th. The 95% bootstrap interval for test PR-AUC is 0.66–0.77.

### Comparison with the other models (test set, 2022–2024)

The other models are the required comparison: Logistic Regression is the baseline, Random Forest is a second statistical model, and the shallow neural network is the required neural model. The grid-only rule is a reference with no machine learning (predict a podium when starting P1–P3). Every model uses the same rows, the same 24 features and the same pre-processing.

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Grid-only rule (no ML) | 0.913 | 0.647 | 0.649 | 0.650 | 0.647 |
| Logistic Regression (baseline) | 0.934 | 0.716 | 0.654 | 0.625 | 0.686 |
| Random Forest | 0.940 | 0.718 | 0.680 | 0.622 | 0.750 |
| **Gradient Boosting (final)** | **0.938** | **0.709** | **0.673** | **0.567** | **0.828** |
| Shallow NN (128→64) | 0.932 | 0.705 | 0.650 | 0.599 | 0.711 |

All models clearly beat the grid-only rule, so the form and standings features add real value. The differences between the four models are smaller than the bootstrap uncertainty. Random Forest and Logistic Regression are slightly higher than Gradient Boosting on test PR-AUC, but the final model was not changed after seeing the test results, because that would turn the test set into a second validation set. Train and validation results for every model are in Section 8.4 of the notebook.

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
