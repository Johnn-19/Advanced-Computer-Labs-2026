Formula 1 Podium Prediction Companion
Overview
This repository contains a complete data-engineering and machine-learning pipeline designed to predict whether a Formula 1 driver will finish on the podium (1st, 2nd, or 3rd place). The model strictly utilizes pre-race information, establishing a firm prediction cutoff after qualifying sessions are complete but before the race start.   
PDF
+ 3

The project emphasizes rigorous leakage control, handling of historical data drift (1950–2024), and model explainability to unpack the factors that give drivers a realistic shot at the podium.   
PDF
+ 1

Dataset
This project uses the Formula 1 World Championship (1950-2024) - Kaggle / Ergast dataset. It is a highly relational dataset comprising 14 CSV tables, including circuits, constructors, drivers, results, qualifying, and standings.   
PDF
+ 2

Core Objectives & Pipeline
1. Data Cleaning & Integrity Auditing
Sentinel Handling: Cleans and formats placeholder "missing" values (e.g., \N strings) and applies semantic typing to dates and lap-time durations.   
PDF

Grain Enforcement: Ensures the modeling table maintains a strict unit of analysis of one row per (raceId, driverId). Resolves historical anomalies, such as the 1950s "shared drives" where teammates swapped cars mid-race, using deterministic disambiguation rules.   
PDF
+ 3

Historical Drift Handling: Accounts for missing qualifying data in early F1 eras and varying field sizes.   
PDF
+ 1

2. Relational Data Engineering
Answers complex, multi-hop relational domain questions backed by exploratory data analysis (EDA):

Front-Row Conversion: Analyzes which circuits most frequently convert a front-row start into a podium, comparing the pre-2014 eras against the hybrid era.   
PDF

Home Race Advantage: Evaluates whether drivers podium more often at their home circuits after controlling for grid starting position.   
PDF

Mechanical Reliability: Compares constructor mechanical-retirement rates between the 2014–2021 turbo-hybrid era and the 2022–2024 ground-effect era.   
PDF
+ 1

3. Leakage-Safe Feature Engineering
Constructs features strictly from data available before the lights go out.   
PDF

Forbidden Features: Prevents data leakage by completely excluding current-race results, race status, pit stops, lap times, and post-race championship standings.   
PDF

Allowed Features: Utilizes the starting grid (accounting for penalties), qualifying metrics, shifted championship standings entering the race, and historical rolling constructor/driver form.   
PDF
+ 1

4. Predictive Modeling
Frames the problem as a binary classification task evaluating models on an unseen test set following a chronological split (e.g., Training ≤ 2019, Validation 2020–2021, Test ≥ 2022):   
PDF
+ 1

Baselines & Advanced Models: Compares at least two statistical machine learning models (e.g., Logistic Regression, Gradient Boosting) against a shallow Feed-Forward Neural Network (FFNN).   
PDF

Metrics: Evaluates performance using ROC-AUC, PR-AUC (critical due to the ~15% podium class imbalance), and F1 Score.   
PDF
+ 1

5. Explainable AI (XAI) & Ablation
Ablation Studies: Measures the performance impact of removing specific feature groups (e.g., qualifying data, standings).   
PDF

Global & Local Explanations: Utilizes Permutation Importance, SHAP (Tree/Kernel), and LIME to interpret model behavior overall and on individual predictions.   
PDF

Disclaimer: Feature importance and SHAP values indicate model reliance and association; they do not represent causal effects.   
PDF

Repository Structure
/data/: Directory for the 14 raw Ergast CSV files (not tracked in version control).

/notebooks/: Contains the run-all Jupyter Notebook demonstrating the complete end-to-end pipeline (EDA, cleaning, DE questions, modeling, XAI, and inference).   
PDF

/reports/: PDF analytical reports detailing DE query answers, feature validity, limitations, and model comparisons.   
PDF
+ 1

/src/: Helper scripts for data preprocessing and the final inference function capable of handling complete or partially-missing race records.   
PDF

Setup & Installation
Clone the repository.

Download the Formula 1 World Championship dataset from Kaggle and extract the CSV files into the /data/ directory.

Install the required dependencies:

Bash
pip install -r requirements.txt
Run the main Jupyter Notebook to reproduce the EDA, modeling, and XAI outputs.
