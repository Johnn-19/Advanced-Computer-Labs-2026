# Formula 1 Podium Prediction Companion

## Overview
This repository contains a complete data-engineering and machine-learning pipeline designed to predict whether a Formula 1 driver will finish on the podium (1st, 2nd, or 3rd place)[cite: 1, 8]. The model strictly utilizes pre-race information, establishing a firm prediction cutoff after qualifying sessions are complete but before the race start[cite: 1, 8]. 

The project emphasizes rigorous leakage control, handling of historical data drift (1950–2024), and model explainability to unpack the factors that give drivers a realistic shot at the podium[cite: 8, 12].

## Dataset
This project uses the **Formula 1 World Championship (1950-2024) - Kaggle / Ergast dataset**[cite: 8, 16]. It is a highly relational dataset comprising 14 CSV tables, including `circuits`, `constructors`, `drivers`, `results`, `qualifying`, and `standings`[cite: 8].

## Core Objectives & Pipeline

### 1. Data Cleaning & Integrity Auditing
* **Sentinel Handling:** Cleans and formats placeholder "missing" values (e.g., `\N` strings) and applies semantic typing to dates and lap-time durations[cite: 9].
* **Grain Enforcement:** Ensures the modeling table maintains a strict unit of analysis of one row per `(raceId, driverId)`[cite: 8, 9]. Resolves historical anomalies, such as the 1950s "shared drives" where teammates swapped cars mid-race, using deterministic disambiguation rules[cite: 7, 9].
* **Historical Drift Handling:** Accounts for missing qualifying data in early F1 eras and varying field sizes[cite: 5, 7].

### 2. Relational Data Engineering
Answers complex, multi-hop relational domain questions backed by exploratory data analysis (EDA):
* **Front-Row Conversion:** Analyzes which circuits most frequently convert a front-row start into a podium, comparing the pre-2014 eras against the hybrid era[cite: 9].
* **Home Race Advantage:** Evaluates whether drivers podium more often at their home circuits after controlling for grid starting position[cite: 10].
* **Mechanical Reliability:** Compares constructor mechanical-retirement rates between the 2014–2021 turbo-hybrid era and the 2022–2024 ground-effect era[cite: 6, 10].

### 3. Leakage-Safe Feature Engineering
* Constructs features strictly from data available before the lights go out[cite: 8].
* **Forbidden Features:** Prevents data leakage by completely excluding current-race results, race status, pit stops, lap times, and post-race championship standings[cite: 8].
* **Allowed Features:** Utilizes the starting grid (accounting for penalties), qualifying metrics, shifted championship standings entering the race, and historical rolling constructor/driver form[cite: 2, 8].

### 4. Predictive Modeling
Frames the problem as a binary classification task evaluating models on an unseen test set following a chronological split (e.g., Training <= 2019, Validation 2020–2021, Test >= 2022)[cite: 4, 10]:
* **Baselines & Advanced Models:** Compares at least two statistical machine learning models (e.g., Logistic Regression, Gradient Boosting) against a shallow Feed-Forward Neural Network (FFNN)[cite: 10].
* **Metrics:** Evaluates performance using ROC-AUC, PR-AUC (critical due to the ~15% podium class imbalance), and F1 Score[cite: 10, 11].

### 5. Explainable AI (XAI) & Ablation
* **Ablation Studies:** Measures the performance impact of removing specific feature groups (e.g., qualifying data, standings)[cite: 11].
* **Global & Local Explanations:** Utilizes Permutation Importance, SHAP (Tree/Kernel), and LIME to interpret model behavior overall and on individual predictions[cite: 11]. 
* *Disclaimer:* Feature importance and SHAP values indicate model reliance and association; they do not represent causal effects[cite: 12].

## Repository Structure
* `/data/`: Directory for the 14 raw Ergast CSV files (not tracked in version control).
* `/notebooks/`: Contains the run-all Jupyter Notebook demonstrating the complete end-to-end pipeline (EDA, cleaning, DE questions, modeling, XAI, and inference)[cite: 12].
* `/reports/`: PDF analytical reports detailing DE query answers, feature validity, limitations, and model comparisons[cite: 12, 15].
* `/src/`: Helper scripts for data preprocessing and the final inference function capable of handling complete or partially-missing race records[cite: 12].

## Setup & Installation
1. Clone the repository.
2. Download the Formula 1 World Championship dataset from Kaggle and extract the CSV files into the `/data/` directory.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
