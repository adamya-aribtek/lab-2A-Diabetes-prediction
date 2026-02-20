# LAB 2A — Diabetes Prediction (Tabular Deep Learning)

This project implements a supervised learning pipeline for diabetes prediction
using structured clinical data. The objective is to compare a deep learning
baseline (MLP) with boosted tree models, evaluate probability calibration, and
analyze model performance across demographic and clinical subgroups.

## Problem Statement
Early identification of diabetes risk is critical for preventive healthcare.
This project frames diabetes prediction as a binary classification task using
demographic and clinical features.

## Dataset
- Source: Kaggle — Diabetes Prediction Dataset
- Samples: 100,000
- Features: Demographic, clinical, and lifestyle indicators
- Target: `diabetes` (binary)

## Methods
- Exploratory data analysis (EDA)
- Feature preprocessing (scaling + encoding)
- Baseline MLP (PyTorch)
- Gradient Boosting classifier (sklearn)
- Calibration analysis (Platt scaling)
- Segment analysis (age, gender, smoking history)

## Key Results
- MLP ROC-AUC: ~0.976
- Gradient Boosting ROC-AUC: ~0.978
- Best model selected based on discrimination and calibration quality
- Stable performance across major subgroups


## Reproducibility
- Random seed fixed where applicable
- Preprocessing fit on training data only
- Model artifacts saved for reuse

## How to Load the Model
```python
import joblib

model = joblib.load("models/gradient_boosting_diabetes_v1.joblib")
preprocessor = joblib.load("models/gradient_boosting_diabetes_v1_preprocessor.joblib")