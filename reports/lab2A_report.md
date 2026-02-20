


```markdown
# LAB 2A Report — Diabetes Prediction (Tabular DL)

## 1. Objective
The objective of this lab is to build and evaluate machine learning and deep
learning models for diabetes prediction using tabular clinical data. The task
focuses on model comparison, probability calibration, and subgroup analysis to
assess real-world decision reliability.

## 2. Dataset
- Source: Kaggle Diabetes Prediction Dataset
- Samples: 100,000
- Target: Binary diabetes indicator
- Features include demographic attributes, lifestyle factors, and clinical
  measurements.
- No personally identifiable information (PII) is used.

## 3. Methodology

### Data Preparation
- Numerical features standardized
- Categorical features one-hot encoded
- Stratified train/validation/test split (70/15/15)
- Preprocessing fit exclusively on training data

### Models
- Baseline MLP (PyTorch)
- Gradient Boosting Classifier (sklearn)

### Calibration
- Post-hoc Platt scaling evaluated
- Reliability curves and Brier score used

### Reproducibility
- Fixed random seeds
- Deterministic preprocessing
- Model artifacts saved

## 4. Results

| Model | ROC-AUC |
|------|--------|
| MLP | ~0.976 |
| Gradient Boosting | ~0.978 |

- Both models demonstrate strong discrimination
- Calibration analysis indicates the uncalibrated gradient boosting model
  provides the best probability reliability

## 5. Segment Analysis
Performance was evaluated across:
- Age bands
- Gender
- Smoking history

Results indicate stable discrimination across segments, with recall increasing
in higher-risk groups consistent with clinical expectations.

## 6. Error Analysis
Most errors occur near the decision boundary and in low-prevalence subgroups.
False negatives are more common in younger populations due to weaker signal.

## 7. Risks and Ethics
- Model predictions should not be used as standalone diagnostic decisions
- Potential bias due to synthetic data assumptions
- Threshold selection must align with clinical priorities

## 8. Conclusion
The gradient boosting model was selected as the final model based on strong
discrimination, stable subgroup behavior, and superior calibration performance.
This work demonstrates the importance of evaluation beyond raw accuracy in
healthcare ML tasks.