# Experiment Log — LAB 2A

## Experiment 1: Baseline MLP
- Model: PyTorch MLP
- Features: All numerical + categorical
- Loss: BCEWithLogitsLoss
- Optimizer: Adam (lr=1e-3)
- ROC-AUC (test): ~0.976
- Notes: Strong discrimination, conservative recall at default threshold

## Experiment 2: Gradient Boosting
- Model: GradientBoostingClassifier
- Parameters:
  - n_estimators=200
  - learning_rate=0.05
  - max_depth=3
- ROC-AUC (test): ~0.978
- Notes: Slightly stronger ranking than MLP

## Experiment 3: Calibration
- Method: Platt scaling
- Validation set used for calibration
- Result: No improvement in Brier score
- Decision: Retain uncalibrated model

## Final Selection
- Selected Model: Gradient Boosting (uncalibrated)
- Rationale: Best calibration-quality tradeoff
- Artifact: `gradient_boosting_diabetes_v1.joblib`