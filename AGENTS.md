# Project Instructions for Codex

## Project
This is an end-to-end predictive maintenance ML web app that predicts whether a machine will fail within 24 hours.

## Tech stack
- Python
- scikit-learn
- XGBoost
- joblib
- Vercel serverless API
- Simple frontend

## Rules
- Do not change production code directly unless asked.
- Prefer small, clear changes.
- Explain changes before editing.
- Do not add unnecessary frameworks.
- Do not generate code unless the user asks for code.

## ML context
- Target: failure_within_24h
- Model: XGBoost pipeline saved as models/xgboost_pipeline.pkl
- API endpoint: /api/predict
- Important planned features:
  - SHAP explanations: make sure to use branch of SHAP and making sure it works before adding it to MAIN branch
  - MLflow tracking
  - Keeping record of inputs from users for only me to see
  - data drift monitoring

## Commands
- Run inference tests before changing deployment logic.
- Keep API compatible with the frontend input fields.