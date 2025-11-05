# Bank Loan Approval Prediction

## Overview
A simple ML pipeline to predict whether a bank loan application will be approved using Kaggle's Loan Prediction dataset.

## What it does
- Performs EDA and produces 2 visualizations.
- Cleans and preprocesses data (imputation, encoding, simple feature engineering).
- Trains Logistic Regression and Random Forest.
- Evaluates models (accuracy, precision, recall, F1); selects best model by F1.
- Saves best model and a 1-page PDF report.

## How to run
1. Place `train.csv` (Kaggle dataset) in this folder.
2. `pip install -r requirements.txt`
3. `python loan_pipeline.py`
4. See `output/` folder for visuals, `one_page_report.pdf`, `best_model.pkl`, and `model_comparison.csv`.

## Files to submit (to Google Form)
- 1-page PDF: `output/one_page_report.pdf`
- GitHub link: this repository
- Demo video: record showing running the script & results

## Chosen metric
F1-score — balances precision and recall; appropriate because both false positives (approving bad loans) and false negatives (rejecting good customers) are costly.
