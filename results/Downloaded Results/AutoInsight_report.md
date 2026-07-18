# AutoInsight Report

## Business Question
Which customers are most likely to churn, and what factors contribute to their churn?

## Dataset Overview
- Rows: 7,043
- Columns: 21
- Duplicate rows: 0
- Memory footprint: 1.95 MB

## Modeling Setup
- Target column: churn
- Task type: classification

## Model Comparison
- Logistic Regression: weighted_f1=0.796; accuracy: 0.801
- Random Forest Classifier: weighted_f1=0.771; accuracy: 0.764

## Recommendations
- The dataset has no major structural quality warnings from the automated profile.
- Use Logistic Regression as the current baseline with weighted F1 of 0.796.
- Compare future models against this score using the same validation split or cross-validation.