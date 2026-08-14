# Telecom Customer Churn Prediction

Predicts which telecom subscribers are likely to churn, using the classic
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
schema (19 customer/account/service features). Built as an end-to-end
supervised ML pipeline: EDA → preprocessing → model comparison → evaluation → demo app.

## Results

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 |
|---|---|---|---|
| Logistic Regression (balanced) | 0.755 | **0.792** | 0.64 |
| XGBoost | 0.742 | 0.774 | 0.62 |

Logistic Regression won on held-out ROC-AUC — a good reminder that a well-regularized
linear model with class weighting can beat a tuned gradient-boosted tree on
a small, mostly-categorical dataset like this one. (See `artifacts/metrics.json`
for full classification reports and confusion matrices.)

**Top predictive features (XGBoost importances):** contract type
(month-to-month vs. annual), tenure, internet service type, tech support,
and payment method — consistent with the EDA in `notebooks/eda.ipynb`.

## Project structure

```
telco-churn-prediction/
├── data/
│   ├── generate_data.py   # synthetic fallback matching the real dataset's schema
│   └── telco_churn.csv    # swap in the real Kaggle CSV here if you have it
├── notebooks/
│   └── eda.ipynb          # exploratory analysis + plots driving feature choices
├── src/
│   └── train.py           # preprocessing pipeline, model comparison, evaluation
├── artifacts/              # generated: trained model + metrics (git-ignored or committed, your call)
├── app.py                 # Streamlit demo
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# optional: use the real Kaggle dataset instead of the synthetic fallback
# by placing it at data/telco_churn.csv (same column names)
python data/generate_data.py   # only needed if you don't have the real CSV

python src/train.py            # trains, evaluates, saves artifacts/churn_model.joblib
streamlit run app.py           # interactive demo
```

## Approach

- **Preprocessing**: numeric features (tenure, monthly/total charges) are
  median-imputed and standard-scaled; categorical features are
  most-frequent-imputed and one-hot encoded, all inside a single
  `sklearn.Pipeline` + `ColumnTransformer` so there's zero leakage between
  train/test folds.
- **Class imbalance** (~30% churn): handled via `class_weight="balanced"`
  (logistic regression) and `scale_pos_weight` (XGBoost), and evaluated with
  ROC-AUC/F1 rather than accuracy.
- **Model selection**: 5-fold stratified cross-validation on the training
  split, final comparison on a held-out 20% test set.
- **Note on data**: `data/telco_churn.csv` here is synthetically generated
  to match the real Kaggle dataset's schema and general churn-risk patterns
  (short tenure, month-to-month contracts, fiber + no tech support, and
  electronic-check payment all increase churn risk, matching known patterns
  in the real data). Swap in the real CSV for production-grade numbers.

## Possible extensions

- Hyperparameter tuning via `RandomizedSearchCV` / Optuna
- SHAP values for per-customer explainability
- Cost-sensitive threshold tuning (false negatives = lost customers are usually costlier than false positives = wasted retention offers)
