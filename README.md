# Telecom Customer Churn Prediction

Predicts which telecom subscribers are likely to churn, using the classic
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
schema (19 customer/account/service features). Built as an end-to-end
supervised ML pipeline: EDA → preprocessing → model comparison → evaluation → demo app.

## Results

Trained on the real [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset (7,043 customers, 20% held out for testing = 1,409 test samples).

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 (churn class) |
|---|---|---|---|
| Logistic Regression (balanced) | 0.846 | 0.841 | 0.61 |
| XGBoost | 0.845 | **0.842** | 0.62 |

XGBoost and Logistic Regression landed within 0.001 ROC-AUC of each other —
effectively tied, with XGBoost taking the win by the smallest possible
margin. That's a meaningful result in itself: it means the signal in this
data is largely linear/additive (driven by a small number of strong
categorical features like contract type), so a much simpler, faster,
more interpretable model performs almost as well as a more complex one.
(See `artifacts/metrics.json` for full classification reports and confusion
matrices.)

**Precision/recall trade-off on the churn class:** recall is ~0.77-0.78
(the model catches ~77-78% of customers who actually churn) while precision
sits around 0.50-0.52 (about half of flagged "at-risk" customers don't
actually churn). This was a deliberate choice — `class_weight="balanced"` /
`scale_pos_weight` bias the model toward recall, since in churn prediction a
missed churner (lost customer, lost revenue) is typically far costlier than
a false alarm (an unnecessary but cheap retention offer).

**Top predictive features (XGBoost importances):** `Contract_Month-to-month`
dominates by a wide margin (0.336 importance — more than 3x the next
feature), followed by `OnlineSecurity_No`, `InternetService_Fiber optic`,
and `TechSupport_No`. Month-to-month contract customers are, by far, the
highest churn-risk segment — consistent with the EDA in `notebooks/eda.ipynb`.

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

# option A (recommended): download the real dataset from Kaggle
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# rename it to telco_churn.csv and place it in data/

# option B: no internet / no Kaggle account — use the synthetic fallback
python data/generate_data.py

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
- **Note on data**: this was trained and evaluated on the real Kaggle Telco
  Customer Churn CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, renamed to
  `data/telco_churn.csv`) — 7,043 real customer records. `data/generate_data.py`
  is kept in the repo as a synthetic fallback matching the same schema and
  churn-risk patterns (short tenure, month-to-month contracts, fiber + no
  tech support, electronic-check payment all increase churn risk), so the
  pipeline still runs end-to-end even without the real file.

## Possible extensions

- Hyperparameter tuning via `RandomizedSearchCV` / Optuna
- SHAP values for per-customer explainability
- Cost-sensitive threshold tuning (false negatives = lost customers are usually costlier than false positives = wasted retention offers)
