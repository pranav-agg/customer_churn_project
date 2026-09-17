

# Customer Churn Prediction — Telco

End-to-end machine learning solution for predicting telecom customer churn, from raw data
through to a served REST API.

**Business problem.** A telecommunications company wants to identify customers likely to churn so
the retention team can engage them before they leave.

**Dataset.** IBM Telco Customer Churn — 7,043 customers, 20 attributes.
**Target.** `Churn` (Yes / No), 26.5% positive.
**Model.** Grid-searched Decision Tree Classifier inside a full scikit-learn preprocessing Pipeline.

---

## REPO LINK:

https://github.com/pranav-agg/customer_churn_project/

## Results at a glance

Test set: 2,113 customers held out from the start and untouched until final evaluation.

| Metric | Score | Cross-validated estimate |
|---|---|---|
| Accuracy | **0.743** | 0.745 |
| Precision | **0.511** | 0.515 |
| Recall | **0.771** | 0.750 |
| F1 Score | **0.614** | 0.609 |

Every metric lands within about half a point of what cross-validation on the training split
predicted, which is the result that says the model generalises rather than having been fitted to the
folds it was selected on.

The model surfaces **434 of 561 real churners (77.4%)**, misses 127, and flags 38.8% of the customer
base. Accuracy sits just above the 74.3% a "predict nobody churns" model scores, because that baseline earns its 73.5% by catching zero churners.

**Model comparison** (5-fold cross-validation on the training split, all metrics at the default 0.50
threshold):

| Configuration | CV Accuracy | CV Precision | CV Recall | CV F1 | Train F1 | Overfit gap |
|---|---|---|---|---|---|---|
| A · Default tree (unconstrained) | 0.734 | 0.500 | 0.523 | 0.511 | 0.997 | 0.486 |
| B · Constrained + balanced | 0.734 | 0.500 | 0.779 | 0.609 | 0.636 | 0.027 |
| **C · Tuned tree (GridSearchCV) ← final** | **0.743** | **0.511** | **0.7714** | **0.615** | 0.652 | **0.037** |
| D · Random Forest (benchmark) | 0.768 | 0.546 | 0.752 | 0.633 | 0.713 | 0.081 |


**Final hyperparameters:** `criterion='entropy'`, `max_depth=4`, `min_samples_leaf=50`,
`min_samples_split=2`, `class_weight='balanced'`, `random_state=42`.

---

## Key findings

1. **Contract type dominates — it alone is ~50% of the model.** The tree's first split is whether the
   customer is on a month-to-month contract, with an importance of **0.504**.
   Everything else is a refinement within that partition.  
2. **Tenure is the second axis (0.126).** Within month-to-month customers, the tree splits on how long
   someone has been around, isolating the first-year cohort as the high-risk group.
3. **Fibre-optic internet (0.088) flags the risky product segment**, consistent with the 41.8% churn
   rate in the EDA section — the premium tier is the leaky one.
4. **Support add-ons act as protection.** `OnlineSecurity = No` (0.033) and `TechSupport = No`
   (0.018) both appear as splits pushing customers into high-risk leaves — the protective pattern
   from EDA, now confirmed as something the model uses.

The highest-risk profile the model describes is : *month-to-month contract, under ~12 months tenure, fibre-optic internet, paying
by electronic check, few or no add-on services.* Customers matching it churn at several times the
base rate, and the model's own structure suggests the three interventions that move them out of that
leaf — **an annual contract offer, automatic payment enrolment, and a bundled security/tech-support
add-on**.

---

## Project structure

```
customer_churn_project/
├── data/
│   ├── TelcoCustomerChurn.csv                  source dataset
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb                    full analysis, executed with outputs
├── model/
│   └── churn_model.pkl                         fitted Pipeline (preprocessing + model)
├── churn_pipeline.py                           shared transformers — imported by BOTH
│                                               the notebook and the API
├── app.py                                      FastAPI service
├── schemas.py                                  Pydantic request/response models
├── requirements.txt
├── README.md
├── sample_batch_request.json
├── sample_request_1.json
└── sample_request_2.json
```

---

## Setup

Requires **Python 3.11+**.

```bash
cd customer_churn_project

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`scikit-learn` is pinned to 1.8.0 because `churn_model.pkl` was fitted with it — unpickling a fitted
estimator across scikit-learn versions is not guaranteed. On a different version, re-run the
notebook to regenerate the model file.

### Run the notebook

```bash
jupyter notebook notebook/churn_analysis.ipynb
```

It runs top-to-bottom from a clean kernel in about 115 seconds and rewrites `model/churn_model.pkl`.
The shipped copy is already executed with all outputs, so it can also just be read.

### Run the API

```bash
uvicorn app:app --reload
```

- Service: <http://127.0.0.1:8000>
- Interactive docs (Swagger): <http://127.0.0.1:8000/docs>


## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/predict` | Score one customer |
| POST | `/predict/batch` | Score up to 1,000 customers (addition — see note below) |
| GET | `/docs` | Swagger UI, generated by FastAPI |

`/predict` is the required endpoint. `/predict/batch` is a deliberate addition: the business problem
describes a retention team working a list of at-risk customers, which is a scoring job over a
segment rather than one customer at a time. It is nine lines and reuses the same scoring path.

### POST /predict

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

**Request** (`sample_request.json`):

```json
{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 94.40,
  "TotalCharges": 189.50
}

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8335
}
```

### Invalid input

Every categorical field is typed as a `Literal` of the values seen in training, so bad input is
rejected at the edge rather than silently encoded as all-zeros and returned as a confident,
meaningless answer. Errors name the field, the problem and the value received (HTTP 422):

```json
{
  "error": "Invalid input",
  "detail": [
    { "field": "tenure", "message": "Input should be a valid integer, unable to parse string as an integer", "received": "twelve" },
    { "field": "Contract", "message": "Field required" },
    { "field": "PaymentMethod", "message": "Input should be 'Electronic check', 'Mailed check', 'Bank transfer (automatic)' or 'Credit card (automatic)'", "received": "Bitcoin" }
  ],
  "hint": "Check field names, types and allowed category values. The interactive schema at /docs lists every valid value."
}
```

Other handled cases: a failure inside the model returns **500** with the reason, and any unexpected
error returns FastAPI's default 500 (the traceback stays in the server log, never in the response).
A blank `TotalCharges` on a `tenure = 0` customer is **accepted**, because that is a real day-one
customer, not an error. If `model/churn_model.pkl` is missing the service **refuses to start**,
naming the file in the traceback — a service that cannot score anything should not accept requests.

---

## Design decisions

**Everything is inside one Pipeline.** `churn_pipeline.py` defines `RawCleaner` → `FeatureEngineer` →
`ColumnTransformer` → classifier, and the whole thing is pickled together. The notebook imports it to
train; `app.py` imports it to serve. There is no preprocessing code in the API, so the two cannot
drift apart — and the test suite asserts the served answer matches the pipeline scored directly, to
4 decimal places.

**Leakage control.** The train/test split happens before anything is fitted. EDA is run on the
training split only. One-hot categories and every hyperparameter are derived from training data
alone. The test set is opened once, in §5, after the model is final.

**The 11 blank `TotalCharges` values** all belong to `tenure = 0` customers who have not been billed
yet, so they are filled with `0.0` — the factually correct value. Dropping them would have deleted
the entire brand-new-customer segment; median-imputing would have invented a billing history.

**"No internet service" is kept as its own category**, not collapsed into "No". Those customers churn
at 6.9%, the *lowest* of any group, while internet customers who declined the add-on churn at 42% —
collapsing them would have cancelled the signal out.

**Class imbalance** is handled with `class_weight='balanced'` (selected by the grid search, not
assumed).

---

## Precision or recall?

**Recall**, because the errors are not symmetric. A false negative loses a customer's entire
remaining lifetime value — perhaps $1,500–2,000 — and is unrecoverable once they have ported their
number. A false positive costs one phone call and possibly a modest discount. The errors differ by
roughly an order of magnitude, and the expensive one is the one recall controls.

**How the model acts on that.** `class_weight='balanced'` encodes the asymmetry during training,
making a missed churner cost the split criterion about 2.8x what a false alarm costs. The effect is
large and measurable: recall rises from 0.51 for the unconstrained baseline to 0.75 for the shipped
model. The result is 425 of 561 churners surfaced, at the price of 395 false alarms — roughly half
the retention team's calls going to customers who were not going to leave.


---

## Bonus work included

- Hyperparameter tuning via `GridSearchCV` (144 tree configurations + 8 forest configurations)
- Class imbalance handled via `class_weight='balanced'`, chosen by the grid search rather than assumed
- Random Forest
- Batch prediction endpoint


