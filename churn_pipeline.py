import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Churn"
ID_COLUMN = "customerID"

#Raw columns the models expects as input
RAW_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureBucket"
]

# The six optional add-on services, used to build the engagement feature.
ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Numeric columns after feature engineering.
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "NumAddOnServices"
]


TENURE_BINS = [-0.1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49m+"]

class RawCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            X = self._coerce(X)
            # New, never-billed customers genuinely owe nothing.
            if "TotalCharges" in X.columns and "tenure" in X.columns:
                new_customer = X["TotalCharges"].isna() & (X["tenure"].fillna(-1) == 0)
                X.loc[new_customer, "TotalCharges"] = 0.0
            return X

    @staticmethod
    def _coerce(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if ID_COLUMN in X.columns:
            X = X.drop(columns=[ID_COLUMN])
        for col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
            if col in X.columns:
                X[col] = pd.to_numeric(
                    X[col].astype("object").replace(r"^\s*$", np.nan, regex=True),
                    errors="coerce",
                )
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds the engineered features described in section 3 of the assignment.

    ``NumAddOnServices`` - count of the six optional services subscribed (0-6).
    A single engagement / stickiness score: the more services a household has
    wired into the provider, the more switching costs them.

    ``TenureBucket`` - lifecycle stage rather than raw month count. Churn risk
    is strongly non-linear in tenure, and buckets give the retention team a
    segment they can actually action.

    This is a stateless transform, so it is safe with respect to leakage: it
    uses no statistic learned across rows.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        tenure = X["tenure"].astype(float)

        X["NumAddOnServices"] = sum(
            (X[col].astype("string") == "Yes").astype(int) for col in ADDON_SERVICES
        )

        X["TenureBucket"] = pd.cut(
            tenure, bins=TENURE_BINS, labels=TENURE_LABELS
        ).astype("object")
        X["TenureBucket"] = X["TenureBucket"].fillna(TENURE_LABELS[0])

        return X

def build_preprocessor(numeric=None, categorical=None) -> ColumnTransformer:
    """One-hot encodes categoricals, passes numerics through untouched.

    ``handle_unknown='ignore'`` means an unseen category at serve time produces
    an all-zero block instead of a crash - important for an API.

    No scaling is applied: the final model is a tree ensemble, which is
    invariant to monotone rescaling, and leaving the numerics on their original
    scale keeps feature importances readable.
    """
    numeric = NUMERIC_FEATURES if numeric is None else list(numeric)
    categorical = CATEGORICAL_FEATURES if categorical is None else list(categorical)
    return ColumnTransformer(
        transformers=[
            ("cat",OneHotEncoder(handle_unknown="ignore", sparse_output=False),categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model, numeric=None, categorical=None) -> Pipeline:
    """Assembles clean -> engineer -> encode -> model as one fittable object."""
    return Pipeline(
        steps=[
            ("clean", RawCleaner()),
            ("engineer", FeatureEngineer()),
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", model),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Readable feature names coming out of the fitted preprocessor."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())
        