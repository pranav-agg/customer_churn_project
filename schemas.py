"""
Request and response models for the churn API.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Literal types generated from the training-time category lists.
Gender = Literal["Female", "Male"]
YesNo = Literal["Yes", "No"]
YesNoPhone = Literal["Yes", "No", "No phone service"]
YesNoInternet = Literal["Yes", "No", "No internet service"]
InternetType = Literal["DSL", "Fiber optic", "No"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
PaymentType = Literal[
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]


class CustomerFeatures(BaseModel):
    """One customer record, in the same shape as a row of the source dataset.
    """

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
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
                "MonthlyCharges": 94.4,
                "TotalCharges": 189.5,
            }
        },
    )

    customerID: Optional[str] = Field(
        default=None, description="Optional; ignored by the model.")
    gender: Gender
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0, le=600, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: YesNoPhone
    InternetService: InternetType
    OnlineSecurity: YesNoInternet
    OnlineBackup: YesNoInternet
    DeviceProtection: YesNoInternet
    TechSupport: YesNoInternet
    StreamingTV: YesNoInternet
    StreamingMovies: YesNoInternet
    Contract: ContractType
    PaperlessBilling: YesNo
    PaymentMethod: PaymentType
    MonthlyCharges: float = Field(..., ge=0, le=10_000)
    # Accepts a number or a numeric string: the source file stores this as text,
    # and a blank means "new customer, never billed" — the same case the pipeline
    # handles for tenure == 0.
    TotalCharges: Union[float, str] = Field(..., description="Number, or a numeric string")

    @field_validator("TotalCharges")
    @classmethod
    def _validate_total_charges(cls, v):
        if isinstance(v, str):
            stripped = v.strip()
            if stripped == "":
                return 0.0          # blank = never billed
            try:
                v = float(stripped)
            except ValueError:
                raise ValueError(
                    f"TotalCharges must be a number or a numeric string, got {v!r}")
        if v < 0:
            raise ValueError("TotalCharges cannot be negative")
        if v > 1_000_000:
            raise ValueError("TotalCharges is implausibly large")
        return float(v)


class BatchPredictionRequest(BaseModel):
    """Score up to 1,000 customers in one call."""

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"prediction": "Yes", "churn_probability": 0.82}}
    )

    prediction: Literal["Yes", "No"]
    churn_probability: float = Field(..., ge=0, le=1)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    predicted_churners: int


class ErrorResponse(BaseModel):
    error: str
    detail: Union[str, list, dict]