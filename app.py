
"""
Customer Churn Prediction API
=============================
Run:
    uvicorn app:app --reload
Docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn_pipeline import RAW_FEATURES
from schemas import (BatchPredictionRequest, BatchPredictionResponse,
                     CustomerFeatures, PredictionResponse)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("churn-api")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "churn_model.pkl"

# Decision threshold: 0.50, scikit-learn's default and the cut-off every model was
# compared at in section 4 of the notebook. Lowering it flags more customers and catches
# more churners; raising it does the reverse. Choosing a different value is a question
# about how many retention calls the team can make, not about the model, so it lives
# here as one named constant that can be changed without retraining.
DECISION_THRESHOLD = 0.50

state: dict = {"model": None}


# ---------------------------------------------------------------------------
# Lifespan: load the model once at startup, not on every request
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup  """
    state["model"] = joblib.load(MODEL_PATH)
    log.info("Loaded model from %s (threshold %.2f)", MODEL_PATH, DECISION_THRESHOLD)
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Turn Pydantic's raw error list into something a client can act on """
    problems = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"] if p not in ("body",)) or "body"
        problem = {"field": field, "message": err["msg"]}
        # For a missing field Pydantic reports the whole body as "input", which is
        # noise; only echo the value back when it is the value that was wrong.
        if err["type"] != "missing":
            received = err.get("input")
            problem["received"] = (received if not isinstance(received, (dict, list))
                                   else f"<{type(received).__name__}>")
        problems.append(problem)
    log.warning("Rejected request: %s", problems)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Invalid input",
            "detail": problems,
            "hint": ("Check field names, types and allowed category values. "
                     "The interactive schema at /docs lists every valid value."),
        },
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score(records: list[CustomerFeatures]) -> list[PredictionResponse]:
    model = state["model"]

    # Build the frame with exactly the columns the pipeline expects, in order.
    frame = pd.DataFrame([r.model_dump() for r in records])
    frame = frame.reindex(columns=RAW_FEATURES)

    try:
        probabilities = model.predict_proba(frame)[:, 1]
    except Exception as exc:
        log.exception("Scoring failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model failed to score the request: {exc}",
        )

    return [
        PredictionResponse(
            prediction="Yes" if p >= DECISION_THRESHOLD else "No",
            churn_probability=round(float(p), 4),
        )
        for p in probabilities
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"],
          responses={422: {"description": "Invalid input"}})
def predict(payload: CustomerFeatures):
    """Score a single customer: returns the churn prediction and its probability."""
    return _score([payload])[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(payload: BatchPredictionRequest):
    """Score up to 1,000 customers in one call — for nightly scoring of a segment."""
    results = _score(payload.customers)
    return BatchPredictionResponse(
        predictions=results,
        count=len(results),
        predicted_churners=sum(r.prediction == "Yes" for r in results),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)