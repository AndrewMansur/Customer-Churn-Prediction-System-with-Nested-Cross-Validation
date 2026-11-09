"""HTTP API that serves churn predictions.

Run with:  uvicorn churn.api:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from churn.predict import DEFAULT_THRESHOLD, load_model, predict_proba
from churn.schemas import BatchRequest, BatchResponse, Customer, Health, Prediction
from churn.train import MODEL_PATH


def create_app(model_path: Path | str | None = None) -> FastAPI:
    """Build the app. The model path can be set here or with CHURN_MODEL_PATH."""
    path = model_path or os.environ.get("CHURN_MODEL_PATH") or MODEL_PATH
    threshold = float(os.environ.get("CHURN_THRESHOLD", DEFAULT_THRESHOLD))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load once at startup. If it's missing, keep running so /health can
        # report the problem instead of the server failing to start.
        try:
            app.state.model = load_model(path)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            app.state.model = None
        yield

    app = FastAPI(
        title="Customer Churn Prediction API",
        description="Predicts how likely a telecom customer is to cancel.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def get_model():
        if app.state.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Train one with: python -m churn.train",
            )
        return app.state.model

    def to_predictions(probabilities: list[float]) -> list[Prediction]:
        return [
            Prediction(
                churn_probability=round(p, 4),
                will_churn=p >= threshold,
                threshold=threshold,
            )
            for p in probabilities
        ]

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/health", response_model=Health)
    def health():
        return Health(status="ok" if app.state.model is not None else "model_not_loaded")

    @app.post("/predict", response_model=Prediction)
    def predict(customer: Customer):
        """Churn probability for one customer."""
        probabilities = predict_proba(get_model(), [customer.model_dump()])
        return to_predictions(probabilities)[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: BatchRequest):
        """Churn probabilities for up to 1,000 customers at once."""
        records = [c.model_dump() for c in request.customers]
        return BatchResponse(predictions=to_predictions(predict_proba(get_model(), records)))

    return app


app = create_app()
