"""FastAPI application for the F1 Race Predictor."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.predictor import (
    get_available_years,
    get_events,
    get_feature_importance,
    run_evaluation,
    run_prediction,
)
from api.live_predictor import run_live_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="F1 Race Predictor API",
    description="Predict Formula 1 race finishing positions using LightGBM.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/years")
def years():
    """Return all available season years."""
    return {"years": get_available_years()}


@app.get("/events")
def events(year: int = Query(..., description="Season year, e.g. 2025")):
    """Return all race events for a given year in race order."""
    result = get_events(year)
    if not result:
        raise HTTPException(status_code=404, detail=f"No events found for year {year}.")
    return {"year": year, "events": result}


@app.get("/predict")
def predict(
    year: int = Query(..., description="Season year"),
    event: str = Query(..., description="Event name, e.g. 'Australian Grand Prix'"),
):
    """Predict finishing positions for a specific race."""
    try:
        return run_prediction(year, event)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate")
def evaluate(year: int = Query(..., description="Season year to evaluate")):
    """Return full-season evaluation metrics."""
    try:
        return run_evaluation(year)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-importance")
def feature_importance(top_n: int = Query(25, description="Number of top features to return")):
    """Return feature importances from the trained model."""
    try:
        return {"features": get_feature_importance(top_n=top_n)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict-live")
def predict_live(
    year: int = Query(..., description="Season year (2025+)"),
    event: str = Query(..., description="Event name, e.g. 'Australian Grand Prix'"),
):
    """
    Predict finishing positions using live qualifying and practice data fetched
    from the FastF1 API.  No race results are used — leakage-free for 2025+.
    """
    try:
        return run_live_prediction(year, event)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
