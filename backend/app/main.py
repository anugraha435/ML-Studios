"""CardioNepal inference API (FastAPI). Run: uvicorn app.main:app --reload"""
from __future__ import annotations
import logging, os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .schemas import ExplainResponse, PredictRequest, PredictResponse
from .service import NOTICE, ModelService

log = logging.getLogger("cardionepal")
RATE = os.environ.get("RATE_LIMIT", "30/minute")
limiter = Limiter(key_func=get_remote_address)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["svc"] = ModelService()
    yield


app = FastAPI(title="CardioNepal inference API", version="1.0.0", lifespan=lifespan,
              description="Educational research API. " + NOTICE)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:4173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"], max_age=600)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.update({"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
                         "Cache-Control": "no-store", "X-Frame-Options": "DENY"})
    return resp


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error")  # details stay in server logs only
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "model_loaded": "svc" in state}


@app.get("/api/v1/model-info", tags=["meta"])
def model_info():
    s: ModelService = state["svc"]
    return {"features": s.features, "models": {k: {"threshold": s.thresholds[k], "calibration": s.calibration[k]} for k in s.thresholds},
            "training": {"dataset": "UCI Heart Disease, Cleveland subset (n=303)", "n_train": 242, "n_test": 61},
            "notice": NOTICE}


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["inference"])
@limiter.limit(RATE)
def predict(request: Request, body: PredictRequest):
    s: ModelService = state["svc"]
    return {"predictions": s.predict(body.features.model_dump(), list(dict.fromkeys(body.models))), "notice": NOTICE}


@app.post("/api/v1/explain/{model}", response_model=ExplainResponse, tags=["inference"])
@limiter.limit(RATE)
def explain(request: Request, model: str, body: PredictRequest):
    if model not in ("rf", "lr"):
        raise HTTPException(status_code=404, detail="Explanations are available for 'rf' (TreeSHAP) and 'lr' (linear SHAP).")
    return state["svc"].explain(body.features.model_dump(), model)
