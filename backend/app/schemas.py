from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

ModelKey = Literal["lr", "dt", "rf", "svm"]


class PatientFeatures(BaseModel):
    """The 13 Cleveland predictors, coded exactly as in the UCI file used for training."""
    model_config = {"extra": "forbid"}
    age: float = Field(..., ge=20, le=100, description="years")
    sex: Literal[0, 1] = Field(..., description="0 = female, 1 = male")
    cp: Literal[1, 2, 3, 4] = Field(..., description="1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., ge=70, le=250, description="resting systolic BP, mm Hg")
    chol: float = Field(..., ge=100, le=700, description="serum cholesterol, mg/dl")
    fbs: Literal[0, 1] = Field(..., description="fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy")
    thalach: float = Field(..., ge=60, le=220, description="maximum heart rate achieved, bpm")
    exang: Literal[0, 1] = Field(..., description="exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=7, description="ST depression induced by exercise, mm")
    slope: Literal[1, 2, 3] = Field(..., description="slope of peak exercise ST segment: 1 up, 2 flat, 3 down")
    ca: Optional[Literal[0, 1, 2, 3]] = Field(None, description="major vessels coloured by fluoroscopy; null = missing (imputed)")
    thal: Optional[Literal[3, 6, 7]] = Field(None, description="3 normal, 6 fixed defect, 7 reversible defect; null = missing (imputed)")

    @field_validator("age", "trestbps", "chol", "thalach", "oldpeak")
    @classmethod
    def finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("must be a finite number")
        return v


class PredictRequest(BaseModel):
    model_config = {"extra": "forbid"}
    features: PatientFeatures
    models: list[ModelKey] = Field(default_factory=lambda: ["lr", "dt", "rf", "svm"], min_length=1, max_length=4)


class ModelPrediction(BaseModel):
    model: ModelKey
    probability: float
    threshold: float
    classification: Literal["above_threshold", "below_threshold"]
    calibration: str


class PredictResponse(BaseModel):
    predictions: list[ModelPrediction]
    notice: str


class Contribution(BaseModel):
    feature: str
    value: Optional[float]
    contribution: float


class ExplainResponse(BaseModel):
    model: Literal["rf", "lr"]
    units: str
    base_value: float
    prediction_check: float
    contributions: list[Contribution]
    notice: str
