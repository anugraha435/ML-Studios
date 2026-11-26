"""Loads the fitted CardioNepal pipelines and runs inference / explanations."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ARTIFACT = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parent.parent / "models" / "cardionepal.joblib"))
NOTICE = ("Educational research demonstration on a historical Cleveland benchmark dataset. Not a medical device, "
          "not a diagnosis, and not validated on any Nepalese population.")
LABEL_GROUPS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "sex", "fbs", "exang", "cp", "restecg", "slope", "thal"]


class ModelService:
    def __init__(self, path: Path = ARTIFACT):
        a = joblib.load(path)
        self.final = a["final"]; self.features = a["features"]; self.thresholds = a["thresholds"]
        self.calibration = a["calibration"]; self.bg = a["rf_background"]; self.lr_mean = a["lr_train_mean"]
        self._tree_expl = None

    def frame(self, feats: dict) -> pd.DataFrame:
        return pd.DataFrame([{k: (np.nan if feats.get(k) is None else feats[k]) for k in self.features}])[self.features]

    def predict(self, feats: dict, models: list[str]) -> list[dict]:
        X = self.frame(feats); out = []
        for m in models:
            p = float(self.final[m].predict_proba(X)[0, 1]); t = float(self.thresholds[m])
            out.append({"model": m, "probability": p, "threshold": t, "calibration": self.calibration[m],
                        "classification": "above_threshold" if p >= t else "below_threshold"})
        return out

    @staticmethod
    def _group(col: str) -> str:
        c = col.split("__", 1)[1]
        for f in LABEL_GROUPS:
            if c == f or c.startswith(f + "_"):
                return f
        raise ValueError(col)

    def _aggregate(self, vals: np.ndarray, cols: list[str]) -> np.ndarray:
        g = np.array([self._group(c) for c in cols])
        return np.array([vals[g == f].sum() for f in LABEL_GROUPS])

    def explain(self, feats: dict, model: str) -> dict:
        X = self.frame(feats); pipe = self.final[model]; prep = pipe.named_steps["prep"]; clf = pipe.named_steps["clf"]
        cols = list(prep.get_feature_names_out()); T = prep.transform(X)
        if model == "rf":
            import shap
            if self._tree_expl is None:
                self._tree_expl = shap.TreeExplainer(clf, data=self.bg, feature_perturbation="interventional", model_output="probability")
            sv = self._tree_expl.shap_values(T, check_additivity=False)
            sv = sv[..., 1] if sv.ndim == 3 else sv
            base = float(np.ravel(self._tree_expl.expected_value)[-1]); units = "probability"
            check = float(clf.predict_proba(T)[0, 1]); phi = sv[0]
        else:  # linear SHAP with an independent background: phi_j = w_j (x_j - mean_j), log-odds units
            phi = clf.coef_[0] * (T[0] - self.lr_mean); units = "log-odds"
            base = float(clf.intercept_[0] + clf.coef_[0] @ self.lr_mean); check = float(clf.decision_function(T)[0])
        agg = self._aggregate(phi, cols)
        contribs = [{"feature": f, "value": None if pd.isna(X.iloc[0][f]) else float(X.iloc[0][f]), "contribution": float(v)}
                    for f, v in zip(LABEL_GROUPS, agg)]
        contribs.sort(key=lambda c: -abs(c["contribution"]))
        return {"model": model, "units": units, "base_value": base, "prediction_check": check, "contributions": contribs, "notice": NOTICE}
