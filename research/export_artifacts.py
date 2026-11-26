"""Export fitted CardioNepal models and verified results for the ML Studio web app and API.

Reads  : results/models.joblib, results/results.json, results/predictions.npz,
         results/shap_results.json, results/shap_values.npz, results/importance_table.csv
Writes : ../frontend/src/data/research/*.json   (browser inference + dashboards)
         ../backend/models/cardionepal.joblib   (server inference)
Nothing here invents values: every number comes from the fitted objects or the saved results.
"""
import json, sys, shutil
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT.parent / "frontend" / "src" / "data" / "research"
BACK = ROOT.parent / "backend" / "models"
OUT.mkdir(parents=True, exist_ok=True); BACK.mkdir(parents=True, exist_ok=True)

d = joblib.load(RES / "models.joblib")


def lr_background_mean():
    """Mean of the background used for the published linear SHAP values. shap's Independent masker subsamples
    large backgrounds to 100 rows, so the published values use that subsample's mean, not the full training mean."""
    import shap
    T = d["final"]["lr"].named_steps["prep"].transform(d["Xtr"])
    ex = shap.LinearExplainer(d["final"]["lr"].named_steps["clf"], shap.maskers.Independent(T))
    return np.asarray(ex.masker.data).mean(axis=0)


LR_BG_MEAN = lr_background_mean()
F = list(d["features"])
res = json.load(open(RES / "results.json"))
NUM, BIN, CAT = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"], ["sex", "fbs", "exang"], ["cp", "restecg", "slope", "thal"]
assert F == NUM + BIN + CAT


def clean(o):
    """Replace NaN/inf with None so the output is strict JSON (a NaN means the quantity is undefined)."""
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o): return None
    return o


def f(x):  # floats with full precision
    return [float(v) for v in np.ravel(x)]


def export_prep(pipe):
    ct = pipe.named_steps["prep"]
    num = ct.named_transformers_["num"]; cat = ct.named_transformers_["cat"]
    oh = cat.named_steps["oh"]
    names = list(ct.get_feature_names_out())
    return {
        "numCols": NUM, "binCols": BIN, "catCols": CAT,
        "median": f(num.named_steps["imp"].statistics_),
        "mean": f(num.named_steps["sc"].mean_), "scale": f(num.named_steps["sc"].scale_),
        "catMostFrequent": f(cat.named_steps["imp"].statistics_),
        "categories": [f(c) for c in oh.categories_],
        "outputNames": names,
    }


def export_tree(t):
    v = t.value[:, 0, :]; p1 = v[:, 1] / v.sum(axis=1)
    return {"left": t.children_left.tolist(), "right": t.children_right.tolist(), "feature": t.feature.tolist(),
            "threshold": f(t.threshold), "p1": f(p1)}


def export_member(clf):
    name = type(clf).__name__
    if name == "LogisticRegression":
        return {"kind": "lr", "coef": f(clf.coef_[0]), "intercept": float(clf.intercept_[0])}
    if name == "DecisionTreeClassifier":
        return {"kind": "tree", "trees": [export_tree(clf.tree_)]}
    if name == "RandomForestClassifier":
        return {"kind": "tree", "trees": [export_tree(e.tree_) for e in clf.estimators_]}
    if name == "SVC":
        assert clf.kernel == "rbf" and list(clf.classes_) == [0, 1]
        g = clf._gamma
        return {"kind": "svm-rbf", "gamma": float(g), "sv": [f(r) for r in clf.support_vectors_],
                "dual": f(clf.dual_coef_[0]), "intercept": float(clf.intercept_[0])}
    raise ValueError(name)


models = {}
for key, est in d["final"].items():
    if hasattr(est, "calibrated_classifiers_"):
        members = []
        for cc in est.calibrated_classifiers_:
            cal = cc.calibrators[0]
            assert type(cal).__name__ == "_SigmoidCalibration"
            members.append({"pipeline": export_prep(cc.estimator), "clf": export_member(cc.estimator.named_steps["clf"]),
                            "calibration": {"a": float(cal.a_), "b": float(cal.b_)}})
        cal_method = "sigmoid"
    else:
        members = [{"pipeline": export_prep(est), "clf": export_member(est.named_steps["clf"]), "calibration": None}]
        cal_method = "none"
    if key == "lr":  # training-set mean of the transformed features: the background for exact linear SHAP
        members[0]["trainMean"] = f(LR_BG_MEAN)
    models[key] = {"threshold": float(res["calibration"][key]["threshold"]), "calibration": cal_method, "members": members}

json.dump({"features": F, "models": models}, open(OUT / "models.json", "w"), separators=(",", ":"))

# ---- parity fixture: real test rows + edge cases (missing ca/thal) with sklearn's own probabilities
Xte = pd.DataFrame(d["Xte"], columns=F) if not isinstance(d["Xte"], pd.DataFrame) else d["Xte"][F]
rng = np.random.default_rng(0)
edge = Xte.sample(20, random_state=1).copy()
edge.iloc[:5, F.index("ca")] = np.nan; edge.iloc[3:8, F.index("thal")] = np.nan
edge.iloc[10:15, F.index("age")] += rng.integers(-20, 20, 5)
fx = pd.concat([Xte, edge]).reset_index(drop=True)
expected = {k: [float(v) for v in d["final"][k].predict_proba(fx)[:, 1]] for k in d["final"]}
json.dump({"rows": [[None if pd.isna(v) else float(v) for v in r] for r in fx[F].to_numpy()], "expected": expected},
          open(OUT / "parity.json", "w"))

# ---- backend artefact
_rng = np.random.default_rng(42)   # same draw as make_figures.py so the API reproduces the published SHAP values
_Ttr = d["final"]["rf"].named_steps["prep"].transform(d["Xtr"])
_bg = _Ttr[_rng.choice(len(_Ttr), 100, replace=False)]
_Ttr_lr = d["final"]["lr"].named_steps["prep"].transform(d["Xtr"])
joblib.dump({"rf_background": _bg, "lr_train_mean": LR_BG_MEAN, "final": d["final"], "features": F, "thresholds": {k: models[k]["threshold"] for k in models},
             "calibration": {k: models[k]["calibration"] for k in models}, "sklearn_version": res["versions"].get("sklearn")},
            BACK / "cardionepal.joblib", compress=3)

# ---- results, curves, SHAP, test-set explorer
yt = np.load(RES / "predictions.npz")
y = yt["y_test"].astype(int)
curves = {}
for k in ["lr", "dt", "rf", "svm"]:
    p = yt[k]
    fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
    # reliability: 5 equal-frequency bins on the 61 test predictions (small n: read with caution)
    order = np.argsort(p); bins = np.array_split(order, 5)
    rel = [{"meanPred": float(p[b].mean()), "obsRate": float(y[b].mean()), "n": int(len(b))} for b in bins]
    curves[k] = {"roc": {"fpr": f(fpr), "tpr": f(tpr)}, "pr": {"precision": f(pr), "recall": f(rc)},
                 "reliability": rel, "auc_check": float(roc_auc_score(y, p)), "probs": f(p)}
    assert abs(curves[k]["auc_check"] - res["test"][k]["at_threshold"]["roc_auc"]) < 1e-9, k
json.dump({"yTest": y.tolist(), "curves": curves}, open(OUT / "curves.json", "w"))

sv = np.load(RES / "shap_values.npz", allow_pickle=True)
shap = json.load(open(RES / "shap_results.json"))
shap["S_rf"] = [f(r) for r in sv["S_rf"]]; shap["S_lr"] = [f(r) for r in sv["S_lr"]]
shap["X_test"] = [[None if np.isnan(v) else float(v) for v in r] for r in sv["X_test"]]
shap["features"] = [str(x) for x in sv["features"]]
shap["importance"] = pd.read_csv(RES / "importance_table.csv").to_dict(orient="records")
json.dump(clean(shap), open(OUT / "shap.json", "w"), allow_nan=False)

json.dump(clean(res), open(OUT / "results.json", "w"), allow_nan=False)
print("exported", {p.name: p.stat().st_size for p in OUT.iterdir()}, BACK.iterdir)
