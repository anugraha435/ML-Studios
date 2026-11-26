#!/usr/bin/env python3
"""
CardioNepal - leakage-aware experiment pipeline (implements Chapter 3 of the thesis).

Usage:
    python run_experiment.py --data <processed.cleveland.data | cleveland_uci_coded.csv> --out outputs

Protocol (all seeds fixed):
  1. Binary target: num > 0 -> 1 (heart disease present), num == 0 -> 0.
  2. One stratified 80/20 split (seed 42). The test set is used once, at the end.
  3. Preprocessing is part of every pipeline (median / mode imputation, one-hot encoding,
     standardisation, optional SMOTE) so that it is fitted on training folds only.
  4. Three imbalance conditions per model: none, class_weight='balanced', SMOTE.
  5. GridSearchCV (stratified 5-fold, scoring = ROC-AUC) on the training set only.
  6. Calibration method (none / sigmoid / isotonic) and the decision threshold are chosen
     from nested out-of-fold TRAINING predictions only.
  7. Final evaluation on the held-out test set, with percentile bootstrap CIs.
  8. Nested cross-validation on the training set (outer 5-fold, inner 5-fold grid search)
     for an approximately unbiased cross-validated estimate.
"""
import argparse, json, platform, time, warnings
import numpy as np, pandas as pd, joblib, sklearn, imblearn
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             brier_score_loss, accuracy_score, precision_score, recall_score, f1_score)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

SEED = 42
TUNE_SEED, CAL_OUTER_SEED, CAL_INNER_SEED, NESTED_SEED, BOOT_SEED = 42, 43, 44, 45, 42
N_BOOT = 2000
TARGET_SENS = 0.90          # screening-oriented threshold rule (training OOF predictions only)
PARSIMONY_TOL = 0.005       # CV ROC-AUC tolerance for preferring a simpler imbalance condition
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "num"]
NUM = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BIN = ["sex", "fbs", "exang"]
CAT = ["cp", "restecg", "slope", "thal"]
FEATURES = NUM + BIN + CAT
MODELS = ["lr", "dt", "rf", "svm"]
NAMES = {"lr": "Logistic Regression", "dt": "Decision Tree", "rf": "Random Forest",
         "svm": "Support Vector Machine"}
CONDS = ["none", "balanced", "smote"]          # order = increasing complexity (parsimony order)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(path):
    if path.endswith(".data"):
        d = pd.read_csv(path, header=None, names=COLS, na_values="?")
    else:
        d = pd.read_csv(path)
    assert list(d.columns) == COLS, d.columns
    return d


def preprocessor():
    num = SkPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = SkPipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM), ("bin", "passthrough", BIN), ("cat", cat, CAT)])


def make_pipe(model, cond):
    cw = "balanced" if cond == "balanced" else None
    clf = {
        "lr": LogisticRegression(solver="saga", max_iter=20000, random_state=SEED, class_weight=cw),
        "dt": DecisionTreeClassifier(random_state=SEED, class_weight=cw),
        "rf": RandomForestClassifier(random_state=SEED, n_jobs=1, class_weight=cw),
        "svm": SVC(random_state=SEED, class_weight=cw, probability=False, cache_size=500),
    }[model]
    steps = [("prep", preprocessor())]
    if cond == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps)


GRIDS = {
    # Table 3.2. Penalties are expressed through l1_ratio (0 = L2, 1 = L1, 0.5 = ElasticNet),
    # because the 'penalty' argument is deprecated in scikit-learn >= 1.8.
    "lr": {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__l1_ratio": [0, 0.5, 1]},
    "dt": {"clf__max_depth": [3, 5, 7, 10, None], "clf__min_samples_split": [2, 5, 10],
           "clf__criterion": ["gini", "entropy"]},
    "rf": {"clf__n_estimators": [50, 100, 200, 300, 500], "clf__max_depth": [5, 10, 15, 20, None],
           "clf__max_features": ["sqrt", "log2", None]},
    # gamma has no effect for the linear kernel, so it is not crossed with it
    "svm": [{"clf__kernel": ["linear"], "clf__C": [0.01, 0.1, 1, 10, 100]},
            {"clf__kernel": ["rbf", "poly"], "clf__C": [0.01, 0.1, 1, 10, 100],
             "clf__gamma": [0.001, 0.01, 0.1, 1, 10]}],
}


def scores_of(est, X):
    """Continuous score used for ROC/PR analysis."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), sensitivity=sens, specificity=spec,
                precision=prec, f1=f1, accuracy=(tp + tn) / len(y))


def full_metrics(y, p, thr):
    m = metrics_at(y, p, thr)
    m.update(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
             brier=brier_score_loss(y, p), threshold=thr)
    return m


def cal_slope_intercept(y, p):
    """Calibration slope and intercept (logistic recalibration on logit(p)); p clipped."""
    pc = np.clip(p, 1e-3, 1 - 1e-3)
    z = np.log(pc / (1 - pc)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def choose_threshold(y, p, target=TARGET_SENS):
    """Largest threshold whose OOF sensitivity >= target (screening-oriented rule)."""
    best = None
    for t in np.unique(p):
        m = metrics_at(y, p, t)
        if m["sensitivity"] >= target:
            best = t if best is None or t > best else best
    return float(best if best is not None else 0.5)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--skip-nested", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import os; os.makedirs(a.out, exist_ok=True)
    R = {"versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "imbalanced-learn": imblearn.__version__, "numpy": np.__version__,
                      "pandas": pd.__version__},
         "settings": dict(seed=SEED, n_boot=N_BOOT, target_sensitivity=TARGET_SENS,
                          parsimony_tolerance=PARSIMONY_TOL, tuning_metric="roc_auc",
                          cv_folds=5, test_size=0.2)}
    # ------------------------------------------------------------------ data
    d = load(a.data)
    y_all = (d["num"] > 0).astype(int).values
    X_all = d[FEATURES].copy()
    R["data"] = dict(n=int(len(d)), n_features=len(FEATURES), n_absent=int((y_all == 0).sum()),
                     n_present=int((y_all == 1).sum()), missing=d[FEATURES].isna().sum().to_dict(),
                     num_distribution=d["num"].value_counts().sort_index().to_dict())
    log("data", R["data"])
    idx = np.arange(len(d))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=SEED)
    Xtr, Xte, ytr, yte = X_all.iloc[tr_idx], X_all.iloc[te_idx], y_all[tr_idx], y_all[te_idx]
    R["split"] = dict(n_train=len(tr_idx), n_test=len(te_idx), train_present=int(ytr.sum()),
                      train_absent=int((ytr == 0).sum()), test_present=int(yte.sum()),
                      test_absent=int((yte == 0).sum()), train_idx=tr_idx.tolist(), test_idx=te_idx.tolist())
    log("split", {k: v for k, v in R["split"].items() if "idx" not in k})
    prev = ytr.mean()
    R["baseline"] = dict(train_prevalence=float(prev),
                         majority_accuracy_test=float(max(yte.mean(), 1 - yte.mean())),
                         constant_brier_test=float(brier_score_loss(yte, np.full(len(yte), prev))))
    # descriptive table (training set), by class
    desc = {}
    for c in NUM + BIN + CAT:
        s = Xtr[c]
        if c in NUM:
            desc[c] = {str(k): [float(g.median()), float(g.quantile(.25)), float(g.quantile(.75))]
                       for k, g in s.groupby(ytr)}
        else:
            desc[c] = {str(k): g.value_counts().sort_index().to_dict() for k, g in s.groupby(ytr)}
    R["train_descriptives"] = jsonable(desc)

    skf_tune = StratifiedKFold(5, shuffle=True, random_state=TUNE_SEED)
    # ------------------------------------------------------------------ tuning
    tuning, best = {}, {}
    for m in MODELS:
        tuning[m] = {}
        for cond in CONDS:
            t0 = time.time()
            gs = GridSearchCV(make_pipe(m, cond), GRIDS[m], scoring="roc_auc", cv=skf_tune,
                              n_jobs=2, refit=True, return_train_score=False)
            gs.fit(Xtr, ytr)
            res = gs.cv_results_
            i = gs.best_index_
            tuning[m][cond] = dict(best_params={k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                                   cv_auc_mean=float(res["mean_test_score"][i]),
                                   cv_auc_sd=float(res["std_test_score"][i]),
                                   n_candidates=int(len(res["mean_test_score"])),
                                   seconds=round(time.time() - t0, 1))
            best[(m, cond)] = gs.best_estimator_
            log(m, cond, tuning[m][cond])
    R["tuning"] = jsonable(tuning)
    # condition selection (pre-specified rule: parsimony within tolerance of best CV AUC)
    selected = {}
    for m in MODELS:
        top = max(tuning[m][c]["cv_auc_mean"] for c in CONDS)
        selected[m] = next(c for c in CONDS if tuning[m][c]["cv_auc_mean"] >= top - PARSIMONY_TOL)
    R["selected_condition"] = selected
    log("selected conditions", selected)

    # ------------------------------------------------------------ calibration
    cal = {}
    oof_store, final, base_models = {}, {}, {}
    cv_out = StratifiedKFold(5, shuffle=True, random_state=CAL_OUTER_SEED)
    cv_in = StratifiedKFold(5, shuffle=True, random_state=CAL_INNER_SEED)
    for m in MODELS:
        pipe = best[(m, selected[m])]
        methods = (["none"] if hasattr(pipe, "predict_proba") else []) + ["sigmoid", "isotonic"]
        cal[m] = {}
        for meth in methods:
            est = clone(pipe) if meth == "none" else CalibratedClassifierCV(clone(pipe), method=meth, cv=cv_in)
            oof = cross_val_predict(est, Xtr, ytr, cv=cv_out, method="predict_proba", n_jobs=1)[:, 1]
            slope, icpt = cal_slope_intercept(ytr, oof)
            cal[m][meth] = dict(oof_brier=float(brier_score_loss(ytr, oof)),
                                oof_auc=float(roc_auc_score(ytr, oof)), slope=slope, intercept=icpt)
            oof_store[(m, meth)] = oof
        chosen = min(cal[m], key=lambda k: (round(cal[m][k]["oof_brier"], 6), 0 if k == "none" else 1))
        cal[m]["chosen"] = chosen
        thr = choose_threshold(ytr, oof_store[(m, chosen)])
        cal[m]["threshold"] = thr
        om = metrics_at(ytr, oof_store[(m, chosen)], thr)
        cal[m]["oof_at_threshold"] = {k: om[k] for k in ("sensitivity", "specificity", "precision", "accuracy")}
        log(m, "calibration", {k: v for k, v in cal[m].items()})
        base = clone(pipe).fit(Xtr, ytr)
        fin = base if chosen == "none" else CalibratedClassifierCV(clone(pipe), method=chosen, cv=cv_in).fit(Xtr, ytr)
        final[m], base_models[m] = fin, base
    R["calibration"] = jsonable(cal)

    # ------------------------------------------------------------- test set
    test, preds = {}, {"y_test": yte, "y_train": ytr}
    rng = np.random.default_rng(BOOT_SEED)
    boots = rng.integers(0, len(yte), size=(N_BOOT, len(yte)))
    for m in MODELS:
        p = final[m].predict_proba(Xte)[:, 1]
        thr = cal[m]["threshold"]
        ent = dict(selected_condition=selected[m], calibration=cal[m]["chosen"],
                   best_params=tuning[m][selected[m]]["best_params"],
                   at_threshold=full_metrics(yte, p, thr), at_0_5=metrics_at(yte, p, 0.5))
        s, ic = cal_slope_intercept(yte, p)
        ent["test_cal_slope"], ent["test_cal_intercept"] = s, ic
        if hasattr(base_models[m], "predict_proba"):
            pu = base_models[m].predict_proba(Xte)[:, 1]
            ent["brier_uncalibrated"] = float(brier_score_loss(yte, pu))
            preds[f"{m}_uncal"] = pu
        else:
            ent["brier_uncalibrated"] = None
            preds[f"{m}_decision"] = base_models[m].decision_function(Xte)
        preds[m] = p
        preds[f"{m}_oof"] = oof_store[(m, cal[m]["chosen"])]
        test[m] = ent
        log(m, "test", {k: round(v, 3) for k, v in ent["at_threshold"].items()})
    # bootstrap CIs
    mets = ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f1", "accuracy", "brier"]
    bs = {m: {k: [] for k in mets} for m in MODELS}
    dauc = {f"{a_}-{b_}": [] for i, a_ in enumerate(MODELS) for b_ in MODELS[i + 1:]}
    skipped = 0
    for b in boots:
        yb = yte[b]
        if yb.min() == yb.max():
            skipped += 1
            continue
        auc_b = {}
        for m in MODELS:
            pb = preds[m][b]
            fm = full_metrics(yb, pb, cal[m]["threshold"])
            auc_b[m] = fm["roc_auc"]
            for k in mets:
                bs[m][k].append(fm[k])
        for k in dauc:
            a_, b_ = k.split("-")
            dauc[k].append(auc_b[a_] - auc_b[b_])
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    for m in MODELS:
        test[m]["ci95"] = {k: ci(bs[m][k]) for k in mets}
    R["bootstrap"] = dict(n_resamples=N_BOOT, skipped_single_class=skipped,
                          auc_difference_ci95={k: dict(mean_diff=float(np.mean(v)), ci95=ci(v)) for k, v in dauc.items()})
    R["test"] = jsonable(test)

    # ------------------------------------------------------------ nested CV
    if not a.skip_nested:
        outer = StratifiedKFold(5, shuffle=True, random_state=NESTED_SEED)
        nested = {}
        for m in MODELS:
            rows = []
            for k, (a_i, b_i) in enumerate(outer.split(Xtr, ytr)):
                gs = GridSearchCV(make_pipe(m, selected[m]), GRIDS[m], scoring="roc_auc", cv=skf_tune,
                                  n_jobs=2).fit(Xtr.iloc[a_i], ytr[a_i])
                e = gs.best_estimator_
                sc = scores_of(e, Xtr.iloc[b_i]); pr = e.predict(Xtr.iloc[b_i]); yb = ytr[b_i]
                rows.append(dict(fold=k + 1, roc_auc=roc_auc_score(yb, sc), accuracy=accuracy_score(yb, pr),
                                 sensitivity=recall_score(yb, pr), specificity=recall_score(yb, pr, pos_label=0),
                                 brier=(brier_score_loss(yb, e.predict_proba(Xtr.iloc[b_i])[:, 1])
                                        if hasattr(e, "predict_proba") else None),
                                 params={k_.replace("clf__", ""): v for k_, v in gs.best_params_.items()}))
            nested[m] = rows
            log(m, "nested", [round(r["roc_auc"], 3) for r in rows])
        R["nested_cv"] = jsonable(nested)

    json.dump(jsonable(R), open(f"{a.out}/results.json", "w"), indent=1)
    np.savez(f"{a.out}/predictions.npz", **preds)
    joblib.dump(dict(final=final, base=base_models, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
                     features=FEATURES, tr_idx=tr_idx, te_idx=te_idx), f"{a.out}/models.joblib")
    log("done")


if __name__ == "__main__":
    main()
