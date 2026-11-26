#!/usr/bin/env python3
"""Figures, SHAP and permutation-importance analysis for CardioNepal (reads run_experiment.py outputs)."""
import json, sys, warnings
import numpy as np, pandas as pd, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import shap

warnings.filterwarnings("ignore")
OUT = sys.argv[1]
FIG = f"{OUT}/fig"
import os; os.makedirs(FIG, exist_ok=True)
R = json.load(open(f"{OUT}/results.json"))
P = np.load(f"{OUT}/predictions.npz")
M = joblib.load(f"{OUT}/models.joblib")
yte, ytr = P["y_test"], P["y_train"]
MODELS = ["lr", "dt", "rf", "svm"]
NAMES = {"lr": "Logistic Regression", "dt": "Decision Tree", "rf": "Random Forest", "svm": "SVM"}
COL = {"lr": "#2a78d6", "dt": "#eb6834", "rf": "#148f64", "svm": "#6250d6"}
LS = {"lr": "-", "dt": "--", "rf": "-.", "svm": ":"}
MK = {"lr": "o", "dt": "s", "rf": "^", "svm": "D"}
INK, MUTED, GRID = "#1f2933", "#5b6770", "#e3e7ea"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
                     "savefig.dpi": 300, "figure.dpi": 100, "legend.frameon": False})
FEAT_LABEL = {"age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting BP", "chol": "Cholesterol",
              "fbs": "Fasting blood sugar", "restecg": "Resting ECG", "thalach": "Max heart rate", "exang": "Exercise angina",
              "oldpeak": "ST depression", "slope": "ST slope", "ca": "Major vessels", "thal": "Thallium result"}
save = lambda f, n: (f.savefig(f"{FIG}/{n}.png", bbox_inches="tight", facecolor="white"), plt.close(f))
thr = {m: R["calibration"][m]["threshold"] for m in MODELS}

# ------------------------------------------------------------ Fig: class distribution
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})
nd = R["data"]["num_distribution"]; lv = ["0", "1", "2", "3", "4"]
vals = [nd[k] for k in lv]
bars = ax[0].bar(lv, vals, color=["#2a78d6"] + ["#eb6834"] * 4, width=.62)
for b, v in zip(bars, vals): ax[0].text(b.get_x() + b.get_width() / 2, v + 2, str(v), ha="center", color=INK, fontsize=8)
ax[0].set_xlabel("Original outcome value (num)"); ax[0].set_ylabel("Patients"); ax[0].set_ylim(0, 190)
ax[0].set_title("Original 5-level outcome", loc="left", fontsize=9, color=INK)
sp = R["split"]
x = np.arange(2); w = .36
a = [sp["train_absent"], sp["test_absent"]]; p_ = [sp["train_present"], sp["test_present"]]
b1 = ax[1].bar(x - w / 2, a, w, color="#2a78d6", label="Absent (0)")
b2 = ax[1].bar(x + w / 2, p_, w, color="#eb6834", label="Present (1)")
for bb in list(b1) + list(b2): ax[1].text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 2, int(bb.get_height()), ha="center", fontsize=8, color=INK)
ax[1].set_xticks(x, [f"Training (n={sp['n_train']})", f"Test (n={sp['n_test']})"]); ax[1].set_ylim(0, 160)
ax[1].set_title("Binary label after stratified split", loc="left", fontsize=9, color=INK); ax[1].legend(loc="upper right", fontsize=8)
fig.tight_layout(); save(fig, "fig_class_distribution")

# ------------------------------------------------------------ Fig: ROC
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5))
for m in MODELS:
    fpr, tpr, _ = roc_curve(yte, P[m]); auc = R["test"][m]["at_threshold"]["roc_auc"]
    ax[0].plot(fpr, tpr, color=COL[m], ls=LS[m], lw=1.8, label=f"{NAMES[m]} (AUC {auc:.3f})")
    pr, rc, _ = precision_recall_curve(yte, P[m])
    ax[1].plot(rc, pr, color=COL[m], ls=LS[m], lw=1.8, label=f"{NAMES[m]} (AP {R['test'][m]['at_threshold']['pr_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=(0, (1, 3)))
ax[0].set_xlabel("1 − Specificity (false-positive rate)"); ax[0].set_ylabel("Sensitivity (true-positive rate)")
ax[0].set_title("ROC curves (test set, n = 61)", loc="left", fontsize=9, color=INK); ax[0].legend(loc="lower right", fontsize=7.5)
ax[1].axhline(yte.mean(), color=MUTED, lw=.8, ls=(0, (1, 3)))
ax[1].set_xlabel("Recall (sensitivity)"); ax[1].set_ylabel("Precision"); ax[1].set_ylim(0.2, 1.02)
ax[1].set_title("Precision–recall curves (test set)", loc="left", fontsize=9, color=INK); ax[1].legend(loc="lower left", fontsize=7.5)
for a_ in ax: a_.set_aspect("auto")
fig.tight_layout(); save(fig, "fig_roc_pr")

# ------------------------------------------------------------ Fig: confusion matrices
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.35))
for a_, m in zip(ax, MODELS):
    t = R["test"][m]["at_threshold"]; cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
    a_.imshow(cm, cmap="Blues", vmin=0, vmax=40); a_.grid(False)
    for i in range(2):
        for j in range(2): a_.text(j, i, cm[i, j], ha="center", va="center", fontsize=13, color="white" if cm[i, j] > 20 else INK, fontweight="bold")
    a_.set_xticks([0, 1], ["Absent", "Present"]); a_.set_yticks([0, 1], ["Absent", "Present"])
    a_.set_xlabel("Predicted"); a_.set_title(f"{NAMES[m]}\nthreshold {t['threshold']:.3f}", fontsize=8.5, color=INK)
    for s in a_.spines.values(): s.set_visible(False)
ax[0].set_ylabel("Actual")
fig.tight_layout(); save(fig, "fig_confusion")

# ------------------------------------------------------------ Fig: metrics with CI
fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.6), sharey=True)
for a_, (k, ttl) in zip(ax, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1-score"), ("brier", "Brier score\n(lower is better)")]):
    for i, m in enumerate(MODELS):
        v = R["test"][m]["at_threshold"][k]; lo, hi = R["test"][m]["ci95"][k]
        a_.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt=MK[m], color=COL[m], ms=6, capsize=3, lw=1.4)
        a_.text(v, i + .36, f"{v:.3f}", ha="center", fontsize=7.5, color=INK)
    a_.set_title(ttl, fontsize=8.5, color=INK); a_.grid(axis="y", visible=False)
ax[0].set_yticks(range(4), [NAMES[m] for m in MODELS]); ax[0].set_ylim(3.7, -0.5)
fig.supxlabel("Point estimate and percentile bootstrap 95% CI (2,000 resamples of the test set)", fontsize=8, color=MUTED, y=-.04)
fig.tight_layout(); save(fig, "fig_metrics_ci")

# ------------------------------------------------------------ Fig: calibration
fig, ax = plt.subplots(2, 4, figsize=(7.6, 4.1), sharex=True, sharey=True)
for j, m in enumerate(MODELS):
    for i, (y_, p_, nb, lab) in enumerate([(ytr, P[f"{m}_oof"], 6, "Training, out-of-fold"), (yte, P[m], 4, "Test")]):
        a_ = ax[i, j]
        fp_, mp_ = calibration_curve(y_, p_, n_bins=nb, strategy="quantile")
        a_.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=(0, (1, 3)))
        a_.plot(mp_, fp_, marker=MK[m], color=COL[m], lw=1.6, ms=5)
        a_.set_title(f"{NAMES[m]}\n{lab}", fontsize=8, color=INK)
        a_.set_xlim(0, 1); a_.set_ylim(0, 1)
        if j == 0: a_.set_ylabel("Observed fraction positive")
        if i == 1: a_.set_xlabel("Mean predicted probability")
fig.tight_layout(); save(fig, "fig_calibration")

# ------------------------------------------------------------ Fig: probability distributions
fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.5), sharey=True)
bins = np.linspace(0, 1, 11)
for a_, m in zip(ax, MODELS):
    a_.hist(P[m][yte == 0], bins=bins, color="#2a78d6", alpha=.8, label="Absent", rwidth=.9)
    a_.hist(P[m][yte == 1], bins=bins, color="#eb6834", alpha=.8, label="Present", rwidth=.9, bottom=np.histogram(P[m][yte == 0], bins=bins)[0])
    a_.axvline(thr[m], color=INK, lw=1, ls="--"); a_.set_title(NAMES[m], fontsize=8.5, color=INK); a_.set_xlabel("Predicted probability")
ax[0].set_ylabel("Test patients"); ax[0].legend(fontsize=7.5, loc="upper center")
fig.tight_layout(); save(fig, "fig_prob_dist")

# ------------------------------------------------------------ Fig: nested CV
if "nested_cv" in R:
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    for i, m in enumerate(MODELS):
        v = [r["roc_auc"] for r in R["nested_cv"][m]]
        ax.scatter(v, np.full(len(v), i) + np.linspace(-.12, .12, len(v)), color=COL[m], marker=MK[m], s=28, zorder=3)
        ax.plot([np.mean(v)] * 2, [i - .27, i + .27], color=INK, lw=1.6)
    ax.set_yticks(range(4), [NAMES[m] for m in MODELS]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Outer-fold ROC-AUC (nested 5-fold CV; vertical bar = mean)")
    fig.tight_layout(); save(fig, "fig_nested_cv")

# ------------------------------------------------------------ SHAP + importance
def group_of(col):          # 'num__age' / 'bin__sex' / 'cat__cp_1.0' -> original feature
    c = col.split("__", 1)[1]
    for f in FEAT_LABEL:
        if c == f or c.startswith(f + "_"): return f
    raise ValueError(col)

rng = np.random.default_rng(42)
FE = list(FEAT_LABEL)
def aggregate(vals, cols):
    g = np.array([group_of(c) for c in cols]); return np.stack([vals[:, g == f].sum(1) for f in FE], 1)

Xtr, Xte = M["Xtr"], M["Xte"]
out = {}
# --- RF (TreeExplainer, interventional, probability units)
rf_pipe = M["base"]["rf"]; prep = rf_pipe.named_steps["prep"]; rf = rf_pipe.named_steps["clf"]
cols = list(prep.get_feature_names_out())
Ttr, Tte = prep.transform(Xtr), prep.transform(Xte)
bg = Ttr[rng.choice(len(Ttr), 100, replace=False)]
ex = shap.TreeExplainer(rf, data=bg, feature_perturbation="interventional", model_output="probability")
sv = ex.shap_values(Tte, check_additivity=False)
sv = sv[..., 1] if sv.ndim == 3 else sv
base_rf = float(np.ravel(ex.expected_value)[-1] if np.ndim(ex.expected_value) else ex.expected_value)
p_rf = rf.predict_proba(Tte)[:, 1]
add_err = np.abs(base_rf + sv.sum(1) - p_rf)
out["rf_additivity_max_abs_error"] = float(add_err.max()); out["rf_base_value"] = base_rf
S_rf = aggregate(sv, cols)
out["rf_calibration_method"] = R["calibration"]["rf"]["chosen"]
# --- LR (LinearExplainer, log-odds)
lr_pipe = M["base"]["lr"]; prep_l = lr_pipe.named_steps["prep"]; lr = lr_pipe.named_steps["clf"]
Ttr_l, Tte_l = prep_l.transform(Xtr), prep_l.transform(Xte)
exl = shap.LinearExplainer(lr, shap.maskers.Independent(Ttr_l))
svl = exl.shap_values(Tte_l); svl = svl[..., 1] if svl.ndim == 3 else svl
S_lr = aggregate(svl, list(prep_l.get_feature_names_out()))
lo = lr.decision_function(Tte_l); out["lr_additivity_max_abs_error"] = float(np.abs(float(np.ravel(exl.expected_value)[0]) + S_lr.sum(1) - lo).max())

glob_rf = np.abs(S_rf).mean(0); glob_lr = np.abs(S_lr).mean(0)
# --- RF impurity importance (aggregated) and permutation importance (test, ROC-AUC)
mdi = aggregate(rf.feature_importances_.reshape(1, -1), cols)[0]
pi = permutation_importance(rf_pipe, Xte, yte, scoring="roc_auc", n_repeats=50, random_state=42, n_jobs=1)
perm = dict(zip(Xte.columns, pi.importances_mean)); perm_sd = dict(zip(Xte.columns, pi.importances_std))
from scipy.stats import spearmanr
imp = pd.DataFrame({"feature": FE, "rf_shap_mean_abs": glob_rf, "lr_shap_mean_abs": glob_lr, "rf_mdi": mdi,
                    "rf_perm_auc_drop": [perm[f] for f in FE], "rf_perm_sd": [perm_sd[f] for f in FE]})
imp = imp.sort_values("rf_shap_mean_abs", ascending=False).reset_index(drop=True)
imp.to_csv(f"{OUT}/importance_table.csv", index=False)
out["spearman"] = {"rf_shap_vs_rf_mdi": float(spearmanr(imp.rf_shap_mean_abs, imp.rf_mdi)[0]),
                   "rf_shap_vs_rf_perm": float(spearmanr(imp.rf_shap_mean_abs, imp.rf_perm_auc_drop)[0]),
                   "rf_shap_vs_lr_shap": float(spearmanr(imp.rf_shap_mean_abs, imp.lr_shap_mean_abs)[0])}
out["top_rf"] = imp.feature[:5].tolist(); out["top_lr"] = imp.sort_values("lr_shap_mean_abs", ascending=False).feature[:5].tolist()
# --- signed means (direction) for the top features, using raw coded values
raw = Xte.reset_index(drop=True)
out["rf_direction_corr"] = {f: float(np.corrcoef(raw[f], S_rf[:, FE.index(f)])[0, 1]) for f in FE}

# Fig: global importance (RF vs LR)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
order = list(imp.feature)[::-1]
ax[0].barh([FEAT_LABEL[f] for f in order], [imp.set_index("feature").loc[f, "rf_shap_mean_abs"] for f in order], color=COL["rf"], height=.62)
ax[0].set_xlabel("Mean |SHAP| (probability units)"); ax[0].set_title("Random Forest (TreeSHAP)", loc="left", fontsize=9, color=INK)
ax[1].barh([FEAT_LABEL[f] for f in order], [imp.set_index("feature").loc[f, "lr_shap_mean_abs"] for f in order], color=COL["lr"], height=.62)
ax[1].set_xlabel("Mean |SHAP| (log-odds units)"); ax[1].set_title("Logistic Regression (linear SHAP)", loc="left", fontsize=9, color=INK)
for a_ in ax: a_.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "fig_shap_global")

# Fig: beeswarm (RF)
expl = shap.Explanation(values=S_rf, base_values=np.full(len(S_rf), base_rf), data=raw[FE].values, feature_names=[FEAT_LABEL[f] for f in FE])
plt.figure(figsize=(6.4, 4.4)); shap.plots.beeswarm(expl, max_display=13, show=False, plot_size=None)
plt.xlabel("SHAP value (change in predicted probability)"); plt.gca().grid(False)
save(plt.gcf(), "fig_shap_beeswarm")

# Fig: comparison of importance measures (rank-normalised to max)
fig, ax = plt.subplots(figsize=(6.6, 3.8)); y = np.arange(len(imp)); h = .2
def nrm(v): v = np.asarray(v, float); return v / v.max()
for k, (col, lab, c) in enumerate([("rf_shap_mean_abs", "RF mean |SHAP|", COL["rf"]), ("rf_mdi", "RF impurity (MDI)", "#eda100"),
                                    ("rf_perm_auc_drop", "RF permutation (ΔAUC)", "#e34948"), ("lr_shap_mean_abs", "LR mean |SHAP|", COL["lr"])]):
    ax.barh(y + (k - 1.5) * h, nrm(np.clip(imp[col], 0, None)), height=h * .9, color=c, label=lab)
ax.set_yticks(y, [FEAT_LABEL[f] for f in imp.feature]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("Importance, scaled so that the largest value of each measure = 1"); ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); save(fig, "fig_importance_compare")

# Local explanations: pre-specified case rules
pr = P["rf"]; cases = {}
tp = np.where((yte == 1) & (pr >= thr["rf"]))[0]; tn = np.where((yte == 0) & (pr < thr["rf"]))[0]
err = np.where(((yte == 1) & (pr < thr["rf"])) | ((yte == 0) & (pr >= thr["rf"])))[0]
cases["A_true_positive_highest_p"] = int(tp[np.argmax(pr[tp])])
cases["B_true_negative_lowest_p"] = int(tn[np.argmin(pr[tn])])
fp = np.where((yte == 0) & (pr >= thr["rf"]))[0]
cases["C_false_positive_highest_p"] = int(fp[np.argmax(pr[fp])]) if len(fp) else None
out["cases"] = {}
for k, i in cases.items():
    if i is None: continue
    out["cases"][k] = dict(test_row=i, orig_index=int(M["te_idx"][i]), y=int(yte[i]), rf_prob=float(pr[i]),
                           lr_prob=float(P["lr"][i]), svm_prob=float(P["svm"][i]), base=base_rf,
                           values={f: float(raw.loc[i, f]) for f in FE},
                           shap={f: float(S_rf[i, FE.index(f)]) for f in FE})
    e1 = shap.Explanation(values=S_rf[i], base_values=base_rf, data=raw.loc[i, FE].values, feature_names=[FEAT_LABEL[f] for f in FE])
    plt.figure(figsize=(5.6, 3.9)); shap.plots.waterfall(e1, max_display=8, show=False)
    save(plt.gcf(), f"fig_shap_waterfall_{k[0]}")
np.savez(f"{OUT}/shap_values.npz", S_rf=S_rf, S_lr=S_lr, X_test=raw[FE].values, features=np.array(FE))
json.dump(out, open(f"{OUT}/shap_results.json", "w"), indent=1)

# ------------------------------------------------------------ Fig: architecture diagram
fig, ax = plt.subplots(figsize=(7.6, 5.2)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.grid(False)
def box(x, y, w, h, t, fc="#eaf2fb", ec="#2a78d6", fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.3)
def arrow(a, b, ls="-", c=INK):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1.2, color=c, ls=ls))
ax.add_patch(FancyBboxPatch((1, 39), 98, 32, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#f6f8fa", ec=GRID, lw=1))
ax.text(3, 68, "Offline research pipeline (Python, run on a developer machine)", fontsize=8.5, fontweight="bold", color=INK)
box(3, 50, 19, 13, "UCI Cleveland\ndata\n(303 × 14)", fc="#fff4e5", ec="#eb6834")
box(28, 50, 22, 13, "Leakage-aware pipeline:\nsplit, impute, encode,\nscale, optional SMOTE")
box(56, 50, 20, 13, "Tuning, calibration\nand threshold choice\n(training data only)")
box(82, 50, 15, 13, "Test-set\nevaluation\nand SHAP")
arrow((22.4, 56.5), (27.6, 56.5)); arrow((50.4, 56.5), (55.6, 56.5)); arrow((76.4, 56.5), (81.6, 56.5))
box(56, 41.5, 41, 5, "Outputs: results.json, figures, tables (Chapter 5)", fc="#e8f6ef", ec="#148f64", fs=7.2)
ax.add_patch(FancyBboxPatch((1, 1), 98, 31, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#f6f8fa", ec=GRID, lw=1))
ax.text(3, 29, "Deployed prototype (static, client-side, GitHub Pages)", fontsize=8.5, fontweight="bold", color=INK)
box(3, 8, 21, 14, "User's browser\n(form input and\nvalidation)", fc="#fff4e5", ec="#eb6834")
box(30, 8, 27, 14, "React 18 application\nsrc/lib/model.js:\nillustrative scoring function")
box(63, 8, 16, 14, "Result gauge\nand explanation\npanel")
box(84, 8, 13, 14, "Model & Data\ntab\n(metrics.js)")
arrow((24.4, 15), (29.6, 15)); arrow((57.4, 15), (62.6, 15)); arrow((79.4, 15), (83.6, 15), ls="--")
arrow((90.5, 40.8), (90.5, 22.8), ls=(0, (4, 3)), c="#148f64")
ax.text(89.3, 32.5, "metrics typed in\nby hand", fontsize=6.8, color="#148f64", va="center", ha="right")
ax.annotate("", xy=(30, 32.6), xytext=(30, 38.4), arrowprops=dict(arrowstyle="-|>", color="#e34948", lw=1.3, ls=(0, (2, 2))))
ax.text(32, 35.5, "no model export: the scoring weights are hand-specified", fontsize=7, color="#e34948", va="center")
save(fig, "fig_architecture")
print("figures done")
