"""Independent checks: (1) data integrity vs Table 3.1, (2) recompute test metrics from saved probabilities."""
import json, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
data, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(data); R = json.load(open(f"{out}/results.json")); P = np.load(f"{out}/predictions.npz")
ok = True
def chk(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)
chk("303 rows, 14 columns", d.shape == (303, 14))
y = (d.num > 0).astype(int)
chk("164 absent / 139 present", (y == 0).sum() == 164 and (y == 1).sum() == 139)
chk("num counts 164/55/36/35/13", d.num.value_counts().sort_index().tolist() == [164, 55, 36, 35, 13])
chk("missing ca=4, thal=2", d.ca.isna().sum() == 4 and d.thal.isna().sum() == 2)
chk("age range 29-77", d.age.min() == 29 and d.age.max() == 77)
chk("97 women", (d.sex == 0).sum() == 97)
yt = P["y_test"]
for m in ["lr", "dt", "rf", "svm"]:
    p = P[m]; t = R["test"][m]["at_threshold"]; thr = t["threshold"]; pred = (p >= thr).astype(int)
    tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum()); fn = int(((pred == 0) & (yt == 1)).sum()); tp = int(((pred == 1) & (yt == 1)).sum())
    chk(f"{m} confusion", (tn, fp, fn, tp) == (t["tn"], t["fp"], t["fn"], t["tp"]))
    chk(f"{m} AUC/PR/Brier", abs(roc_auc_score(yt, p) - t["roc_auc"]) < 1e-9 and abs(average_precision_score(yt, p) - t["pr_auc"]) < 1e-9 and abs(brier_score_loss(yt, p) - t["brier"]) < 1e-9)
    chk(f"{m} sens/spec", abs(tp / (tp + fn) - t["sensitivity"]) < 1e-9 and abs(tn / (tn + fp) - t["specificity"]) < 1e-9)
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
