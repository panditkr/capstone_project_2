"""
Part 2 - Supervised ML: Regression (GPA) + Classification (Burnout)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, confusion_matrix,
                              classification_report, roc_curve, roc_auc_score,
                              precision_score, recall_score, f1_score)

PLOT_DIR = "/home/claude/project/outputs/plots"
DATA_DIR = "/home/claude/project/outputs/data"
np.random.seed(42)

log = {}

df = pd.read_csv(f"{DATA_DIR}/cleaned_data.csv")
print("Loaded cleaned_data.csv, shape:", df.shape)

# ---------------------------------------------------------------------------
# Task 1: Define X, y_reg, y_clf
# ---------------------------------------------------------------------------
y_reg = df["gpa"].copy()
y_clf = df["feels_burned_out"].astype(int).copy()

X = df.drop(columns=["student_id", "gpa", "feels_burned_out"]).copy()
print("\nFeature matrix columns:", list(X.columns))
print("y_reg (gpa) description:\n", y_reg.describe())
print("\ny_clf (feels_burned_out) value counts:\n", y_clf.value_counts())

# ---------------------------------------------------------------------------
# Task 2: Encode categoricals
# ---------------------------------------------------------------------------
# education_level has a natural order: High School < Undergraduate < Graduate
edu_map = {"High School": 0, "Undergraduate": 1, "Graduate": 2}
X["education_level"] = X["education_level"].map(edu_map)

# uses_sleep_app is already boolean -> cast to int (binary, no encoding needed)
X["uses_sleep_app"] = X["uses_sleep_app"].astype(int)

# gender has no natural order -> one-hot encode, drop first to avoid multicollinearity
X = pd.get_dummies(X, columns=["gender"], drop_first=True)
bool_cols = X.select_dtypes(include=["bool"]).columns
X[bool_cols] = X[bool_cols].astype(int)

print("\nEncoded feature matrix (head):\n", X.head())
print("\nFinal feature columns:", list(X.columns))
feature_names = list(X.columns)
log["feature_names"] = feature_names

# ---------------------------------------------------------------------------
# Task 3: Leak-free split + scaling
# ---------------------------------------------------------------------------
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)  # fit ONLY on training data to avoid leakage
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------------------------------------------------------------------------
# Task 4: Regression - Linear Regression
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 4: LINEAR REGRESSION")
print("=" * 80)
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_reg_train)
y_pred_reg = lin_reg.predict(X_test_scaled)

mse_lin = mean_squared_error(y_reg_test, y_pred_reg)
r2_lin = r2_score(y_reg_test, y_pred_reg)
print(f"Linear Regression -> MSE: {mse_lin:.4f}  R2: {r2_lin:.4f}")

coef_df = pd.DataFrame({"feature": feature_names, "coefficient": lin_reg.coef_})
coef_df["abs_coef"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)
print("\nCoefficients (sorted by |coef|):\n", coef_df)
top3_feats = coef_df.head(3)

log["lin_reg"] = dict(mse=mse_lin, r2=r2_lin,
                       coefficients=coef_df.set_index("feature")["coefficient"].round(4).to_dict(),
                       top3=top3_feats["feature"].tolist())

# Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
print(f"\nRidge Regression (alpha=1.0) -> MSE: {mse_ridge:.4f}  R2: {r2_ridge:.4f}")
log["ridge_reg"] = dict(mse=mse_ridge, r2=r2_ridge)

# ---------------------------------------------------------------------------
# Task 5: Classification - Logistic Regression
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 5: LOGISTIC REGRESSION")
print("=" * 80)
vc = y_clf_train.value_counts()
print("Train class counts BEFORE balancing:\n", vc)
minority_frac = vc.min() / vc.sum()
print(f"Minority class fraction: {minority_frac:.3f}")

use_balanced = minority_frac < 0.35
print("Class imbalance handling: class_weight='balanced'" if use_balanced else
      "Classes sufficiently balanced; class_weight left as default.")

clf_kwargs = dict(max_iter=1000, random_state=42)
if use_balanced:
    clf_kwargs["class_weight"] = "balanced"

log_reg = LogisticRegression(**clf_kwargs)
log_reg.fit(X_train_scaled, y_clf_train)

y_pred_clf = log_reg.predict(X_test_scaled)
y_proba_clf = log_reg.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_clf_test, y_pred_clf)
report = classification_report(y_clf_test, y_pred_clf, output_dict=True)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_clf_test, y_pred_clf))

fpr, tpr, thresholds = roc_curve(y_clf_test, y_proba_clf)
auc = roc_auc_score(y_clf_test, y_proba_clf)
print(f"\nAUC: {auc:.4f}")

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#1d3557", linewidth=2, label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression (Burnout Classifier)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_roc_curve.png", dpi=130)
plt.close()

log["logreg"] = dict(class_counts_before=vc.to_dict(), use_balanced=use_balanced,
                      confusion_matrix=cm.tolist(), auc=float(auc),
                      accuracy=report["accuracy"], precision_1=report["1"]["precision"],
                      recall_1=report["1"]["recall"], f1_1=report["1"]["f1-score"])

# ---------------------------------------------------------------------------
# Task 5b: Decision-threshold sensitivity
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 5b: THRESHOLD SENSITIVITY")
print("=" * 80)
thresh_rows = []
for t in [0.30, 0.40, 0.50, 0.60, 0.70]:
    preds_t = (y_proba_clf >= t).astype(int)
    p = precision_score(y_clf_test, preds_t, zero_division=0)
    r = recall_score(y_clf_test, preds_t, zero_division=0)
    f1 = f1_score(y_clf_test, preds_t, zero_division=0)
    thresh_rows.append(dict(threshold=t, precision=p, recall=r, f1=f1))
    print(f"Threshold={t:.2f} | Precision={p:.4f} | Recall={r:.4f} | F1={f1:.4f}")

best_thresh_row = max(thresh_rows, key=lambda r: r["f1"])
print("\nF1-maximizing threshold:", best_thresh_row)
log["threshold_table"] = thresh_rows
log["best_f1_threshold"] = best_thresh_row

# ---------------------------------------------------------------------------
# Task 6: Regularization experiment (C=0.01 vs C=1.0)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 6: REGULARIZATION EXPERIMENT")
print("=" * 80)
clf_kwargs_strong = dict(max_iter=1000, random_state=42, C=0.01)
if use_balanced:
    clf_kwargs_strong["class_weight"] = "balanced"
log_reg_strong = LogisticRegression(**clf_kwargs_strong)
log_reg_strong.fit(X_train_scaled, y_clf_train)
y_pred_strong = log_reg_strong.predict(X_test_scaled)
y_proba_strong = log_reg_strong.predict_proba(X_test_scaled)[:, 1]

p_strong = precision_score(y_clf_test, y_pred_strong, zero_division=0)
r_strong = recall_score(y_clf_test, y_pred_strong, zero_division=0)
auc_strong = roc_auc_score(y_clf_test, y_proba_strong)

p_base = report["1"]["precision"]
r_base = report["1"]["recall"]
auc_base = auc

print("C=1.0 (baseline)  -> Precision: {:.4f}  Recall: {:.4f}  AUC: {:.4f}".format(p_base, r_base, auc_base))
print("C=0.01 (strong reg) -> Precision: {:.4f}  Recall: {:.4f}  AUC: {:.4f}".format(p_strong, r_strong, auc_strong))

log["c_comparison"] = dict(baseline=dict(C=1.0, precision=p_base, recall=r_base, auc=auc_base),
                            strong_reg=dict(C=0.01, precision=p_strong, recall=r_strong, auc=auc_strong))

# ---------------------------------------------------------------------------
# Task 6b: Bootstrap CI for AUC difference
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 6b: BOOTSTRAP CI FOR AUC DIFFERENCE")
print("=" * 80)
n_boot = 500
y_clf_test_arr = y_clf_test.reset_index(drop=True).values
diffs = []
n = len(y_clf_test_arr)
for i in range(n_boot):
    idx = np.random.choice(n, size=n, replace=True)
    y_sample = y_clf_test_arr[idx]
    # need at least both classes present to compute AUC
    if len(np.unique(y_sample)) < 2:
        continue
    proba_base_sample = y_proba_clf[idx]
    proba_strong_sample = y_proba_strong[idx]
    auc_base_s = roc_auc_score(y_sample, proba_base_sample)
    auc_strong_s = roc_auc_score(y_sample, proba_strong_sample)
    diffs.append(auc_base_s - auc_strong_s)

diffs = np.array(diffs)
mean_diff = diffs.mean()
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
print(f"Bootstrap iterations used: {len(diffs)}")
print(f"Mean AUC difference (C=1.0 - C=0.01): {mean_diff:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
excludes_zero = (ci_low > 0) or (ci_high < 0)
print("95% CI excludes zero:", excludes_zero)

log["bootstrap_auc_diff"] = dict(mean_diff=float(mean_diff), ci_low=float(ci_low),
                                  ci_high=float(ci_high), excludes_zero=bool(excludes_zero),
                                  n_valid=int(len(diffs)))

# ---------------------------------------------------------------------------
# Save artifacts needed downstream
# ---------------------------------------------------------------------------
np.savez(f"{DATA_DIR}/part2_arrays.npz",
         X_train=X_train.values, X_test=X_test.values,
         X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,
         y_reg_train=y_reg_train.values, y_reg_test=y_reg_test.values,
         y_clf_train=y_clf_train.values, y_clf_test=y_clf_test.values)

X_train.to_csv(f"{DATA_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{DATA_DIR}/X_test.csv", index=False)
pd.Series(feature_names).to_csv(f"{DATA_DIR}/feature_names.csv", index=False, header=["feature"])

with open(f"{DATA_DIR}/part2_log.json", "w") as f:
    json.dump(log, f, indent=2, default=str)

print("\nPART 2 COMPLETE.")
