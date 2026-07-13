"""
Part 3 - Ensembles, Tuning, and Full ML Pipeline
"""
import pandas as pd
import numpy as np
import json
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_DIR = "/home/claude/project/outputs/data"
MODEL_DIR = "/home/claude/project/outputs/models"
np.random.seed(42)
log = {}

arrs = np.load(f"{DATA_DIR}/part2_arrays.npz")
X_train_scaled = arrs["X_train_scaled"]
X_test_scaled = arrs["X_test_scaled"]
y_clf_train = arrs["y_clf_train"]
y_clf_test = arrs["y_clf_test"]

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
feature_names = pd.read_csv(f"{DATA_DIR}/feature_names.csv")["feature"].tolist()

print("Loaded arrays. X_train_scaled:", X_train_scaled.shape)

# ---------------------------------------------------------------------------
# Task 1: Decision Tree baseline (unconstrained)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 1: DECISION TREE BASELINE (unconstrained)")
print("=" * 80)
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train_scaled, y_clf_train)
train_acc_full = accuracy_score(y_clf_train, dt_full.predict(X_train_scaled))
test_acc_full = accuracy_score(y_clf_test, dt_full.predict(X_test_scaled))
print(f"Train acc: {train_acc_full:.4f}  Test acc: {test_acc_full:.4f}")
log["dt_full"] = dict(train_acc=train_acc_full, test_acc=test_acc_full)

# ---------------------------------------------------------------------------
# Task 2: Controlled Decision Tree
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: CONTROLLED DECISION TREE (max_depth=5, min_samples_split=20)")
print("=" * 80)
dt_ctrl = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_ctrl.fit(X_train_scaled, y_clf_train)
train_acc_ctrl = accuracy_score(y_clf_train, dt_ctrl.predict(X_train_scaled))
test_acc_ctrl = accuracy_score(y_clf_test, dt_ctrl.predict(X_test_scaled))
print(f"Train acc: {train_acc_ctrl:.4f}  Test acc: {test_acc_ctrl:.4f}")
log["dt_ctrl"] = dict(train_acc=train_acc_ctrl, test_acc=test_acc_ctrl)

# ---------------------------------------------------------------------------
# Task 3: Gini vs Entropy
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 3: GINI vs ENTROPY (max_depth=5)")
print("=" * 80)
dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
dt_gini.fit(X_train_scaled, y_clf_train)
acc_gini = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))

dt_entropy = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)
dt_entropy.fit(X_train_scaled, y_clf_train)
acc_entropy = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))

print(f"Gini test acc: {acc_gini:.4f}   Entropy test acc: {acc_entropy:.4f}")
log["gini_vs_entropy"] = dict(gini_acc=acc_gini, entropy_acc=acc_entropy)

# ---------------------------------------------------------------------------
# Task 4: Random Forest
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 4: RANDOM FOREST")
print("=" * 80)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_clf_train)
rf_train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
rf_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
print(f"Train acc: {rf_train_acc:.4f}  Test acc: {rf_test_acc:.4f}  AUC: {rf_auc:.4f}")

importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 5 features by importance:\n", importances.head(5))
log["rf"] = dict(train_acc=rf_train_acc, test_acc=rf_test_acc, auc=rf_auc,
                  top5_importance=importances.head(5).round(4).to_dict())

# ---------------------------------------------------------------------------
# Task 4a: Gradient Boosting
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 4a: GRADIENT BOOSTING")
print("=" * 80)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train_scaled, y_clf_train)
gb_train_acc = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
gb_auc = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])
print(f"Train acc: {gb_train_acc:.4f}  Test acc: {gb_test_acc:.4f}  AUC: {gb_auc:.4f}")
log["gb"] = dict(train_acc=gb_train_acc, test_acc=gb_test_acc, auc=gb_auc)

# ---------------------------------------------------------------------------
# Task 4b: Feature ablation study
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 4b: FEATURE ABLATION STUDY")
print("=" * 80)
lowest5 = importances.tail(5).index.tolist()
print("5 lowest-importance features:", lowest5)

keep_idx = [i for i, f in enumerate(feature_names) if f not in lowest5]
X_train_reduced = X_train_scaled[:, keep_idx]
X_test_reduced = X_test_scaled[:, keep_idx]

rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_reduced, y_clf_train)
auc_full = rf_auc
auc_reduced = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])
print(f"AUC full model: {auc_full:.4f}   AUC reduced model (5 feats removed): {auc_reduced:.4f}")
log["ablation"] = dict(removed_features=lowest5, auc_full=auc_full, auc_reduced=auc_reduced)

# ---------------------------------------------------------------------------
# Task 5: Cross-validated comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 5: CROSS-VALIDATED COMPARISON (5-fold StratifiedKFold, ROC-AUC)")
print("=" * 80)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_cv = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "DecisionTree(depth=5)": DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
}

cv_results = {}
for name, model in models_cv.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=skf, scoring="roc_auc")
    cv_results[name] = dict(mean_auc=scores.mean(), std_auc=scores.std())
    print(f"{name:25s} mean AUC={scores.mean():.4f}  std AUC={scores.std():.4f}")

log["cv_results"] = cv_results

# ---------------------------------------------------------------------------
# Task 6: GridSearchCV on Random Forest pipeline
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 6: GRIDSEARCHCV - RANDOM FOREST PIPELINE")
print("=" * 80)
pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          RandomForestClassifier(random_state=42))

param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}

grid = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                     scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_clf_train)  # unscaled inputs -- pipeline scales internally

print("Best params:", grid.best_params_)
print("Best CV score (AUC):", grid.best_score_)

n_configs = 1
for v in param_grid.values():
    n_configs *= len(v)
total_fits = n_configs * 5
print(f"Total grid configurations: {n_configs}  x 5 folds = {total_fits} fits")

log["grid_search"] = dict(best_params=grid.best_params_, best_score=grid.best_score_,
                           n_configs=n_configs, total_fits=total_fits)

best_pipeline = grid.best_estimator_

# ---------------------------------------------------------------------------
# Task 7: Manual learning curve using best pipeline
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 7: MANUAL LEARNING CURVE")
print("=" * 80)
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_rows = []
for f in fractions:
    n_rows = int(f * len(X_train))
    X_sub = X_train.iloc[:n_rows]
    y_sub = y_clf_train[:n_rows]

    pipe_f = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            RandomForestClassifier(random_state=42,
                                                    **{k.split("__")[1]: v for k, v in grid.best_params_.items()}))
    pipe_f.fit(X_sub, y_sub)

    train_auc = roc_auc_score(y_sub, pipe_f.predict_proba(X_sub)[:, 1])
    test_auc = roc_auc_score(y_clf_test, pipe_f.predict_proba(X_test)[:, 1])
    learning_rows.append(dict(fraction=f, n_rows=n_rows, train_auc=train_auc, test_auc=test_auc))
    print(f"Fraction={f:.1f}  n_rows={n_rows}  Train AUC={train_auc:.4f}  Test AUC={test_auc:.4f}")

log["learning_curve"] = learning_rows

# ---------------------------------------------------------------------------
# Task 8: Serialize best model
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 8: SERIALIZE BEST MODEL")
print("=" * 80)
joblib.dump(best_pipeline, f"{MODEL_DIR}/best_model.pkl")
print(f"Saved best pipeline -> {MODEL_DIR}/best_model.pkl")

# reload & predict demonstration
reloaded = joblib.load(f"{MODEL_DIR}/best_model.pkl")
sample_rows = X_test.iloc[:2]
sample_preds = reloaded.predict(sample_rows)
print("Reloaded model predictions on 2 hand-picked test rows:", sample_preds)

# ---------------------------------------------------------------------------
# Task 9: Summary comparison table
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 9: SUMMARY COMPARISON")
print("=" * 80)
test_auc_map = {
    "LogisticRegression": log["logreg_auc"] if "logreg_auc" in log else None,
    "DecisionTree(depth=5)": None,
    "RandomForest": rf_auc,
    "GradientBoosting": gb_auc,
}
# compute test AUC for logistic + controlled tree directly for completeness
logreg_test = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
logreg_test.fit(X_train_scaled, y_clf_train)
test_auc_map["LogisticRegression"] = roc_auc_score(y_clf_test, logreg_test.predict_proba(X_test_scaled)[:, 1])
test_auc_map["DecisionTree(depth=5)"] = roc_auc_score(y_clf_test, dt_ctrl.predict_proba(X_test_scaled)[:, 1])

summary_rows = []
for name in models_cv:
    summary_rows.append(dict(
        model=name,
        cv_mean_auc=cv_results[name]["mean_auc"],
        cv_std_auc=cv_results[name]["std_auc"],
        test_auc=test_auc_map[name],
    ))
summary_df = pd.DataFrame(summary_rows)
print(summary_df)
log["summary_table"] = summary_df.round(4).to_dict(orient="records")

with open(f"{DATA_DIR}/part3_log.json", "w") as f:
    json.dump(log, f, indent=2, default=str)

print("\nPART 3 COMPLETE.")
