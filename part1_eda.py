"""
Part 1 - Data Acquisition, Cleaning, and Exploratory Analysis
Dataset: Student Sleep & Mental Health 2026
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style("whitegrid")
PLOT_DIR = "/home/claude/project/outputs/plots"
DATA_DIR = "/home/claude/project/outputs/data"

log = {}  # collects results to dump as JSON for README generation

# ---------------------------------------------------------------------------
# Task 1: Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("/home/claude/project/data/student_sleep_mental_health_2026.csv")

print("=" * 80)
print("TASK 1: LOAD DATA")
print("=" * 80)
print("\nFirst 5 rows:\n", df.head())
print("\nDtypes:\n", df.dtypes)
print("\nShape:", df.shape)

log["shape"] = list(df.shape)
log["dtypes_raw"] = {c: str(t) for c, t in df.dtypes.items()}

# ---------------------------------------------------------------------------
# Task 2: Null value analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: NULL VALUE ANALYSIS")
print("=" * 80)
null_counts = df.isnull().sum()
null_pct = (null_counts / df.shape[0]) * 100
null_table = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct.round(3)})
print(null_table)

cols_over_20 = null_table[null_table["null_pct"] > 20].index.tolist()
print("\nColumns exceeding 20% null rate:", cols_over_20 if cols_over_20 else "NONE")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if col not in cols_over_20 and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("\nNulls remaining after median-fill pass:\n", df.isnull().sum().sum(), "total nulls")
log["null_table"] = null_table.to_dict(orient="index")
log["cols_over_20pct_null"] = cols_over_20

# ---------------------------------------------------------------------------
# Task 3: Duplicate detection and removal
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 3: DUPLICATE DETECTION")
print("=" * 80)
n_dupes = df.duplicated().sum()
print("Duplicate rows found:", n_dupes)
rows_before = df.shape[0]
null_pct_before_dedup = (df.isnull().sum() / df.shape[0]) * 100
df = df.drop_duplicates()
rows_after = df.shape[0]
null_pct_after_dedup = (df.isnull().sum() / df.shape[0]) * 100
print(f"Rows removed: {rows_before - rows_after}")
print("Did null percentages change after de-dup?",
      not null_pct_before_dedup.equals(null_pct_after_dedup))

log["n_duplicates"] = int(n_dupes)
log["rows_removed_dupes"] = int(rows_before - rows_after)

# ---------------------------------------------------------------------------
# Task 4: Data type correction
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 4: DATA TYPE CORRECTION")
print("=" * 80)
mem_before = df.memory_usage(deep=True).sum()
print("Memory usage BEFORE conversion (bytes):", mem_before)

# No numeric column was mis-typed as object on ingest (verified in Task 1 dtypes).
# The genuine dtype inefficiency in this dataset is the repetitive string
# columns (gender, education_level) stored as generic string/object dtype,
# and the boolean flags, which all benefit from conversion to 'category'.
df["gender"] = df["gender"].astype("category")
df["education_level"] = df["education_level"].astype("category")
df["uses_sleep_app"] = df["uses_sleep_app"].astype("category")
df["feels_burned_out"] = df["feels_burned_out"].astype("category")

# Demonstrate the numeric-coercion pattern requested by the task (defensive
# coding in case future data drops arrive with numbers-as-strings):
df["caffeine_drinks_per_day"] = pd.to_numeric(df["caffeine_drinks_per_day"], errors="coerce")

mem_after = df.memory_usage(deep=True).sum()
print("Memory usage AFTER conversion (bytes):", mem_after)
print(f"Memory reduction: {mem_before - mem_after} bytes "
      f"({(1 - mem_after/mem_before)*100:.2f}%)")

log["mem_before"] = int(mem_before)
log["mem_after"] = int(mem_after)

# ---------------------------------------------------------------------------
# Task 5: Descriptive statistics and skewness
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 5: DESCRIPTIVE STATS & SKEWNESS")
print("=" * 80)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "student_id"]
print(df[numeric_cols].describe())

skew_vals = df[numeric_cols].skew().sort_values(key=lambda s: s.abs(), ascending=False)
print("\nSkewness (sorted by |skew|):\n", skew_vals)

top_skew_col = skew_vals.index[0]
print(f"\nMost skewed column: {top_skew_col} (skew={skew_vals.iloc[0]:.3f})")

log["skewness"] = skew_vals.round(4).to_dict()
log["top_skew_col"] = top_skew_col
top2_skew_cols = skew_vals.index[:2].tolist()
log["top2_skew_cols"] = top2_skew_cols

# ---------------------------------------------------------------------------
# Task 6: Outlier detection with IQR
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 6: OUTLIER DETECTION (IQR)")
print("=" * 80)
iqr_cols = ["anxiety_score", "gpa"]
iqr_report = {}
for col in iqr_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: Q1={Q1:.3f} Q3={Q3:.3f} IQR={IQR:.3f} "
          f"lower={lower:.3f} upper={upper:.3f} outliers={n_outliers}")
    iqr_report[col] = dict(Q1=Q1, Q3=Q3, IQR=IQR, lower=lower, upper=upper,
                            n_outliers=int(n_outliers))
log["iqr_report"] = iqr_report

# ---------------------------------------------------------------------------
# Task 7: Visualizations
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 7: VISUALIZATIONS")
print("=" * 80)

# 7.1 Line plot
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["anxiety_score"].values, linewidth=0.7, color="#3b5bdb")
plt.title("Anxiety Score by Row Index")
plt.xlabel("Row Index")
plt.ylabel("Anxiety Score")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_line_anxiety.png", dpi=130)
plt.close()

# 7.2 Bar chart: mean GPA by education level
plt.figure(figsize=(7, 5))
means = df.groupby("education_level", observed=True)["gpa"].mean().sort_values()
plt.bar(means.index.astype(str), means.values, color="#4c956c")
plt.title("Mean GPA by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Mean GPA")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_bar_gpa_by_education.png", dpi=130)
plt.close()
log["mean_gpa_by_education"] = means.round(3).to_dict()

# 7.3 Histogram of most skewed column
plt.figure(figsize=(7, 5))
sns.histplot(df[top_skew_col], bins=20, kde=True, color="#e07a5f")
plt.title(f"Distribution of {top_skew_col} (most skewed, skew={skew_vals.iloc[0]:.2f})")
plt.xlabel(top_skew_col)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_hist_{top_skew_col}.png", dpi=130)
plt.close()

# 7.4 Scatter plot: stress_level vs anxiety_score (expected correlation)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="stress_level", y="anxiety_score", alpha=0.4, color="#5f0f40")
plt.title("Stress Level vs Anxiety Score")
plt.xlabel("Stress Level")
plt.ylabel("Anxiety Score")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_scatter_stress_anxiety.png", dpi=130)
plt.close()
log["corr_stress_anxiety_pearson"] = float(df["stress_level"].corr(df["anxiety_score"]))

# 7.5 Box plot: avg_sleep_hours split by feels_burned_out
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="feels_burned_out", y="avg_sleep_hours", palette="Set2")
plt.title("Average Sleep Hours by Burnout Status")
plt.xlabel("Feels Burned Out")
plt.ylabel("Average Sleep Hours")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_box_sleep_by_burnout.png", dpi=130)
plt.close()
log["sleep_by_burnout_medians"] = df.groupby("feels_burned_out", observed=True)["avg_sleep_hours"].median().round(3).to_dict()

# 7.6 Correlation heat map
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heat Map (Numeric Features)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_heatmap_correlation.png", dpi=130)
plt.close()

corr_unstack = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).abs().unstack().sort_values(ascending=False)
top_pair = corr_unstack.index[0]
print("\nHighest |correlation| pair:", top_pair, "=", corr_unstack.iloc[0])
log["pearson_corr_matrix"] = corr_matrix.round(4).to_dict()
log["top_corr_pair"] = list(top_pair)
log["top_corr_value"] = float(corr_unstack.iloc[0])

# ---------------------------------------------------------------------------
# Task 8a: Imputation strategy comparison (mean vs median) on top-2 skew cols
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 8a: IMPUTATION STRATEGY COMPARISON")
print("=" * 80)
imputation_report = {}
for col in top2_skew_cols:
    mean_v = df[col].mean()
    median_v = df[col].median()
    print(f"{col}: mean={mean_v:.4f} | median={median_v:.4f} | skew={df[col].skew():.4f}")
    imputation_report[col] = dict(mean=mean_v, median=median_v, skew=df[col].skew())
    # apply chosen strategy (median, robust to skew) to any remaining nulls
    df[col] = df[col].fillna(median_v)

print("\nNulls remaining in these columns:\n", df[top2_skew_cols].isnull().sum())
log["imputation_report"] = imputation_report

# ---------------------------------------------------------------------------
# Task 8b: Spearman vs Pearson
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 8b: SPEARMAN vs PEARSON CORRELATION")
print("=" * 80)
spearman_matrix = df[numeric_cols].corr(method="spearman")
print("Pearson matrix:\n", corr_matrix.round(3))
print("\nSpearman matrix:\n", spearman_matrix.round(3))

diff_matrix = (spearman_matrix - corr_matrix).abs()
diff_pairs = diff_matrix.where(~np.eye(len(diff_matrix), dtype=bool)).unstack().sort_values(ascending=False)
diff_pairs = diff_pairs[~diff_pairs.index.duplicated()]
# dedupe symmetric pairs (a,b) vs (b,a)
seen = set()
top3 = []
for (a, b), v in diff_pairs.items():
    key = frozenset([a, b])
    if key in seen or a == b:
        continue
    seen.add(key)
    top3.append((a, b, v, corr_matrix.loc[a, b], spearman_matrix.loc[a, b]))
    if len(top3) == 3:
        break

print("\nTop 3 pairs by |Spearman - Pearson|:")
diff_table_rows = []
for a, b, v, pear, spear in top3:
    print(f"  {a} vs {b}: Pearson={pear:.4f} Spearman={spear:.4f} |diff|={v:.4f}")
    diff_table_rows.append(dict(col_a=a, col_b=b, pearson=pear, spearman=spear, abs_diff=v))
log["spearman_pearson_diff_top3"] = diff_table_rows

# ---------------------------------------------------------------------------
# Task 8c: Grouped aggregation
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 8c: GROUPED AGGREGATION")
print("=" * 80)
group_agg = df.groupby("education_level", observed=True)["gpa"].agg(["mean", "std", "count"])
print(group_agg)

highest_mean_grp = group_agg["mean"].idxmax()
highest_std_grp = group_agg["std"].idxmax()
mean_ratio = group_agg["mean"].max() / group_agg["mean"].min()
print(f"\nHighest mean group: {highest_mean_grp}")
print(f"Highest std group: {highest_std_grp}")
print(f"Ratio of highest to lowest group mean: {mean_ratio:.4f}")

log["group_agg"] = group_agg.round(4).to_dict(orient="index")
log["highest_mean_grp"] = str(highest_mean_grp)
log["highest_std_grp"] = str(highest_std_grp)
log["mean_ratio"] = float(mean_ratio)

# ---------------------------------------------------------------------------
# Save cleaned dataset
# ---------------------------------------------------------------------------
df_out = df.copy()
# store category/bool columns back as plain types for portability in CSV
for c in ["gender", "education_level"]:
    df_out[c] = df_out[c].astype(str)
for c in ["uses_sleep_app", "feels_burned_out"]:
    df_out[c] = df_out[c].astype(bool)

df_out.to_csv(f"{DATA_DIR}/cleaned_data.csv", index=False)
print(f"\nSaved cleaned dataset -> {DATA_DIR}/cleaned_data.csv  shape={df_out.shape}")

with open(f"{DATA_DIR}/part1_log.json", "w") as f:
    json.dump(log, f, indent=2, default=str)

print("\nPART 1 COMPLETE.")
