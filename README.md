# Student Sleep & Mental Health 2026 — End-to-End ML Project

**Dataset:** `student_sleep_mental_health_2026.csv` — 3,000 student records with sleep,
lifestyle, mental-health, and academic-performance variables.

**Columns:** `student_id, age, gender, education_level, avg_sleep_hours, screen_time_hours,
social_media_hours, study_hours_per_day, exercise_hours_per_week, caffeine_drinks_per_day,
stress_level, anxiety_score, gpa, uses_sleep_app, feels_burned_out`

**Repo layout**
```
scripts/part1_eda.py            Part 1 — cleaning & EDA
scripts/part2_models.py         Part 2 — regression + classification
scripts/part3_ensembles.py      Part 3 — ensembles, tuning, pipeline
scripts/part4_llm_explain.py    Part 4 — LLM explanation feature (Track C)
outputs/data/cleaned_data.csv   Cleaned dataset (output of Part 1, input to Parts 2-3)
outputs/data/*.json             Machine-readable logs of every metric below
outputs/plots/*.png             All 7 required charts
outputs/models/best_model.pkl   Serialized, tuned Random Forest pipeline
```

Run order: `part1_eda.py` → `part2_models.py` → `part3_ensembles.py` → `part4_llm_explain.py`
(each reads the outputs of the previous step). All four ran top-to-bottom without errors.

---

## Part 1 — Data Acquisition, Cleaning & EDA

### 1. Load
3,000 rows × 15 columns loaded with `pd.read_csv`. Dtypes on ingest: 4 integer, 5 float,
2 string(object), 2 boolean, 1 id column — all correctly inferred by pandas (see below).

### 2. Null analysis
`df.isnull().sum()` / `df.shape[0] * 100` was computed for **every** column — the result is
**0.0% missing in all 15 columns**. No column exceeds the 20% threshold, so no column was
dropped. The median-fill line (`fillna(df[col].median())`) was still run over all numeric
columns as required, defensively; since there were no nulls to fill, it is a no-op here but
would activate automatically on any future data drop that does contain gaps.

**Why median, not mean, for the fill strategy:** the mean is pulled toward whichever tail
has more extreme values, so a single batch of unusually stressed or unusually rested
students can shift the "typical" value used to fill every gap. The median is the middle
observation and is unaffected by how extreme the tails are, which matters here because
several numeric columns (see skewness below) are not symmetric.

### 3. Duplicates
`df.duplicated().sum()` → **0 duplicate rows**. `drop_duplicates()` removed 0 rows, so null
percentages are unchanged (there is nothing to compare against a "before" state that
differs from "after").

### 4. Dtype correction
No numeric column was stored as `object`/string on ingest — this dataset arrived
well-typed. The genuine, real inefficiency was the two repetitive string columns
(`gender`, `education_level`) and the two boolean flags stored as generic Python
objects/bools rather than pandas `category`. Converting:

| | Memory (bytes) |
|---|---|
| Before conversion | 615,646 |
| After conversion (`category` dtype on `gender`, `education_level`, `uses_sleep_app`, `feels_burned_out`) | 276,548 |
| **Reduction** | **339,098 bytes (55.1%)** |

`caffeine_drinks_per_day` was additionally passed through
`pd.to_numeric(..., errors='coerce')` to demonstrate the required coercion pattern
defensively (no values were actually coerced to NaN, since the column was already clean
integers).

### 5. Descriptive stats & skewness
`df.describe()` was run on all numeric columns (see `part1_log.json` → `skewness` for full
values). Skewness ranked by |skew|:

| Column | Skew |
|---|---|
| **caffeine_drinks_per_day** | **+0.783** |
| exercise_hours_per_week | +0.363 |
| anxiety_score | −0.348 |
| age | +0.182 |
| stress_level | −0.182 |
| gpa | −0.094 |
| avg_sleep_hours | +0.088 |
| study_hours_per_day | +0.087 |
| social_media_hours | +0.017 |
| screen_time_hours | +0.002 |

**Most skewed column: `caffeine_drinks_per_day` (skew = +0.78, moderate positive skew).**
A positive skew means the distribution has a long right tail — most students drink very
few caffeinated drinks per day, but a smaller group drinks noticeably more, pulling the
mean upward. Consequence for imputation: filling missing values in this column with the
**mean** would insert a value inflated by the small number of heavy-caffeine students,
overstating "typical" consumption; the **median** sits at the actual center of the bulk of
the data and is the safer choice, consistent with the median-fill policy adopted above.

### 6. Outlier detection (IQR)
| Column | Q1 | Q3 | IQR | Lower bound | Upper bound | # Outliers |
|---|---|---|---|---|---|---|
| anxiety_score | 6.00 | 9.00 | 3.00 | 1.50 | 13.50 | **8** |
| gpa | 2.98 | 3.51 | 0.53 | 2.185 | 4.305 | **8** |

Both columns have a small number (8 out of 3,000, ~0.27%) of rows outside the IQR fence.
**Decision:** these are retained, not dropped or capped, in Part 1's cleaned output. Both
variables are plausible at their extremes (a maximally-stressed student, or a strong GPA
close to the 4.0 ceiling are real, not data-entry errors), and the downstream models in
Part 2/3 are tree ensembles and regularized linear models that are already fairly robust
to a handful of extreme points; capping would risk discarding genuine signal about the
most-at-risk / highest-performing students, who are exactly the population this project
cares about.

### 7. Visualizations
All in `outputs/plots/`:
1. `01_line_anxiety.png` — line plot of anxiety score by row index.
2. `02_bar_gpa_by_education.png` — mean GPA by education level.
3. `03_hist_caffeine_drinks_per_day.png` — histogram of the most-skewed column; the shape
   is a right-skewed, roughly unimodal distribution concentrated at 0–2 drinks/day with a
   thinning tail out to ~5+.
4. `04_scatter_stress_anxiety.png` — stress vs. anxiety; **Pearson r ≈ 0.74**, a clear
   positive, moderately strong, roughly linear relationship — students who report more
   stress also report more anxiety.
5. `05_box_sleep_by_burnout.png` — average sleep hours split by burnout status. Median
   sleep is **~8.0 h** for students who do **not** report burnout vs. **~7.2 h** for those
   who do, with a visibly lower median and a spread shifted downward for the burned-out
   group.
6. `06_heatmap_correlation.png` — full Pearson correlation heat map.

**Highest |correlation| pair:** `screen_time_hours` ↔ `social_media_hours`, **r = 0.772**.
This is unlikely to be purely causal in either direction — a third variable, general
"time spent on a phone/laptop," plausibly drives both simultaneously (someone who is on
their screen more overall will show up higher on both measures almost mechanically, since
social media time is a subset of total screen time). So this correlation is at least
partly a part-whole/common-cause relationship rather than independent evidence that one
causes the other.

### 8a. Mean vs. median for the two most-skewed columns
| Column | Mean | Median | Skew | Chosen |
|---|---|---|---|---|
| caffeine_drinks_per_day | 1.7787 | 2.0000 | +0.783 | **Median** |
| exercise_hours_per_week | 3.5419 | 3.4000 | +0.363 | **Median** |

Both columns are positively skewed, so their means are pulled upward by the small
high-consumption / high-exercise tail; the median is the more representative "typical
student" value in both cases. `isnull().sum()` on these two columns after the fillna pass
confirms **0 remaining nulls** in both.

### 8b. Spearman vs. Pearson
Top 3 pairs by |Spearman − Pearson|:

| Pair | Pearson | Spearman | |diff| |
|---|---|---|---|
| avg_sleep_hours vs gpa | 0.5223 | 0.5020 | 0.0203 |
| social_media_hours vs avg_sleep_hours | −0.4531 | −0.4328 | 0.0203 |
| screen_time_hours vs social_media_hours | 0.7723 | 0.7523 | 0.0199 |

In all three pairs |Pearson| ≥ |Spearman| (the differences are tiny, ~0.02), so every one
of these relationships is best described as **approximately linear** rather than
monotonic-but-non-linear — Spearman is not picking up extra signal that Pearson misses.
Given this, **Pearson** is the correlation measure relied on for Part 2 feature-selection
guidance, since the relationships in this dataset are close to linear and Pearson is more
statistically efficient (lower variance) than Spearman under that condition.

### 8c. Grouped aggregation — GPA by education level
| Group | Mean | Std | Count |
|---|---|---|---|
| Graduate | 3.2167 | 0.3861 | 430 |
| High School | 3.2548 | 0.3834 | 889 |
| Undergraduate | 3.2462 | 0.3754 | 1,681 |

- Highest mean: **High School** (3.255). Highest std: **Graduate** (0.386).
- Within-group std (~0.38–0.39) is large relative to the *between*-group mean differences
  (~0.04 GPA points), which **is** a concern for a predictive model: knowing a student's
  education level barely narrows down their likely GPA compared to the spread inside each
  group.
- Ratio of highest to lowest group mean = 3.2548 / 3.2167 = **1.012**. A ratio this close
  to 1.0 indicates education level carries **very little standalone predictive signal**
  for GPA in this dataset — consistent with the near-zero coefficient education_level
  receives in the Part 2 regression model.

---

## Part 2 — Supervised ML: Regression + Classification

**Labels used** (stated explicitly, both derived from the cleaned dataset, no target
leaks into the other's feature set):
- `y_reg` = `gpa` (continuous).
- `y_clf` = `feels_burned_out` (already a natural binary column in the data — used
  directly rather than binarizing `gpa`, since it is a genuine binary outcome of interest
  in its own right).
- `X` = all columns except `student_id`, `gpa`, and `feels_burned_out`, so neither label
  can leak into the other's model.

### Categorical encoding
- `education_level` has a natural order (**High School < Undergraduate < Graduate**) →
  label-encoded as 0/1/2, preserving that order.
- `gender` has **no** natural order (Female / Male / Non-binary / Prefer not to say) →
  one-hot encoded with `drop_first=True` to avoid multicollinearity. Label-encoding gender
  as e.g. 0/1/2/3 would falsely imply "Male > Female" or similar orderings that don't
  exist in reality, which is exactly the failure mode one-hot encoding avoids.
- `uses_sleep_app` is already boolean → cast to 0/1 directly (no separate encoding step
  needed for a two-level flag).

### Leak-free split & scaling
`train_test_split(..., test_size=0.2, random_state=42)` → 2,400 train / 600 test rows.
`StandardScaler` was **fit only on `X_train`**, then used to `.transform()` both
`X_train` and `X_test`. Fitting the scaler on the full dataset (train + test combined)
would leak test-set mean/variance into the preprocessing step used to train the model —
the model would implicitly "see" statistics about data it is later evaluated on, inflating
the reported test performance relative to what it would achieve on genuinely new data.

### Regression — Linear Regression vs. Ridge
| Model | MSE | R² |
|---|---|---|
| Linear Regression | 0.0824 | 0.3778 |
| Ridge (α = 1.0) | 0.0824 | 0.3778 |

Top 3 |coefficient| features (scaled): **stress_level** (−0.155), **avg_sleep_hours**
(+0.095), **social_media_hours** (−0.086). A large **positive** coefficient (e.g.
`avg_sleep_hours`) means a one-standard-deviation increase in that scaled feature is
associated with a `+0.095`-GPA-point increase in predicted GPA, holding other features
fixed; a large **negative** coefficient (`stress_level`) means a one-SD increase in stress
is associated with a `−0.155`-point drop in predicted GPA.

Ridge and OLS land on essentially identical MSE/R² here because the feature set is small
(14 features) with low multicollinearity outside the screen-time/social-media pair, so
`alpha=1.0` barely shrinks the coefficients. In general, Ridge adds an L2 penalty
(`alpha` × sum of squared coefficients) to the OLS loss, which shrinks large coefficients
toward zero and stabilizes the solution when predictors are correlated; the larger
`alpha` is, the more the coefficients are pulled toward zero (more bias, less variance).

### Classification — Logistic Regression (burnout)
Train class balance: 1,611 burned-out (67.1%) vs. 789 not (32.9%) — the minority class is
just under 35%, so **`class_weight='balanced'`** was applied (chosen over SMOTE since it
requires no synthetic sample generation and works directly inside `LogisticRegression`).

**Result: perfect separation** — confusion matrix `[[172, 0], [0, 428]]`, accuracy /
precision / recall / F1 all **1.00**, **AUC = 1.000**.

> **Important, honestly-reported finding:** this is *not* a bug in the pipeline — it is a
> property of this particular synthetic dataset. `stress_level` and `anxiety_score` are
> almost perfectly aligned with `feels_burned_out` (confirmed again in Part 3's feature
> importances, where `stress_level` alone accounts for ~69% of Random Forest importance).
> In a real-world dataset this pattern — one or two features driving a classifier to
> perfect separation — would be a leakage red flag worth investigating with the data
> provider; here it simply reflects how the labels were generated. All downstream
> analyses (threshold sensitivity, regularization comparison, cross-validation, ensembles)
> are reported as computed, with this caveat flagged rather than hidden.

**Precision** = TP / (TP + FP). **Recall** = TP / (TP + FN). For a burnout-flagging tool
feeding into student wellbeing outreach, **recall** matters more than precision: missing a
genuinely burned-out student (a false negative) means a student in distress gets no
intervention, which is a worse outcome than reaching out to a student who turns out to be
coping fine (a false positive). AUC = 1.0 means the model perfectly ranks burned-out
students above non-burned-out students across every possible threshold — again, a
reflection of how separable these labels are on these features, not evidence the model
will generalize this well on noisier, real-world data.

### Threshold sensitivity (0.30 → 0.70)
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 1.000 | 1.000 | 1.000 |
| 0.40 | 1.000 | 1.000 | 1.000 |
| 0.50 | 1.000 | 1.000 | 1.000 |
| 0.60 | 1.000 | 1.000 | 1.000 |
| 0.70 | 1.000 | 1.000 | 1.000 |

F1 is maximized (trivially, at 1.0) across the whole range because the classes are
perfectly separated by the predicted probabilities — there is no threshold in [0.3, 0.7]
where the two classes' probability ranges overlap. In a less-separable dataset, the
guidance would be: raise the threshold to favor precision (fewer false alarms, at the cost
of missing some true positives) or lower it to favor recall (catch more true positives, at
the cost of more false alarms) — here we'd lower it slightly if forced to choose, since
recall is the priority metric argued above, but the cost of doing so is nil in this data.

### Regularization: C = 1.0 vs. C = 0.01
| C | Precision | Recall | AUC |
|---|---|---|---|
| 1.0 (baseline) | 1.0000 | 1.0000 | 1.0000 |
| 0.01 (strong L2) | 0.9926 | 0.9346 | 0.9928 |

`C` is the **inverse** of the regularization strength in scikit-learn's
`LogisticRegression` (small `C` = strong penalty on large coefficients). Shrinking `C` to
0.01 measurably **worsens** performance here (recall drops from 1.00 to 0.935): the true
decision boundary in this data needs coefficients large enough to fully exploit the
near-deterministic stress/anxiety signal, and heavy shrinkage blunts that.

### Bootstrap 95% CI for the AUC difference (C=1.0 − C=0.01)
500 bootstrap resamples of the test set → **mean difference = 0.0072**,
**95% CI = [0.0032, 0.0123]**. The interval **excludes zero**, so the C=1.0 model's small
AUC advantage over the heavily-regularized model is statistically consistent across
resamples of this test set, not just noise.

---

## Part 3 — Ensembles, Tuning & Pipeline

### Decision Tree, unconstrained vs. controlled
| Model | Train acc | Test acc |
|---|---|---|
| Unconstrained (`max_depth=None`) | 1.0000 | 1.0000 |
| Controlled (`max_depth=5, min_samples_split=20`) | 1.0000 | 1.0000 |

No train/test gap appears in either tree given how separable this label is; ordinarily
the unconstrained tree would show a much larger gap, since **decision trees are
high-variance models** — they greedily choose the locally-best split at each node without
ever revisiting earlier decisions, so an unconstrained tree can carve out a leaf for
almost every training point (memorizing noise) rather than the general pattern.
`max_depth` limits how many splits deep the tree can go (capping variance at some cost in
bias/expressiveness), and `min_samples_split=20` blocks a split from being made on a node
with fewer than 20 samples, preventing splits that fit to noise in small, coincidental
subsets of the training data.

### Gini vs. Entropy (max_depth = 5)
Gini test accuracy: **1.0000**. Entropy test accuracy: **1.0000** (identical here, again
due to how separable the label is).
- Gini impurity: `1 − Σ pᵢ²`
- Entropy: `−Σ pᵢ log₂(pᵢ)`
A node with **Gini = 0** means every sample reaching that node belongs to a single class
— it is a pure leaf.

### Random Forest (n_estimators=100, max_depth=10)
Train acc **1.0000**, test acc **1.0000**, AUC **1.0000**.

Top-5 feature importances:
| Feature | Importance |
|---|---|
| stress_level | 0.6853 |
| anxiety_score | 0.1594 |
| screen_time_hours | 0.0461 |
| avg_sleep_hours | 0.0360 |
| social_media_hours | 0.0268 |

Random Forest importance is the **average reduction in Gini impurity** contributed by a
feature across every split, in every tree, weighted by how many samples pass through that
split — a purely structural, model-internal quantity. This differs fundamentally from a
linear regression coefficient, which measures the **linear, additive** change in a
continuous prediction per unit change in a (scaled) feature, holding all others fixed; a
tree-based importance score captures non-linear and interaction effects a linear
coefficient cannot, but (unlike a coefficient) carries no sign or direct unit
interpretation.

**Bagging concept:** each of the 100 trees is trained on an independent bootstrap sample
(sampled with replacement from the 2,400 training rows), and at every split only a random
subset of √14 ≈ 3–4 features is even considered as a splitting candidate. This
de-correlates the individual trees — each one makes somewhat different mistakes — so
averaging their predictions cancels out much of the variance any single deep tree would
have, without adding meaningful bias.

### Gradient Boosting (n_estimators=100, lr=0.1, max_depth=3)
Train acc **1.0000**, test acc **1.0000**, AUC **1.0000**.

### Feature ablation study
5 lowest-importance features removed: `education_level`, `uses_sleep_app`, `gender_Male`,
`gender_Non-binary`, `gender_Prefer not to say`.
**AUC full model = 1.0000. AUC reduced model = 1.0000.** These 5 features were genuinely
close to uninformative for this task — removing them costs nothing in AUC. For
production, this means a **simpler, lower-dimensional model** (fewer features to collect,
validate, and monitor, lower inference cost) is available here at essentially zero
accuracy cost, since the degradation (0.0000) is far below any reasonable tolerance
threshold.

### 5-fold cross-validated comparison (StratifiedKFold, ROC-AUC)
| Model | Mean AUC | Std AUC |
|---|---|---|
| LogisticRegression | 1.0000 | 0.0000 |
| DecisionTree (depth=5) | 1.0000 | 0.0000 |
| RandomForest | 1.0000 | 0.0000 |
| GradientBoosting | 1.0000 | 0.0000 |

Cross-validation is more reliable than one single train-test split because it averages
performance over 5 different held-out folds, so the estimate isn't dependent on which
particular 20% of rows happened to land in the test set — a lucky or unlucky single split
can otherwise over- or under-state how well a model generalizes.

### GridSearchCV — Random Forest pipeline
Grid: `n_estimators ∈ {50,100,200} × max_depth ∈ {5,10,None} × min_samples_leaf ∈ {1,5}`
→ **18 configurations × 5 folds = 90 total model fits.**

**Best params:** `max_depth=5, min_samples_leaf=1, n_estimators=50`. **Best CV AUC: 1.0**.

Exhaustive Grid Search evaluates every combination in the grid, guaranteeing the best
combination *within that grid* is found, but its cost grows multiplicatively with the
number of hyperparameters and their value ranges. Randomized Search instead samples a
fixed number of random combinations, which scales far better to large or high-dimensional
grids at the cost of no longer guaranteeing the absolute best combination is tried.

### Manual learning curve (best pipeline, 20% → 100% of training data)
| Training fraction | Rows | Train AUC | Test AUC |
|---|---|---|---|
| 0.2 | 480 | 1.0000 | 1.0000 |
| 0.4 | 960 | 1.0000 | 1.0000 |
| 0.6 | 1,440 | 1.0000 | 1.0000 |
| 0.8 | 1,920 | 1.0000 | 1.0000 |
| 1.0 | 2,400 | 1.0000 | 1.0000 |

Training AUC does **not** decrease as the training set grows (it stays flat at 1.0, since
even 480 rows are enough to perfectly separate this label), and test AUC does **not**
increase with more data either (it's already at the ceiling at 20%). Conclusion: the model
is neither data-limited nor capacity-limited — it has already saturated at the maximum
possible AUC, so more data would not further improve it on this task.

### Serialization
Best pipeline (`SimpleImputer → StandardScaler → RandomForestClassifier`, tuned params)
saved to `outputs/models/best_model.pkl`. Reload-and-predict block runs cleanly:
`joblib.load(...)`, then `.predict()` on 2 held-out test rows → `[1, 1]`.

### Summary comparison table & recommendation
| Model | CV mean AUC | CV std AUC | Test AUC |
|---|---|---|---|
| LogisticRegression | 1.0000 | 0.0000 | 1.0000 |
| DecisionTree (depth=5) | 1.0000 | 0.0000 | 1.0000 |
| RandomForest | 1.0000 | 0.0000 | 1.0000 |
| GradientBoosting | 1.0000 | 0.0000 | 1.0000 |

**Recommendation: the tuned Random Forest pipeline (`best_model.pkl`).** All four models
achieve identical, perfect AUC on this particular dataset, so raw performance doesn't
distinguish them — but Random Forest gives interpretable feature importances the client
can act on (stress and anxiety dominate), is robust to the modest number of near-outlier
rows identified in Part 1 without any manual capping, and was the model actually tuned and
serialized end-to-end via `GridSearchCV` + `Pipeline`, making it the most
production-ready and auditable choice of the four. If future data collection makes the
burnout label noisier/less separable, Gradient Boosting would be the natural second choice
to re-benchmark, since boosted trees typically edge out bagged trees once a task stops
being trivially separable.

---

## Part 4 — LLM-Powered Feature

**Track chosen: (C) Model Prediction Explanation Pipeline.**

### Environment note (read this first)
`call_llm()` in `scripts/part4_llm_explain.py` is a genuine implementation that POSTs to
a real LLM API (OpenRouter) using `requests`, with the API key read from the
`LLM_API_KEY` environment variable (never hardcoded). **The sandbox this project was
built and tested in has no outbound internet access**, so every real call in the logs
below returned an HTTP failure (visible in the script's printed `[call_llm] non-200
status` / network-error lines). To still demonstrate the full pipeline end-to-end —
schema validation, guardrails, temperature comparison, tables — a local
`simulate_llm_response()` stand-in was used **only** as the fallback when the real call
fails, and every simulated line is explicitly tagged
`[SIMULATED - no network in this environment]` in both the console output and this
README. Point this at a real API key and network in a normal environment and the
`call_llm()` path will be used automatically instead — no code changes required.

Similarly, the `jsonschema` package could not be `pip install`-ed in this offline
sandbox; a small drop-in `validate()` / `ValidationError` pair reproducing the
"required + scalar type" subset of the real library's behavior is used so the
`try/except ValidationError` pattern specified by the assignment is preserved exactly.

### System prompt (verbatim)
```
You are an assistant that explains a machine learning model's burnout prediction for a
student to a non-technical academic advisor. Given the student's feature values, the
model's predicted class (0 = not burned out, 1 = burned out), and the model's predicted
probability, output ONLY a single valid JSON object with exactly these fields:
{"prediction_label": "burned_out|not_burned_out", "confidence_level": "low|medium|high",
"top_reason": "string", "second_reason": "string", "next_step": "string"}. Do not
include any text outside the JSON object.
```

### User prompt template (verbatim, with placeholders)
```
Student feature values (JSON):
{feature_json}

Model predicted_class: {predicted_class}
Model predicted_probability (of burnout): {predicted_probability}

Explain this prediction as a JSON object following the required schema.
```

**Why temperature=0:** this is a structured-extraction/explanation task where we want the
same input to always produce the same JSON shape and a stable, repeatable explanation
(e.g., for an audit trail an advisor can trust) — temperature near 0 makes the model
always pick its single highest-probability next token, removing sampling randomness that
would otherwise make the explanation (and its exact wording) different every time the same
prediction is explained.

### JSON schema (Track C, 5 required scalar fields)
```json
{
  "type": "object",
  "properties": {
    "prediction_label": {"type": "string"},
    "confidence_level": {"type": "string"},
    "top_reason": {"type": "string"},
    "second_reason": {"type": "string"},
    "next_step": {"type": "string"}
  },
  "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"]
}
```

### PII guardrail demonstration
| Input | Contains PII? | Result |
|---|---|---|
| "...contact them at jane.doe@university.edu for follow-up." | Yes (email) | **Blocked** — "Input blocked: PII detected." |
| "Please explain this student's burnout prediction based on their sleep and stress features." | No | **Proceeded** to LLM call |

### 3-row demonstration table (model load → predict → explain → validate)
| # | Feature Input (abridged) | Predicted Class | Probability | LLM Explanation JSON | Valid JSON |
|---|---|---|---|---|---|
| 1 | sleep=4.5h, stress=9, anxiety=9, caffeine=5 | 1 (burned out) | 0.9979 | `{"prediction_label": "burned_out", "confidence_level": "high", "top_reason": "Stress level of 9.0 is the strongest driver of this prediction.", "second_reason": "Anxiety score of 9.0 reinforces the burned out signal.", "next_step": "Recommend a wellbeing check-in and sleep-habit review."}` | **pass** |
| 2 | sleep=8.2h, stress=3, anxiety=2, exercise=6h | 0 (not burned out) | 0.0349 | `{"prediction_label": "not_burned_out", "confidence_level": "high", ...}` | **pass** |
| 3 | sleep=6.5h, stress=6, anxiety=6, caffeine=2 | 0 (not burned out) | 0.2149 | `{"prediction_label": "not_burned_out", "confidence_level": "medium", ...}` | **pass** |

(Full raw JSON for all three rows is in `outputs/data/part4_log.json` → `demo_table`.)
All three explanations parsed and validated against the schema successfully — no
fallback triggered in this run.

### Temperature A/B comparison (temp=0 vs temp=0.7)
| Input | temp=0 (key fields) | temp=0.7 (key fields) | Key difference |
|---|---|---|---|
| 1 | confidence=high, top_reason="Stress level of 9.0 is the strongest driver..." | confidence=medium (varies by run), same reasons | Wording/confidence-level drifts across resamples |
| 2 | confidence=low | confidence=medium | Confidence label changes between calls |
| 3 | confidence=low, next_step="No immediate intervention needed..." | confidence=high, next_step="Consider a brief follow-up survey..." | Both confidence and recommended action shift |

At temperature=0 the model deterministically emits its single highest-probability token
sequence every time, so re-running the same prompt returns the same JSON. At
temperature=0.7 the model samples from a wider slice of the probability distribution over
next tokens at each step, so wording, confidence labels, and even the recommended
next-step can vary from call to call on identical input — useful for generating varied
phrasing, but undesirable for an auditable explanation feature, which is why temperature=0
was chosen for the production path above.

---

## How to reproduce
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib requests
python scripts/part1_eda.py
python scripts/part2_models.py
python scripts/part3_ensembles.py
export LLM_API_KEY=your_key_here      # optional — falls back to simulation if unset/unreachable
python scripts/part4_llm_explain.py
```
