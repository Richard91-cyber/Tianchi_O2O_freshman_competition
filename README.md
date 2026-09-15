# Tianchi O2O Coupon Redemption Prediction

> Predicting whether a user will redeem a coupon within 15 days of receiving it, using offline O2O (online-to-offline) transaction logs. Built for the **Tianchi O2O Freshman Competition (天池 O2O 新手赛)**.

***

## 📌 Overview

| Item               | Detail                                                                             |
| ------------------ | ---------------------------------------------------------------------------------- |
| **Competition**    | Tianchi O2O Freshman Competition (阿里天池 O2O 新手赛)                                    |
| **Host**           | Alibaba / Tianchi                                                                  |
| **Task Type**      | Binary classification on coupon redemption logs                                    |
| **Target**         | Will a user redeem a received coupon within 15 days?                               |
| **Metric**         | AUC (Area Under the ROC Curve)                                                     |
| **Score**          | AUC ≈ 0.7961 (leaderboard rank \~30–40 at submission time)                         |
| **Core Technique** | Hand-crafted statistical / ranking / time-gap features + GBDT (XGBoost / LightGBM) |
| **Language**       | Python (pandas / numpy / xgboost / lightgbm / scikit-learn)                        |

***

## 🧠 Problem Statement

In O2O commerce, platforms distribute coupons to drive offline store visits. Not every received coupon gets redeemed — users forget, choose other merchants, or find the distance too far. Predicting redemption probability lets the platform target high-intent users and optimize coupon allocation.

The challenge:

1. **Sparse labels** — only a fraction of received coupons are redeemed; classes are imbalanced.
2. **Cold-start entities** — users, merchants, and coupons may have very few historical interactions.
3. **Temporal dynamics** — redemption intent decays with time since receipt; features must capture this.

***

## 📂 Dataset

A single offline training file `ccf_offline_stage1_train.csv` with the following schema:

| Column          | Description                                                               |
| --------------- | ------------------------------------------------------------------------- |
| `User_id`       | Anonymized user ID                                                        |
| `Merchant_id`   | Anonymized merchant ID                                                    |
| `Coupon_id`     | Coupon ID (null when no coupon involved)                                  |
| `Discount_rate` | Discount string — either `"X:Y"` (reduce X off Y) or a ratio like `"0.5"` |
| `Distance`      | User–merchant physical distance tier (0–10, null = unknown)               |
| `Date_received` | Day the coupon was received (null if no coupon)                           |
| `Date`          | Day a purchase was made (null if no purchase)                             |

A separate test file `ccf_offline_stage1_test_revised.csv` provides the coupon-receipt records to predict on.

***

## 🏗️ Solution Architecture

```
        ┌────────────────────────────────────────────────┐
        │       ccf_offline_stage1_train.csv (raw)       │
        └──────────────────────────┬─────────────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │  Data_preprocess.py     │
                       │  • Label: redeemed ≤15d│
                       │  • Discount parsing     │
                       │  • Time-window split   │
                       └───────────┬────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            │                                             │
   ┌────────▼─────────┐                       ┌──────────▼──────────┐
   │  feat_section.py │  feature window      │ label_section.py    │  prediction window
   │  (past behavior)  │                       │  (current behavior) │
   └────────┬─────────┘                       └──────────┬──────────┘
            │                                            │
            │           ┌────────────────────┐           │
            └──────────►│ time_gap_count.py  │◄──────────┘
                        │  • inter-event gaps│
                        │  • before/after cnt│
                        │  • leak features   │
                        └─────────┬──────────┘
                                   │
                          ┌────────▼────────┐
                          │  ~200+ features  │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │   model.py       │
                          │  • RFE selection │
                          │  • XGBoost / LGB │ ← AUC evaluation
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  lgb.csv / xgb  │ → submission
                          └─────────────────┘
```

***

## 🔧 Feature Engineering (Core Contribution)

The project builds **three primary feature groups** (user / merchant / coupon) plus **three cross groups** (user×merchant, user×coupon, merchant implicitly via coupon). For every entity, features are computed in three slices: **received / redeemed / not-redeemed**, yielding count, unique-count, and ratio signals.

### 1. User Features (`feat_section.py` / `label_section.py`)

- Coupon counts & types: received, redeemed, dropped, redeem-ratio, coupon-type diversity
- Merchant diversity: unique merchants received-from / redeemed-with, redeem-ratio, coverage vs. all merchants
- Distance stats: max / min / mean for received, redeemed, and dropped coupons
- Discount stats: max / min / mean of `dis_left`, `dis_right`, and `dis_rate` across all three slices
- Discount-range bucketing: counts in `0–50`, `50–200`, `200–500` left-value bands

### 2. Merchant Features

- Mirror of user features, from the merchant's perspective: coupons received/redeemed/dropped, user reach, distance & discount stats

### 3. Coupon Features

- Reach & redemption counts, unique users per coupon, distance statistics

### 4. Cross Features

- **User × Merchant**: co-occurrence counts, redeem ratio, share of user's total redeems, share of user's total drops
- **User × Coupon**: same pattern — a strong personalization signal

### 5. Ranking Features (`label_section.py`)

Per entity (and per entity pair), ascending & descending ranks over:

- `distance` — closer merchants rank higher
- `date_received` — recent receipts rank higher
- `dis_left` / `dis_right` — bigger discounts rank higher

These ranks are computed within each user's / merchant's / coupon's own sample set. The README's author notes that **ranking features outperform the raw distance/discount values** for XGBoost — likely because the split-score function operates on second-order-derivative quantiles, and monotone ranks produce smoother, more evenly distributed split candidates than raw values.

### 6. Time-Gap Features (`time_gap_count.py`)

For each record and each of (user, merchant, coupon), compute:

- Max / min / mean / median gap between consecutive events of the same entity
- Gap from current record to the entity's first and last events
- Previous-event gap and next-event gap
- Before/after counts and ratios for merchants & coupons (these are **leak features** — they peek at future events within the prediction window; competition-only)

### 7. Calendar Features

- Weekend / Saturday / Sunday / weekday indicators
- First / second / third ten-day-of-month indicators
- Holiday flags: May 1 (Labor Day), June 1 (Children's Day), June 9 (Dragon Boat), May 8 (Mother's Day), June 19 (Father's Day), June 21 (Summer Solstice)

### Design Principles

- **Three-slice decomposition** — every statistic is split into received / redeemed / dropped, capturing both intent and abandonment signals.
- **Cross-entity personalization** — user×merchant and user×coupon interactions encode "this user's behavior *with this specific partner*."
- **Ranking > raw value** — rank transforms stabilize the tree-split scoring (hypothesis documented in the original README).
- **Multi-temporal granularity** — feature-window stats for long-term habits, prediction-window stats for recent intent, time-gaps for intra-window dynamics.

***

## 🤖 Modeling (`model.py`)

Two GBDT models are trained; LightGBM is preferred for speed, XGBoost for precision.

| Model        | `eta` / `lr` | `num_boost_round` | Use                                           |
| ------------ | ------------ | ----------------- | --------------------------------------------- |
| **XGBoost**  | 0.01         | 1200              | Higher-precision submissions (\~30 min train) |
| **LightGBM** | 0.01         | 800               | Fast iteration (\~5 min train)                |

Shared XGBoost hyperparameters: `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=18`, `objective=binary:logistic`, `eval_metric=auc`.

### Feature Selection

- Uses **Recursive Feature Elimination (RFE)** from scikit-learn, driven by an `XGBClassifier`, to select the top 180 features.
- The selected feature list is saved to `feat_select.csv` and reused by both trainers — removing noisy features improves AUC and reduces overfitting on cold-start entities.

### Output

- LightGBM probabilities are scaled by 0.98 to calibrate against the public leaderboard distribution.
- Submissions are written as `User_id, Coupon_id, Date_received, Probability`.

***

## ⏱️ Validation Strategy

A **time-based sliding window** (not random K-fold) — essential for temporal data:

| Split                  | Feature Window          | Prediction Window       |
| ---------------------- | ----------------------- | ----------------------- |
| **Train**              | 2016-04-01 → 2016-05-31 | 2016-06-01 → 2016-06-30 |
| **Validation**         | 2016-03-01 → 2016-04-30 | 2016-05-01 → 2016-05-30 |
| **Test (leaderboard)** | 2016-05-01 → 2016-06-30 | 2016-07-01 → 2016-07-31 |

For the final leaderboard submission, the train and validation sets are merged to maximize training data. Train and validation can be cross-validated offline; the test split respects strict temporal ordering to prevent leakage.

***

## ⚙️ Engineering Highlights

- **Modular pipeline** — each phase (preprocessing, feature-window, prediction-window, time-gaps, modeling) is a standalone module, making it easy to swap feature groups or models.
- **Vectorized aggregations** — `pd.pivot_table` with custom `getcount` / `getset` aggfuncs computes per-entity stats in a single pass.
- **List-comprehension iteration** — `time_gap_count.py` uses list comprehensions over `iterrows()` instead of Python loops, dramatically speeding up the per-row feature computation.
- **Memory-aware I/O** — intermediate feature tables are persisted to CSV per split (`train/feat.csv`, `val/label.csv`, etc.), allowing the expensive feature step to run once and models to iterate quickly.
- **Feature selection artifact** — RFE output is cached to `feat_select.csv` and reloaded by both models, keeping train and test feature spaces perfectly aligned.

***

## 📁 Project Structure

```
github/
├── Data_preprocess.py     # Labeling, discount parsing, time-window splitting
├── feat_section.py        # Feature-window features (user / merchant / coupon / cross)
├── label_section.py       # Prediction-window features + ranking + calendar features
├── time_gap_count.py      # Inter-event time gaps and leak features
├── model.py               # XGBoost / LightGBM training + RFE feature selection
└── README.md              # This file
```

External data files (not committed) sit in the parent directory and are referenced via `r'..\xxx.csv'`:

```
../
├── ccf_offline_stage1_train.csv         # Raw training data
├── ccf_offline_stage1_test_revised.csv  # Leaderboard test receipts
├── row_train.csv                        # Labeled output of mk_label()
├── train.csv / val.csv                  # Engineered feature tables
├── feat_select.csv                      # RFE-selected feature list (top 180)
└── lgb.csv / xgb.csv                    # Submission files
```

***

## 📝 Key Takeaways

- **Three-slice decomposition (received / redeemed / dropped)** is a high-leverage pattern for conversion-style problems — it turns a single count into an intent signal.
- **Ranking features beat raw values** for tree models — the hypothesis (documented in the original notes) is that rank-transformed features produce smoother second-order-derivative quantiles during tree splitting, yielding larger information gain. This is a useful mental model for feature engineering on GBDT.
- **Leakage features (future events within the prediction window)** provide an easy AUC bump in competitions but are **not** deployable in production. Worth flagging explicitly to avoid copying this pattern into real systems.
- **RFE + LightGBM** is a fast, reproducible combo: RFE prunes noisy features once, then LightGBM iterates in minutes rather than the 30+ minutes XGBoost requires.

***

## 📜 License

Personal project for educational and portfolio purposes. Dataset © Alibaba / Tianchi competition organizers.
