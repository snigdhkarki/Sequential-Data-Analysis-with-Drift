"""
Predict "chance of an alarm in the next 100 rows" for each row.

- Trains on rows [0, 70000).
- Predicts for rows [70000, 100000), giving each row a 0-100% probability
  that an alarm (alarm == 1) occurs somewhere in the *next 100 rows*.
- All features are causal: for a test row i, they are built only from data
  between row 70,000 and row i itself (a rolling/expanding window that
  resets at the start of the test partition) -- never from the future,
  and never by reaching back into the training rows.

Model: HistGradientBoostingClassifier (sklearn's fast gradient-boosted
trees). It's a good fit here because:
  - the 5 anomaly types produce very different, non-linear feature
    signatures (a spike looks nothing like a slow drift or a variance
    blow-up), and boosted trees handle that kind of heterogeneity without
    needing hand-picked interaction terms.
  - it natively handles the moderate class imbalance via sample weights.
  - it's fast enough to comfortably handle 100k rows / dozens of features.

Usage:
    pip install pandas numpy scikit-learn
    python train_alarm_predictor.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

CSV_PATH = "sensor_data.csv"
TRAIN_ROWS = 70_000
FUTURE_HORIZON = 100          # "alarm in the next 100 rows"
ROLL_WINDOWS = [20, 50, 100, 300]   # short -> catches spikes, long -> catches slow drift
LONG_BASELINE_WINDOW = 1000         # mirrors the generator's own rolling baseline
DECISION_THRESHOLD = 0.5            # only used for the printed classification report
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Feature engineering (fully causal: everything is a rolling window
# ending at the current row, so it only ever "sees" the past)
# ----------------------------------------------------------------------
def add_features(df):
    df = df.copy()

    # rows since the last observed alarm, counted *within this partition only*
    # -> a fresh test partition starts this counter over at row 0
    last_alarm_pos = df.index.to_series().where(df["alarm"] == 1).ffill()
    df["rows_since_last_alarm"] = (df.index.to_series() - last_alarm_pos).fillna(len(df))

    for col in ["temp", "pressure"]:
        long_mean = df[col].rolling(LONG_BASELINE_WINDOW, min_periods=1).mean()

        for w in ROLL_WINDOWS:
            roll = df[col].rolling(w, min_periods=1)
            df[f"{col}_mean_{w}"] = roll.mean()
            df[f"{col}_std_{w}"] = roll.std().fillna(0)
            df[f"{col}_min_{w}"] = roll.min()
            df[f"{col}_max_{w}"] = roll.max()
            df[f"{col}_range_{w}"] = df[f"{col}_max_{w}"] - df[f"{col}_min_{w}"]

        diff = df[col].diff().fillna(0)
        df[f"{col}_diff_abs_mean_20"] = diff.abs().rolling(20, min_periods=1).mean()
        df[f"{col}_diff_std_20"] = diff.rolling(20, min_periods=1).std().fillna(0)
        # how far the short-term average has drifted from the long-term baseline
        # (catches spikes/drift/spirals that pull the short average away from normal)
        df[f"{col}_dev_from_baseline"] = df[f"{col}_mean_20"] - long_mean

    return df


def add_future_alarm_label(df, horizon=FUTURE_HORIZON):
    """y[i] = 1 if alarm occurs in rows (i, i+horizon], else 0. Uses this
    partition's own data only -- used for training targets and for scoring,
    never fed to the model as an input feature."""
    alarm_shifted = df["alarm"].shift(-1)                       # value at i = alarm at i+1
    future_max = alarm_shifted[::-1].rolling(horizon, min_periods=1).max()[::-1]
    return future_max.fillna(0).astype(int)


def main():
    raw = pd.read_csv(CSV_PATH)

    train_raw = raw.iloc[:TRAIN_ROWS].reset_index(drop=True)
    test_raw = raw.iloc[TRAIN_ROWS:].reset_index(drop=True)

    train_feat = add_features(train_raw)
    test_feat = add_features(test_raw)  # rolling windows restart at row 70,000

    train_feat["y"] = add_future_alarm_label(train_raw)
    test_feat["y"] = add_future_alarm_label(test_raw)  # for evaluation only

    feature_cols = [c for c in train_feat.columns if c not in ("temp", "pressure", "alarm", "y")]

    X_train, y_train = train_feat[feature_cols], train_feat["y"]
    X_test, y_test = test_feat[feature_cols], test_feat["y"]

    print(f"Train rows: {len(X_train)}  (positive rate: {y_train.mean():.2%})")
    print(f"Test rows:  {len(X_test)}  (positive rate: {y_test.mean():.2%})")

    # balance the (moderate) class imbalance with sample weights
    pos, neg = y_train.sum(), len(y_train) - y_train.sum()
    sample_weight = np.where(y_train == 1, neg / pos, 1.0)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    test_probs = model.predict_proba(X_test)[:, 1]

    # ---------------- evaluation ----------------
    auc = roc_auc_score(y_test, test_probs)
    ap = average_precision_score(y_test, test_probs)
    print(f"\nTest ROC-AUC: {auc:.4f}")
    print(f"Test PR-AUC (average precision): {ap:.4f}")

    preds_binary = (test_probs >= DECISION_THRESHOLD).astype(int)
    print(f"\nClassification report @ threshold {DECISION_THRESHOLD}:")
    print(classification_report(y_test, preds_binary, digits=3))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, preds_binary))

    print("\nComputing permutation importance on a sample (this takes a bit)...")
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=5000, replace=False)
    imp = permutation_importance(
        model, X_test.iloc[sample_idx], y_test.iloc[sample_idx],
        n_repeats=5, random_state=RANDOM_STATE, scoring="average_precision", n_jobs=-1,
    )
    top_features = pd.Series(imp.importances_mean, index=feature_cols).sort_values(ascending=False).head(10)
    print("\nTop 10 features (by drop in PR-AUC when shuffled):")
    print(top_features.to_string())

    # ---------------- save per-row predictions ----------------
    results = pd.DataFrame({
        "row_index": np.arange(TRAIN_ROWS, TRAIN_ROWS + len(test_raw)),
        "temp": test_raw["temp"],
        "pressure": test_raw["pressure"],
        "actual_alarm": test_raw["alarm"],
        "alarm_within_next_100_rows_pct": np.round(test_probs * 100, 2),
    })
    out_path = "alarm_probability_predictions.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved per-row predictions to {out_path}")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
