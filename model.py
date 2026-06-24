"""
Diabetes Prediction - Machine Learning Model
Dataset: Pima Indians Diabetes Dataset
Features: Pregnancies, Glucose, BloodPressure, SkinThickness,
          Insulin, BMI, DiabetesPedigreeFunction, Age
Target: Outcome (0 = No Diabetes, 1 = Diabetes)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DATA_PATH   = "1782285865556_diabetes.csv"   # CSV file path
MODEL_PATH  = "diabetes_model.pkl"           # Saved model file
SCALER_PATH = "diabetes_scaler.pkl"          # Saved scaler file
FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age",
]
TARGET_COL   = "Outcome"
RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the diabetes CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'. "
                                "Please place the CSV file in the same folder.")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESS DATA
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    - Replace physiologically impossible zeros with column median.
    - Split into features (X) and target (y).
    - Train / test split.
    - Standard-scale the features.
    Returns X_train, X_test, y_train, y_test, scaler.
    """
    # Columns where 0 is biologically impossible
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_invalid:
        median_val = df[col].replace(0, np.nan).median()
        df[col]    = df[col].replace(0, median_val)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[INFO] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler


# ─────────────────────────────────────────────
# 3. TRAIN MODEL
# ─────────────────────────────────────────────
def train_model(X_train, y_train) -> RandomForestClassifier:
    """Train a Random Forest classifier and show cross-validation score."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"[TRAIN] Cross-Val Accuracy (5-fold): "
          f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return model


# ─────────────────────────────────────────────
# 4. TEST / EVALUATE MODEL
# ─────────────────────────────────────────────
def test_model(model, X_test, y_test) -> float:
    """Evaluate the model on the held-out test set and print metrics."""
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(
        y_test, y_pred, target_names=["No Diabetes (0)", "Diabetes (1)"]
    )

    print("\n" + "=" * 55)
    print("           MODEL EVALUATION  (TEST SET)")
    print("=" * 55)
    print(f"  Accuracy      : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC Score : {roc_auc:.4f}")
    print("-" * 55)
    print("  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print("-" * 55)
    print("  Classification Report:")
    print(report)
    print("=" * 55)
    return acc


# ─────────────────────────────────────────────
# 5. SAVE MODEL
# ─────────────────────────────────────────────
def save_model(model, scaler,
               model_path: str = MODEL_PATH,
               scaler_path: str = SCALER_PATH) -> None:
    """Persist the trained model and scaler to disk using joblib."""
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n[SAVE] Model  saved → '{model_path}'")
    print(f"[SAVE] Scaler saved → '{scaler_path}'")


# ─────────────────────────────────────────────
# 6. LOAD MODEL
# ─────────────────────────────────────────────
def load_model(model_path: str  = MODEL_PATH,
               scaler_path: str = SCALER_PATH):
    """Load a previously saved model and scaler from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: '{model_path}'. "
                                "Run train first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: '{scaler_path}'. "
                                "Run train first.")
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"[LOAD] Model  loaded ← '{model_path}'")
    print(f"[LOAD] Scaler loaded ← '{scaler_path}'")
    return model, scaler


# ─────────────────────────────────────────────
# 7. PREDICT (single patient)
# ─────────────────────────────────────────────
def predict(input_data, model=None, scaler=None):
    """
    Predict diabetes for a single patient.

    Parameters
    ----------
    input_data : list or 1-D array-like
        Values in order: [Pregnancies, Glucose, BloodPressure,
                          SkinThickness, Insulin, BMI,
                          DiabetesPedigreeFunction, Age]
    model  : trained model  (loaded automatically if None)
    scaler : fitted scaler  (loaded automatically if None)

    Returns
    -------
    prediction : int   → 0 (No Diabetes) or 1 (Diabetes)
    probability: float → probability of Diabetes (class 1)
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    sample       = np.array(input_data, dtype=float).reshape(1, -1)
    sample_scaled = scaler.transform(sample)

    prediction   = model.predict(sample_scaled)[0]
    probability  = model.predict_proba(sample_scaled)[0][1]

    label = "Diabetes (1)" if prediction == 1 else "No Diabetes (0)"
    print(f"\n[PREDICT] Input  : {dict(zip(FEATURE_COLS, input_data))}")
    print(f"[PREDICT] Result : {label}  |  Probability: {probability:.4f}")
    return int(prediction), round(float(probability), 4)


# ─────────────────────────────────────────────
# MAIN  –  full pipeline
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("   DIABETES PREDICTION  –  ML PIPELINE")
    print("=" * 55 + "\n")

    # Step 1: Load
    df = load_data()

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    # Step 3: Train
    print("\n[TRAIN] Training Random Forest model …")
    model = train_model(X_train, y_train)
    print("[TRAIN] Training complete.")

    # Step 4: Test / Evaluate
    accuracy = test_model(model, X_test, y_test)

    # Step 5: Save
    save_model(model, scaler)

    # ── Demo: reload from disk and predict ──────────────────
    print("\n[DEMO] Loading saved model and predicting a sample patient …")
    loaded_model, loaded_scaler = load_model()

    # Example patient: [Pregnancies, Glucose, BloodPressure,
    #                   SkinThickness, Insulin, BMI,
    #                   DiabetesPedigreeFunction, Age]
    sample_patient = [6, 148, 72, 35, 0, 33.6, 0.627, 50]   # Expected → Diabetes
    pred, prob = predict(sample_patient, loaded_model, loaded_scaler)

    print(f"\n[RESULT] Prediction = {'Diabetes' if pred == 1 else 'No Diabetes'}")
    print(f"[RESULT] Confidence = {prob * 100:.2f}%")
    print(f"[RESULT] Test Accuracy = {accuracy * 100:.2f}%")
    print("\n[INFO] Pipeline finished successfully.\n")


if __name__ == "__main__":
    main()
