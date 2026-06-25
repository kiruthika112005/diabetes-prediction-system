"""
Diabetes Prediction - Machine Learning Model
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

DATA_PATH   = "diabetes.csv"
MODEL_PATH  = "diabetes_model.pkl"
SCALER_PATH = "diabetes_scaler.pkl"

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age",
]
TARGET_COL   = "Outcome"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' file not found!\n"
            "diabetes.csv - ஐ model.py அதே folder-ல வையுங்கள்."
        )
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded → {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def preprocess(df):
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_invalid:
        median_val = df[col].replace(0, np.nan).median()
        df[col] = df[col].replace(0, median_val)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print(f"[INFO] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_split=5, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"[TRAIN] CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")
    return model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)
    print(f"\n Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")
    print(f" TN={cm[0,0]} FP={cm[0,1]} | FN={cm[1,0]} TP={cm[1,1]}")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes","Diabetes"]))
    return acc

def save_model(model, scaler, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[SAVE] Model → '{model_path}' | Scaler → '{scaler_path}'")

def load_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"'{model_path}' not found. Run model.py first!")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"'{scaler_path}' not found. Run model.py first!")
    return joblib.load(model_path), joblib.load(scaler_path)

def predict(input_data, model=None, scaler=None):
    if model is None or scaler is None:
        model, scaler = load_model()
    sample = np.array(input_data, dtype=float).reshape(1, -1)
    scaled = scaler.transform(sample)
    pred   = int(model.predict(scaled)[0])
    prob   = round(float(model.predict_proba(scaled)[0][1]), 4)
    return pred, prob

def main():
    print("=" * 50)
    print(" DIABETES PREDICTION – ML PIPELINE")
    print("=" * 50)
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    model = train_model(X_train, y_train)
    accuracy = test_model(model, X_test, y_test)
    save_model(model, scaler)
    print(f"\n[DONE] Model trained & saved. Accuracy: {accuracy*100:.2f}%\n")

if __name__ == "__main__":
    main()
