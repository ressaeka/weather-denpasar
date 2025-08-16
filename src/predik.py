import os
import joblib
import pandas as pd
import numpy as np
from typing import List

# ---------------- REGRESI ----------------

def load_model(target_col: str):
    path = os.path.join("models", f"{target_col}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model {target_col} tidak ditemukan di {path}")
    return joblib.load(path)

def predict_feature(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Prediksi satu kolom numeric
    """
    model = load_model(target_col)
    # pakai kolom numeric selain target_col
    feature_cols = [c for c in df.select_dtypes(include='number').columns if c != target_col]
    X_new = df[feature_cols].copy()
    preds = model.predict(X_new)
    return pd.Series(preds, index=df.index, name=f"{target_col}_prediksi")

def predict_all(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """
    Prediksi semua kolom numeric
    """
    df_pred = df.copy()
    for col in target_cols:
        print(f"Prediksi {col}...")
        df_pred[f"{col}_prediksi"] = predict_feature(df, col)
    return df_pred

# ---------------- KLASIFIKASI ----------------

def load_classifier(path: str = os.path.join("models", "weather_predictor.pkl")):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model klasifikasi tidak ditemukan: {path}")
    return joblib.load(path)

def predict_weather_class(df: pd.DataFrame):
    artefact = load_classifier()
    imp = artefact["imputer"]
    clf = artefact["model"]
    features = artefact["features"]
    X = df[features].copy()
    X_imp = imp.transform(X)
    preds = clf.predict(X_imp)
    return pd.Series(preds, index=df.index, name="Pred_Kategori")
