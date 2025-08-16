import os
import joblib
import pandas as pd
from typing import List

# -----------------------
# REGRESI
# -----------------------

def load_model(target_col: str):
    """
    Load model regresi dari folder 'models' berdasarkan nama target_col.
    """
    path = os.path.join("models", f"{target_col}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model {target_col} tidak ditemukan di {path}")
    return joblib.load(path)


def predict_feature(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Prediksi satu kolom numeric menggunakan model regresi.
    - target_col: nama kolom target numeric
    - df: DataFrame input (harus mengandung semua fitur numeric)
    """
    model = load_model(target_col)
    # gunakan semua kolom numeric selain target
    feature_cols = [c for c in df.select_dtypes(include='number').columns if c != target_col]
    X_new = df[feature_cols].copy()
    preds = model.predict(X_new)
    return pd.Series(preds, index=df.index, name=f"{target_col}_prediksi")


def predict_all(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """
    Prediksi beberapa kolom numeric sekaligus.
    - target_cols: daftar nama kolom target
    """
    df_pred = df.copy()
    for col in target_cols:
        print(f"Prediksi {col}...")
        df_pred[f"{col}_prediksi"] = predict_feature(df, col)
    return df_pred


# -----------------------
# KLASIFIKASI
# -----------------------

def load_classifier(path: str = os.path.join("models", "weather_predictor.pkl")):
    """
    Load model klasifikasi cuaca.
    Model disimpan sebagai dict {'imputer': ..., 'model': ..., 'features': [...]}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model klasifikasi tidak ditemukan: {path}")
    return joblib.load(path)


def predict_weather_class(df: pd.DataFrame) -> pd.Series:
    """
    Prediksi kategori cuaca menggunakan model klasifikasi.
    - df: DataFrame input (harus mengandung fitur yang sama seperti saat training)
    """
    artefact = load_classifier()
    imp = artefact["imputer"]       # SimpleImputer atau pipeline transform
    clf = artefact["model"]         # classifier (RandomForest, XGB, dll)
    features = artefact["features"] # list fitur input
    
    X = df[features].copy()
    X_imp = imp.transform(X)
    preds = clf.predict(X_imp)
    return pd.Series(preds, index=df.index, name="Pred_Kategori")
