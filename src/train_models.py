import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# -----------------------
# REGRESI
# -----------------------

def train_regression_feature(df: pd.DataFrame, target_col: str) -> Tuple[Pipeline, float]:
    """
    Latih model LinearRegression untuk target_col.
    - df: DataFrame input
    - target_col: nama kolom target numeric
    Returns:
        - pipeline yang sudah fit
        - MSE pada test set
    """
    # Ambil fitur numeric selain target
    feature_cols = [c for c in df.select_dtypes(include='number').columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Pipeline: imputasi + LinearRegression
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("lr", LinearRegression())
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Latih model
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # Simpan model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{target_col}_model.pkl")
    joblib.dump(pipe, model_path)
    print(f"[REGRESI] {target_col} saved to {model_path} | MSE={mse:.3f}")

    return pipe, mse


def train_all_regression(df: pd.DataFrame, target_cols: List[str]):
    """
    Latih semua kolom numeric sekaligus.
    """
    for col in target_cols:
        print(f"Melatih {col}...")
        train_regression_feature(df, col)


# -----------------------
# KLASIFIKASI CUACA
# -----------------------

def build_weather_label(df: pd.DataFrame) -> pd.Series:
    """
    Buat label kategori cuaca sederhana berdasarkan Kelembaban & Awan.
    - Hujan: Kelembaban >=85% atau Awan >=70
    - Berawan: Awan >=40
    - Cerah: sisanya
    """
    hum = df["Kelembaban"] if "Kelembaban" in df.columns else pd.Series(0, index=df.index)
    awn = df["Awan"] if "Awan" in df.columns else pd.Series(0, index=df.index)

    label = np.where((hum >= 85) | (awn >= 70), "Hujan",
             np.where(awn >= 40, "Berawan", "Cerah"))
    return pd.Series(label, index=df.index, name="kategori_cuaca")


def train_classifier_weather(df: pd.DataFrame):
    """
    Latih RandomForestClassifier untuk prediksi kategori cuaca.
    Menyimpan artefact: {'imputer', 'model', 'features'}
    """
    feature_cols = [c for c in ["Suhu", "Kelembaban", "Curah_Hujan", "Angin", "Awan"] if c in df.columns]
    y = build_weather_label(df)
    X = df[feature_cols].copy()

    # Imputasi median
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42, stratify=y)

    # RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Print evaluasi
    print(f"[KLASIFIKASI] Accuracy={acc:.3f}")
    print(classification_report(y_test, y_pred))

    # Simpan model & artefact
    artefact = {"imputer": imp, "model": clf, "features": feature_cols}
    os.makedirs("models", exist_ok=True)
    joblib.dump(artefact, "models/weather_predictor.pkl")
    print("[KLASIFIKASI] Model saved to models/weather_predictor.pkl")

    return clf, acc
