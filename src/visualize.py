import os
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)

def plot_distributions(df: pd.DataFrame, outdir="outputs"):
    ensure_dir(outdir)
    for col in ["Suhu", "Kelembaban", "Angin", "Awan"]:
        if col in df.columns and df[col].notna().sum() > 0:
            plt.figure(figsize=(7,4))
            plt.hist(df[col].dropna(), bins=30, edgecolor="black")
            plt.title(f"Distribusi {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"dist_{col.lower()}.png"))
            plt.close()

def plot_trend_with_prediction(df: pd.DataFrame, preds: pd.Series, outpath="outputs/trend_suhu_pred.png"):
    ensure_dir(os.path.dirname(outpath) or "outputs")
    if "Tanggal" not in df.columns or "Suhu" not in df.columns:
        return
    plt.figure(figsize=(12,5))
    plt.plot(df["Tanggal"], df["Suhu"], label="Suhu Aktual")
    plt.plot(df["Tanggal"], preds, label="Suhu Prediksi", linestyle="--")
    plt.title("Tren Suhu: Aktual vs Prediksi")
    plt.xlabel("Tanggal")
    plt.ylabel("Suhu (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_trend_rain(df: pd.DataFrame, outpath="outputs/trend_curah_hujan.png"):
    ensure_dir(os.path.dirname(outpath) or "outputs")
    if "Tanggal" not in df.columns or "Curah_Hujan" not in df.columns:
        return
    if df["Curah_Hujan"].notna().sum() == 0:
        return
    plt.figure(figsize=(12,4))
    plt.plot(df["Tanggal"], df["Curah_Hujan"])
    plt.title("Tren Curah Hujan")
    plt.xlabel("Tanggal")
    plt.ylabel("Curah Hujan (mm)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
