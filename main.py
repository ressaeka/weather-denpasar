import os
import pandas as pd
from tabulate import tabulate

from src.data_prosessing import load_data, clean_data
from src.train_models import train_all_regression, train_classifier_weather
from src.predik import predict_all, predict_weather_class
from src.visualize import plot_distributions, plot_trend_with_prediction, plot_trend_rain

DATA_PATH = os.path.join("data", "cuaca_clean.csv")

def main():
    # 1) Load + Clean
    df_raw = load_data(DATA_PATH)
    df = clean_data(df_raw)
    if df.empty:
        print("DF kosong, hentikan.")
        return

    # 2) Train REGRESI untuk semua numeric
    target_cols = [c for c in df.select_dtypes(include='number').columns]  # numeric saja
    print("\n--- LATIH REGRESI SEMUA NUMERIC ---")
    train_all_regression(df, target_cols=target_cols)

    # 3) Prediksi semua numeric
    print("\n--- PREDIKSI SEMUA NUMERIC ---")
    df_pred = predict_all(df, target_cols=target_cols)

    # 4) Visualisasi
    plot_distributions(df)
    if "Suhu_prediksi" in df_pred.columns:
        plot_trend_with_prediction(df_pred, df_pred["Suhu_prediksi"])
    if "Curah_Hujan_prediksi" in df_pred.columns:
        plot_trend_rain(df_pred)

    # 5) Train KLASIFIKASI (kategori cuaca)
    print("\n--- LATIH KLASIFIKASI KATEGORI CUACA ---")
    clf_model, acc = train_classifier_weather(df)

    # Prediksi kategori cuaca 5 baris pertama
    sample = df.head(5)
    classes = predict_weather_class(sample)
    out = sample.copy()

    # Tambahkan prediksi numeric (hanya yang ada di df_pred)
    for col in target_cols:
        pred_col = f"{col}_prediksi"
        if pred_col in df_pred.columns:
            out[pred_col] = df_pred.loc[:4, pred_col]  # 5 baris pertama

    out["Pred_Kategori"] = classes

    print(f"\nPrediksi 5 baris pertama:")
    print(tabulate(
        out.reset_index(drop=True),
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False
    ))

if __name__ == "__main__":
    main()
