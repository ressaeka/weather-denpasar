import os
import pandas as pd
from typing import Optional, List

DATE_CANDIDATES: List[str] = ["Tanggal", "tanggal", "Date", "date"]

def find_date_col(columns: List[str]) -> Optional[str]:
    for c in DATE_CANDIDATES:
        if c in columns:
            return c
    return None

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"File tidak ditemukan: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error baca CSV: {e}")
        return pd.DataFrame()

    date_col = find_date_col(df.columns.tolist())
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(by=date_col).reset_index(drop=True)
            df.rename(columns={date_col: "Tanggal"}, inplace=True)
        except Exception as e:
            print(f"Peringatan: gagal parse kolom tanggal: {e}")
    else:
        print("Peringatan: kolom tanggal tidak ditemukan.")

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        print("DataFrame kosong setelah load.")
        return pd.DataFrame()

    rename_map = {
        "Suhu": "Suhu",
        "Curah_Hujan": "Curah_Hujan",
        "Kelembaban": "Kelembaban",
        "Angin": "Angin",
        "Kecepatan_Angin": "Angin",
        "Awan": "Awan",
        "Tutupan_Awan": "Awan",
    }
    for k, v in list(rename_map.items()):
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    keep = ["Tanggal", "Suhu", "Curah_Hujan", "Kelembaban", "Angin", "Awan"]
    cols_exist = [c for c in keep if c in df.columns]
    df = df[cols_exist].copy()

    numeric_cols = [c for c in ["Suhu", "Curah_Hujan", "Kelembaban", "Angin", "Awan"] if c in df.columns]

    if "Curah_Hujan" in df.columns:
        if df["Curah_Hujan"].notna().sum() == 0:
            df["Curah_Hujan"] = 0.0
        else:
            df["Curah_Hujan"] = df["Curah_Hujan"].fillna(df["Curah_Hujan"].median())

    for c in numeric_cols:
        if c == "Curah_Hujan":
            continue
        if df[c].notna().sum() == 0:
            df[c] = 0
        else:
            df[c] = df[c].fillna(df[c].median())

    return df
