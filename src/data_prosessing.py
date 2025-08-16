import os
import pandas as pd
from typing import Optional, List

# Kandidat nama kolom tanggal
DATE_CANDIDATES: List[str] = ["Tanggal", "tanggal", "Date", "date"]

# -----------------------
# Fungsi: cari kolom tanggal
# -----------------------
def find_date_col(columns: List[str]) -> Optional[str]:
    """Cek apakah salah satu kandidat tanggal ada di daftar kolom."""
    for c in DATE_CANDIDATES:
        if c in columns:
            return c
    return None

# -----------------------
# Fungsi: load CSV & parsing tanggal
# -----------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV dari path.
    Jika kolom tanggal ditemukan, parse ke datetime dan rename jadi 'Tanggal'.
    Data akan diurutkan berdasarkan tanggal jika ada.
    """
    if not os.path.exists(path):
        print(f"File tidak ditemukan: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error baca CSV: {e}")
        return pd.DataFrame()

    # Deteksi kolom tanggal
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

# -----------------------
# Fungsi: clean & standardisasi data
# -----------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan data cuaca:
    - Rename kolom agar seragam
    - Pilih kolom penting ['Tanggal','Suhu','Curah_Hujan','Kelembaban','Angin','Awan']
    - Isi missing numeric dengan median atau 0
    """
    if df is None or df.empty:
        print("DataFrame kosong setelah load.")
        return pd.DataFrame()

    # Mapping kolom seragam
    rename_map = {
        "Suhu": "Suhu",
        "Curah_Hujan": "Curah_Hujan",
        "Kelembaban": "Kelembaban",
        "Angin": "Angin",
        "Kecepatan_Angin": "Angin",
        "Awan": "Awan",
        "Tutupan_Awan": "Awan",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Pilih kolom yang ada
    keep = ["Tanggal", "Suhu", "Curah_Hujan", "Kelembaban", "Angin", "Awan"]
    cols_exist = [c for c in keep if c in df.columns]
    df = df[cols_exist].copy()

    # Kolom numerik
    numeric_cols = [c for c in ["Suhu", "Curah_Hujan", "Kelembaban", "Angin", "Awan"] if c in df.columns]

    # Tangani missing Curah_Hujan
    if "Curah_Hujan" in df.columns:
        if df["Curah_Hujan"].notna().sum() == 0:
            df["Curah_Hujan"] = 0.0
        else:
            df["Curah_Hujan"] = df["Curah_Hujan"].fillna(df["Curah_Hujan"].median())

    # Tangani missing kolom numerik lain
    for c in numeric_cols:
        if c == "Curah_Hujan":
            continue
        if df[c].notna().sum() == 0:
            df[c] = 0
        else:
            df[c] = df[c].fillna(df[c].median())

    return df
