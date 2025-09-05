"""
Clean & Transform TikTok OCR Data
- Lee data/ocr_extracted_lives.csv
- Genera 3 datasets listos para Power BI:
    * tiktok_live_dataset_2025.csv
    * daily_summary_2025.csv
    * monthly_summary_2025.csv
"""

import pandas as pd
from pathlib import Path

IN_CSV = Path("data/ocr_extracted_lives.csv")
OUT_DATASET = Path("data/tiktok_live_dataset_2025.csv")
OUT_DAILY = Path("data/daily_summary_2025.csv")
OUT_MONTHLY = Path("data/monthly_summary_2025.csv")

FORCE_YEAR = 2025

def main():
    df = pd.read_csv(IN_CSV)

    # Convertir fecha
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if FORCE_YEAR:
        df["date"] = df["date"].apply(lambda d: d.replace(year=FORCE_YEAR) if pd.notna(d) else d)

    # Construir columnas datetime
    df["start_dt"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + df["start"], errors="coerce")
    df["end_dt"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + df["end"], errors="coerce")

    # Calcular duración
    df["duration_minutes"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 60
    df.loc[df["duration_minutes"] < 0, "duration_minutes"] = 0

    # Columnas extra
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["weekday"] = df["date"].dt.day_name()
    df["hour_start"] = df["start_dt"].dt.hour
    df["hour_end"] = df["end_dt"].dt.hour

    dataset = df[["date","start_dt","end_dt","duration_minutes","likes","month","week","weekday","hour_start","hour_end"]]

    # Resúmenes
    daily = df.groupby("date", as_index=False).agg(
        lives_count=("likes","count"),
        total_likes=("likes","sum"),
        avg_likes=("likes","mean"),
        total_duration_min=("duration_minutes","sum"),
    )
    monthly = df.groupby("month", as_index=False).agg(
        lives_count=("likes","count"),
        total_likes=("likes","sum"),
        avg_likes=("likes","mean"),
        total_duration_min=("duration_minutes","sum"),
    )

    # Guardar
    dataset.to_csv(OUT_DATASET, index=False)
    daily.to_csv(OUT_DAILY, index=False)
    monthly.to_csv(OUT_MONTHLY, index=False)
    print(f"[OK] Generados:\n- {OUT_DATASET}\n- {OUT_DAILY}\n- {OUT_MONTHLY}")

if __name__ == "__main__":
    main()
