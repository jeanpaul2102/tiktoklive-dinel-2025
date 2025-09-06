"""
Clean & Transform TikTok OCR Data (ajustado para OCR v3)
- Lee 'ocr_extracted_lives.csv' generado por el script OCR
- Genera 3 datasets listos para análisis / Power BI:
    * tiktok_live_dataset_2025.csv
    * daily_summary_2025.csv
    * monthly_summary_2025.csv
- Normaliza fechas, horas, duración y likes
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\colla\Downloads\TiktokScreenshots")

IN_CSV = BASE_DIR / "ocr_extracted_lives.csv"
OUT_DATASET = BASE_DIR / "tiktok_live_dataset_2025.csv"
OUT_DAILY = BASE_DIR / "daily_summary_2025.csv"
OUT_MONTHLY = BASE_DIR / "monthly_summary_2025.csv"

# ⚠️ Cambia a 2025 si quieres forzar todo a ese año
FORCE_YEAR = None

def main():
    # --- Leer OCR ---
    df = pd.read_csv(IN_CSV)

    # --- Convertir fecha ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if FORCE_YEAR:
        df["date"] = df["date"].apply(lambda d: d.replace(year=FORCE_YEAR) if pd.notna(d) else d)

    # --- Construir datetime ---
    df["start"] = df["start"].astype(str)
    df["end"] = df["end"].astype(str)
    df["start_dt"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + df["start"], errors="coerce")
    df["end_dt"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + df["end"], errors="coerce")

    # --- Calcular duración ---
    duration = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 60
    df["duration_minutes"] = duration.mask(duration < 0, 0).fillna(0).round(0).astype(int)

    # --- Likes ---
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).round(0).astype(int)

    # --- Extra: mes, semana, día ---
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["week"] = df["date"].dt.isocalendar().week.astype("Int64").fillna(0).astype(int)
    df["weekday"] = df["date"].dt.day_name()

    # --- Formato de salida ---
    df["date_str"] = df["date"].dt.date.astype(str)
    df["start_dt_str"] = df["start_dt"].dt.strftime("%Y-%m-%d %H:%M")
    df["end_dt_str"] = df["end_dt"].dt.strftime("%Y-%m-%d %H:%M")

    dataset = df[[
        "date_str","start_dt_str","end_dt_str","duration_minutes",
        "likes","month","week","weekday"
    ]].rename(columns={
        "date_str": "date",
        "start_dt_str": "start_dt",
        "end_dt_str": "end_dt",
    })

    # --- Resúmenes diarios y mensuales ---
    daily = df.groupby(df["date"].dt.date.astype(str), as_index=False).agg(
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

    # --- Tipos ---
    daily["lives_count"] = daily["lives_count"].astype(int)
    daily["total_likes"] = daily["total_likes"].astype(int)
    monthly["lives_count"] = monthly["lives_count"].astype(int)
    monthly["total_likes"] = monthly["total_likes"].astype(int)

    # --- Guardar ---
    dataset.to_csv(OUT_DATASET, index=False, encoding="utf-8-sig")
    daily.to_csv(OUT_DAILY, index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_MONTHLY, index=False, encoding="utf-8-sig")

    print(f"[OK] Generados:\n- {OUT_DATASET}\n- {OUT_DAILY}\n- {OUT_MONTHLY}")

if __name__ == "__main__":
    main()
