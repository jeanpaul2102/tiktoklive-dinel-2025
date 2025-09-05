"""
TikTok Live OCR Script
- Lee capturas de pantalla en data/raw_screenshots/
- Usa pytesseract para extraer texto
- Intenta detectar fecha, hora de inicio/fin y likes
- Genera data/ocr_extracted_lives.csv
"""
Author: jeanpaul2102
import os
import re
import pandas as pd
from pathlib import Path
from PIL import Image
import pytesseract

RAW_DIR = Path("data/raw_screenshots")
OUT_CSV = Path("data/ocr_extracted_lives.csv")

# Regex para tiempos y likes (ajustable según idioma de la app)
TIME_RGX = re.compile(r"(?P<start>\d{1,2}:\d{2})\s*[-–]\s*(?P<end>\d{1,2}:\d{2})")
LIKES_RGX = re.compile(r"(\d+[.,]?\d*[kK]?)\s*likes", re.IGNORECASE)

def normalize_likes(token: str) -> int:
    token = token.lower().replace(",", "").strip()
    if token.endswith("k"):
        return int(float(token[:-1]) * 1000)
    return int(float(token))

def extract_records_from_image(path: Path) -> list[dict]:
    img = Image.open(path)
    text = pytesseract.image_to_string(img, lang="eng")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    records = []
    for ln in lines:
        mtime = TIME_RGX.search(ln)
        if mtime:
            start = mtime.group("start")
            end = mtime.group("end")
            mlikes = LIKES_RGX.search(ln)
            likes = normalize_likes(mlikes.group(1)) if mlikes else None
            records.append({
                "date": None,  # se puede ajustar manualmente si no está en OCR
                "start": start,
                "end": end,
                "likes": likes
            })
    return records

def main():
    all_rows = []
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for f in RAW_DIR.glob("*.jpg"):
        all_rows.extend(extract_records_from_image(f))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Guardado: {OUT_CSV}")

if __name__ == "__main__":
    main()