"""
TikTok Live OCR (Layout-aware v2)
- Fecha: detecta 'MES' + 'DIA' en la banda izquierda (aunque estén en líneas distintas)
- Horas: detecta 'HH:MM - HH:MM'
- Likes: ANCLADO a la palabra 'likes' / 'me gusta'; normaliza '24 . 8 K', '171 K', '9,572'
- Evita confundir horas/días con likes
- Guarda ocr_extracted_lives.csv junto a las imágenes

Autor: jeanpaul2102
"""

from pathlib import Path
from datetime import datetime, date, time
from typing import List, Dict, Optional
import re

import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import pytesseract

# ⚠️ Ruta de Tesseract (ajusta si es necesario)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Rutas ---
ROOT = Path(r"C:\Users\colla\Downloads\TiktokScreenshots")
OUT_CSV = ROOT / "ocr_extracted_lives.csv"

# --- Regex base ---
TIME_RGX = re.compile(r"\b(?P<start>\d{1,2}:\d{2})\b\s*[–—\-]\s*\b(?P<end>\d{1,2}:\d{2})\b")
LIKES_HINT_RGX = re.compile(r"(likes?|me\s*gusta|me\s*gustas?)", re.IGNORECASE)

# token de número con miles/decimales y K opcional (con o sin espacio), usado ANCLADO a 'likes'
LIKES_ANCHORED_RGX = re.compile(
    r"(?P<val>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?(?:\s*[kK])?)\s*(?:likes?|me\s*gusta|me\s*gustas?)",
    re.IGNORECASE,
)
# token “suelto” por si hay que hacer fallback (ya normalizado con K opcional)
LIKES_ANY_TOKEN_RGX = re.compile(r"(?P<val>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?(?:\s*[kK])?)\b")

# Meses EN/ES
MONTHS_MAP = {
    "jan": 1, "january": 1, "ene": 1, "enero": 1,
    "feb": 2, "february": 2, "febrero": 2,
    "mar": 3, "march": 3, "marzo": 3,
    "apr": 4, "april": 4, "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "june": 6, "junio": 6,
    "jul": 7, "july": 7, "julio": 7,
    "aug": 8, "august": 8, "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "september": 9, "septiembre": 9, "set": 9, "setiembre": 9,
    "oct": 10, "october": 10, "octubre": 10,
    "nov": 11, "november": 11, "noviembre": 11,
    "dec": 12, "december": 12, "dic": 12, "diciembre": 12,
}
MONTH_TOKEN_RGX = re.compile(r"^(jan|ene|feb|mar|apr|abr|may|jun|jul|aug|ago|sep|sept|set|oct|nov|dec|dic)$", re.I)
DAY_TOKEN_RGX = re.compile(r"^(0?[1-9]|[12]\d|3[01])$")

# ---------- Utilidades ----------
def preprocess(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.SHARPEN)
    return g

def ocr_dataframe(path: Path) -> pd.DataFrame:
    img = Image.open(path)
    img = preprocess(img)
    df = pytesseract.image_to_data(
        img, lang="eng+spa", config="--oem 3 --psm 6", output_type=pytesseract.Output.DATAFRAME
    )
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    return df

def month_to_num(s: str) -> Optional[int]:
    key = (
        s.lower()
        .replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
    )
    key3 = key[:3]
    return MONTHS_MAP.get(key3)

def normalize_likes(token: str) -> Optional[int]:
    if not token:
        return None
    t = token.lower().replace(" ", "")
    k = t.endswith("k")
    if k:
        t = t[:-1]
    # normaliza separadores
    if "," in t and "." in t:
        t = t.replace(",", "")
    else:
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\,\d+)?", t):
            t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
    try:
        val = float(t)
    except ValueError:
        digits = re.sub(r"\D", "", t)
        if not digits:
            return None
        val = float(digits)
    if k:
        val *= 1000
    return int(round(val))

def to_time(hhmm: str) -> Optional[time]:
    try:
        return datetime.strptime(hhmm, "%H:%M").time()
    except Exception:
        return None

def duration_minutes(t1: Optional[time], t2: Optional[time]) -> Optional[int]:
    if not t1 or not t2:
        return None
    dt1 = datetime.combine(date.today(), t1)
    dt2 = datetime.combine(date.today(), t2)
    if dt2 < dt1:
        dt2 = dt2.replace(day=dt2.day + 1)
    return int((dt2 - dt1).total_seconds() // 60)

# ---------- Fecha por layout (banda izquierda) ----------
def find_date_blocks(ts: pd.DataFrame) -> List[Dict]:
    if ts.empty:
        return []
    img_right = (ts["left"] + ts["width"]).max()
    left_band = ts[ts["left"] < img_right * 0.45].copy()

    months = left_band[left_band["text"].str.fullmatch(MONTH_TOKEN_RGX)].copy()
    days = left_band[left_band["text"].str.fullmatch(DAY_TOKEN_RGX)].copy()
    if months.empty or days.empty:
        return []

    blocks = []
    for _, mrow in months.iterrows():
        mnum = month_to_num(mrow["text"])
        if not mnum:
            continue
        my = mrow["top"] + mrow["height"] / 2.0

        cand = days[(days["top"] >= mrow["top"] - 220) & (days["top"] <= mrow["top"] + 280)].copy()
        if cand.empty:
            continue
        cand["yc"] = cand["top"] + cand["height"]/2.0
        cand["dist"] = (cand["yc"] - my).abs()
        drow = cand.sort_values("dist").iloc[0]
        day = int(drow["text"])
        try:
            iso = date(datetime.now().year, mnum, day).isoformat()
        except Exception:
            continue
        blocks.append({"y": (my + drow["yc"]) / 2.0, "month": mnum, "day": day, "iso": iso})

    blocks.sort(key=lambda b: b["y"])
    return blocks

# ---------- Likes anclados a la palabra 'likes' ----------
def parse_likes_from_line(text_line: str) -> Optional[int]:
    s = text_line

    # Normalización de errores típicos del OCR:
    # ' O likes' -> ' 0 likes'
    s = re.sub(r"\bO\b(?=\s*likes?)", "0", s, flags=re.I)
    # Unifica '24 . 8  K' => '24.8K'
    s = re.sub(r"(\d{1,3})\s*\.\s*(\d+)\s*[kK]\b", r"\1.\2K", s)
    # Unifica '171  K' => '171K'
    s = re.sub(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*[kK]\b", r"\1K", s)

    # 1) Intento fuerte: número justo antes de 'likes'
    last = None
    for m in LIKES_ANCHORED_RGX.finditer(s):
        last = m  # me quedo con el último por si repite
    if last:
        return normalize_likes(last.group("val"))

    # 2) Fallback: eliminar horarios y día inicial; luego buscar el MAYOR token
    s2 = re.sub(r"\b\d{1,2}:\d{2}\b", " ", s)     # quita horas
    s2 = re.sub(r"^\s*\d{1,2}\b", " ", s2)        # quita día inicial ('31 20:..')
    cand = [normalize_likes(m.group("val")) for m in LIKES_ANY_TOKEN_RGX.finditer(s2)]
    cand = [c for c in cand if c is not None]
    return max(cand) if cand else None

# ---------- Detección de líneas con horas ----------
def extract_time_rows(ts: pd.DataFrame) -> List[Dict]:
    rows = []
    if ts.empty:
        return rows

    for (page, block, par, line), df_line in ts.groupby(["page_num","block_num","par_num","line_num"]):
        text_line = " ".join(df_line["text"].tolist())
        m = TIME_RGX.search(text_line)
        if not m:
            continue
        start_s, end_s = m.group("start"), m.group("end")
        likes = parse_likes_from_line(text_line)
        y_center = (df_line["top"] + df_line["height"]/2.0).mean()
        rows.append({"y": y_center, "start": start_s, "end": end_s, "likes": likes, "raw": text_line})
    return rows

def assign_nearest_date(rows: List[Dict], blocks: List[Dict]) -> None:
    if not blocks:
        for r in rows:
            r["date"] = None
        return
    for r in rows:
        best = min(blocks, key=lambda b: abs(b["y"] - r["y"]))
        r["date"] = best["iso"]

# ---------- Proceso por imagen ----------
def process_image(path: Path) -> List[Dict]:
    ts = ocr_dataframe(path)
    date_blocks = find_date_blocks(ts)
    time_rows = extract_time_rows(ts)
    assign_nearest_date(time_rows, date_blocks)

    out = []
    for r in time_rows:
        out.append({
            "source_file": str(path),
            "date": r.get("date"),
            "start": r["start"],
            "end": r["end"],
            "likes": r["likes"],
            "duration_min": duration_minutes(to_time(r["start"]), to_time(r["end"])),
            "ocr_line": r["raw"],
        })

    # Si no hubo filas de tiempo, intenta likes global + fecha del primer bloque
    if not out:
        whole = " ".join(ts["text"].tolist())
        likes = parse_likes_from_line(whole) if LIKES_HINT_RGX.search(whole) else None
        out.append({
            "source_file": str(path),
            "date": date_blocks[0]["iso"] if date_blocks else None,
            "start": None,
            "end": None,
            "likes": likes,
            "duration_min": None,
            "ocr_line": None,
        })
    return out

def collect_images(root: Path) -> List[Path]:
    files = []
    for pat in ("*.jpg","*.jpeg","*.png"):
        files.extend(root.rglob(pat))
    return sorted(files)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    imgs = collect_images(ROOT)
    print(f"Imágenes encontradas: {len(imgs)}")

    for p in imgs:
        try:
            rows.extend(process_image(p))
        except Exception as e:
            rows.append({
                "source_file": str(p),
                "date": None, "start": None, "end": None,
                "likes": None, "duration_min": None,
                "ocr_line": f"[ERROR OCR] {e}",
            })

    df = pd.DataFrame(rows)

    # Orden y deduplicación ligera (por seguridad)
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values(["date_parsed","source_file","start","end"], inplace=True, ignore_index=True)
        df.drop(columns=["date_parsed"], inplace=True)

    # Dedup por combinación clave
    df.drop_duplicates(subset=["date","start","end","likes","source_file"], inplace=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] Registros: {len(df)}  |  Guardado: {OUT_CSV}")

if __name__ == "__main__":
    main()
