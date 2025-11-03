# compat_columns.py
from __future__ import annotations
import re
import pandas as pd
from pathlib import Path

# === 1) คำจำกัดความคอลัมน์มาตรฐานที่แอป "ต้องการ" ให้มี ===
STANDARD_COLS = [
    "รหัสหัวข้อ",
    "หัวข้อ",
    "วัน-เวลา ที่เกิดเหตุ",
    "ระดับความรุนแรง",
    "สรุปปัญหา/เหตุการณ์โดยย่อ",
    "กลุ่มอุบัติการณ์",
    "หมวด",
]

# === 2) ชุด alias ที่รองรับสำหรับแต่ละคอลัมน์ (ทั้งไทย/อังกฤษ/ชื่อเดิม) ===
ALIASES = {
    "รหัสหัวข้อ": [
        r"^รหัสหัวข้อ$",
        r"^รหัส$",
        r"^incident$",
        r"^incident\s*id$",
        r"^code$",
        r"^incident code$",
        r"^เลขที่รายงาน$",
    ],
    "หัวข้อ": [
        r"^หัวข้อ$",
        r"^ชื่ออุบัติการณ์ความเสี่ยง$",
        r"^incident\s*name$",
        r"^title$",
        r"^event\s*title$",
        r"^หัวข้ออุบัติการณ์$",
    ],
    "วัน-เวลา ที่เกิดเหตุ": [
        r"^วัน[-\s]*เวลา\s*ที่เกิดเหตุ$",
        r"^วันที่เกิด$",
        r"^วดป\.ที่เกิด$",
        r"^occurrence\s*date$",
        r"^event\s*date$",
        r"^date$",
        r"^datetime$",
    ],
    "ระดับความรุนแรง": [
        r"^ระดับความรุนแรง$",
        r"^impact$",
        r"^impact\s*level$",
        r"^severity$",
        r"^severity\s*level$",
        r"^ความรุนแรง$",
    ],
    "สรุปปัญหา/เหตุการณ์โดยย่อ": [
        r"^สรุปปัญหา/เหตุการณ์โดยย่อ$",
        r"^รายละเอียดการเกิด$",
        r"^รายละเอียด$",
        r"^description$",
        r"^summary$",
        r"^เหตุการณ์$",
        r"^รายละเอียดเหตุการณ์$",
    ],
    "กลุ่มอุบัติการณ์": [
        r"^กลุ่มอุบัติการณ์$",
        r"^กลุ่ม$",
        r"^group$",
        r"^incident\s*group$",
    ],
    "หมวด": [
        r"^หมวด$",
        r"^หมวดหมู่$",
        r"^category$",
        r"^incident\s*category$",
    ],
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def _match_col(colname: str, patterns: list[str]) -> bool:
    c = _norm(colname)
    for p in patterns:
        if re.compile(p, re.I).match(c):
            return True
    return False

def _find_first(df: pd.DataFrame, patterns: list[str]) -> str | None:
    for col in df.columns:
        if _match_col(col, patterns):
            return col
    return None

def normalize_dataframe_columns(raw_df: pd.DataFrame, allcode_path: str | None = None):
    """
    คืนค่า:
      df_norm  : DataFrame ที่รีเนมคอลัมน์เป็นชื่อมาตรฐานชุดไทย
      missing  : รายการคอลัมน์มาตรฐานที่ยังไม่มีข้อมูล (จะถูกเติมค่าว่างเอาไว้ใช้งานต่อได้)
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), STANDARD_COLS[:]  # ทั้งหมดขาด

    df = raw_df.copy()
    col_map = {}

    # 1) หาและรีเนมทีละคอลัมน์ตาม ALIASES
    for std_col, patterns in ALIASES.items():
        found = _find_first(df, patterns)
        if found:
            col_map[found] = std_col

    if col_map:
        df = df.rename(columns=col_map)

    # 2) สร้างคอลัมน์มาตรฐานที่ยังไม่มี ให้เป็นค่าว่าง
    for std in STANDARD_COLS:
        if std not in df.columns:
            df[std] = ""

    # 3) แปลงค่าที่พบบ่อย
    # 3.1 วัน-เวลา ที่เกิดเหตุ -> datetime
    if "วัน-เวลา ที่เกิดเหตุ" in df.columns:
        df["วัน-เวลา ที่เกิดเหตุ"] = pd.to_datetime(df["วัน-เวลา ที่เกิดเหตุ"], errors="coerce")

    # 3.2 ระดับความรุนแรง: map A–I/1–5 ให้คงที่ (ถ้าจำเป็น)
    if "ระดับความรุนแรง" in df.columns:
        df["ระดับความรุนแรง"] = df["ระดับความรุนแรง"].astype(str).str.strip().str.upper()
        # แปลงตัวเลขไทย/รูปแบบสะกด
        replacements = {
            "AB": "A-B",
            "CD": "C-D",
            "EF": "E-F",
            "GH": "G-H",
            "I(5)": "I",
        }
        df["ระดับความรุนแรง"] = df["ระดับความรุนแรง"].replace(replacements)

    # 4) เติม กลุ่มอุบัติการณ์/หมวด จาก Code2024.xlsx หากว่าง และมีพาธ
    if allcode_path:
        p = Path(str(allcode_path))
        if p.exists() and p.is_file():
            try:
                code = pd.read_excel(p)
                # พยายามหา field รหัส/กลุ่ม/หมวด
                # รองรับหัวข้อไทย/อังกฤษพื้นฐาน
                key_col = None
                for cand in ["รหัส", "code", "incident", "incident code", "รหัสหัวข้อ"]:
                    if cand in code.columns:
                        key_col = cand
                        break
                if key_col is None:
                    # เดาแบบหลวม ๆ
                    key_col = code.columns[0]

                # กลุ่ม
                grp_col = None
                for cand in ["กลุ่ม", "กลุ่มอุบัติการณ์", "group", "incident group"]:
                    if cand in code.columns:
                        grp_col = cand
                        break
                # หมวด
                cat_col = None
                for cand in ["หมวด", "หมวดหมู่", "category", "incident category"]:
                    if cand in code.columns:
                        cat_col = cand
                        break

                # เตรียมคีย์จาก df (ใช้ รหัสหัวข้อ เป็นหลัก ถ้าไม่มีใช้คอลัมน์เดิมที่แมตช์ได้)
                if "รหัสหัวข้อ" not in df.columns or df["รหัสหัวข้อ"].eq("").all():
                    # ลองยกค่าจากคอลัมน์ที่เคยเป็น 'รหัส' (ก่อนรีเนม)
                    for c in raw_df.columns:
                        if _match_col(c, ALIASES["รหัสหัวข้อ"]):
                            df["รหัสหัวข้อ"] = raw_df[c].astype(str).str.strip()
                            break
                # ทำคีย์ให้สะอาด
                df["รหัสหัวข้อ"] = df["รหัสหัวข้อ"].astype(str).str.strip()
                code[key_col] = code[key_col].astype(str).str.strip()

                look = code[[key_col] + [c for c in [grp_col, cat_col] if c]].drop_duplicates()

                df = df.merge(look, left_on="รหัสหัวข้อ", right_on=key_col, how="left")
                if grp_col and "กลุ่มอุบัติการณ์" in df.columns:
                    df["กลุ่มอุบัติการณ์"] = df["กลุ่มอุบัติการณ์"].mask(
                        df["กลุ่มอุบัติการณ์"].astype(str).str.strip().eq("")
                    , df[grp_col])
                if cat_col and "หมวด" in df.columns:
                    df["หมวด"] = df["หมวด"].mask(
                        df["หมวด"].astype(str).str.strip().eq("")
                    , df[cat_col])

                # ลบคอลัมน์ช่วย
                if key_col in df.columns:
                    df = df.drop(columns=[key_col], errors="ignore")
                if grp_col:
                    df = df.drop(columns=[grp_col], errors="ignore")
                if cat_col:
                    df = df.drop(columns=[cat_col], errors="ignore")
            except Exception:
                # ถ้าอ่าน codebook ไม่ได้ ก็ข้ามไป
                pass

    # 5) คำนวณ missing (คอลัมน์ที่ยังว่างทั้งหมด)
    missing = []
    for std in STANDARD_COLS:
        if std not in df.columns or df[std].isna().all() or df[std].astype(str).str.strip().eq("").all():
            missing.append(std)

    # 6) จัดลำดับคอลัมน์ให้อยู่หน้าตามมาตรฐาน
    front = [c for c in STANDARD_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    return df, missing
