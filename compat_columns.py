# compat_columns.py
from __future__ import annotations
import pandas as pd
from typing import Tuple, List, Dict, Optional

# คอลัมน์มาตรฐานที่แอป "รุ่นใหม่" ต้องมี
REQUIRED = [
    "รหัสหัวข้อ",
    "หัวข้อ",
    "วัน-เวลา ที่เกิดเหตุ",
    "ระดับความรุนแรง",
    "สรุปปัญหา/เหตุการณ์โดยย่อ",
    "กลุ่มอุบัติการณ์",
    "หมวด",
]

# ชื่อคอลัมน์ที่พบบ่อย (ไทย/อังกฤษ/แบบเก่า) -> ชื่อมาตรฐาน
COLUMN_ALIASES: Dict[str, str] = {
    # รหัส
    "รหัส": "รหัสหัวข้อ",
    "incident": "รหัสหัวข้อ",
    "incident code": "รหัสหัวข้อ",
    "code": "รหัสหัวข้อ",

    # ชื่อเหตุการณ์/หัวข้อ
    "ชื่ออุบัติการณ์ความเสี่ยง": "หัวข้อ",
    "หัวข้อ": "หัวข้อ",
    "incident name": "หัวข้อ",
    "title": "หัวข้อ",

    # วันเวลาเกิดเหตุ
    "occurrence date": "วัน-เวลา ที่เกิดเหตุ",
    "วัน-เวลา ที่เกิดเหตุ": "วัน-เวลา ที่เกิดเหตุ",
    "วดป.ที่เกิด": "วัน-เวลา ที่เกิดเหตุ",
    "วันที่เกิด": "วัน-เวลา ที่เกิดเหตุ",
    "date": "วัน-เวลา ที่เกิดเหตุ",
    "datetime": "วัน-เวลา ที่เกิดเหตุ",

    # ระดับความรุนแรง (ตัวอักษร A-I หรือ 1-5 ก็ได้)
    "impact": "ระดับความรุนแรง",
    "ความรุนแรง": "ระดับความรุนแรง",
    "impact level": "ระดับความรุนแรง",
    "ระดับ": "ระดับความรุนแรง",

    # สรุปเหตุการณ์/รายละเอียด
    "รายละเอียดการเกิด": "สรุปปัญหา/เหตุการณ์โดยย่อ",
    "รายละเอียด": "สรุปปัญหา/เหตุการณ์โดยย่อ",
    "สรุปปัญหา/เหตุการณ์โดยย่อ": "สรุปปัญหา/เหตุการณ์โดยย่อ",
    "description": "สรุปปัญหา/เหตุการณ์โดยย่อ",
    "summary": "สรุปปัญหา/เหตุการณ์โดยย่อ",

    # กลุ่ม/หมวด
    "กลุ่ม": "กลุ่มอุบัติการณ์",
    "group": "กลุ่มอุบัติการณ์",
    "หมวด": "หมวด",
    "category": "หมวด",
}

# สร้าง alias แบบย้อนกลับเพื่อ “เติมคอลัมน์ legacy” ให้โค้ดเก่ายังรันได้
LEGACY_BACKFILL = {
    "รหัส": "รหัสหัวข้อ",
    "Incident": "รหัสหัวข้อ",
    "ชื่ออุบัติการณ์ความเสี่ยง": "หัวข้อ",
    "Occurrence Date": "วัน-เวลา ที่เกิดเหตุ",
    "Impact": "ระดับความรุนแรง",
    "รายละเอียดการเกิด": "สรุปปัญหา/เหตุการณ์โดยย่อ",
    "กลุ่ม": "กลุ่มอุบัติการณ์",
    "หมวด": "หมวด",
}

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED)

    # ทำชื่อคอลัมน์ให้เรียบก่อน
    new_cols = []
    for c in df.columns:
        c_norm = str(c).strip()
        c_key = c_norm.lower().replace("\n", " ").replace("\r", " ").replace("  ", " ").strip()
        mapped = COLUMN_ALIASES.get(c_key, None)
        new_cols.append(mapped if mapped else c_norm)
    df = df.copy()
    df.columns = new_cols

    # ถ้าคอลัมน์มาตรฐานยังไม่ครบ ให้เติมคอลัมน์ว่างเข้ามา
    for col in REQUIRED:
        if col not in df.columns:
            df[col] = pd.NA

    # จัดชุดคอลัมน์ให้อยู่หน้าสุดก่อน แล้วตามด้วยอื่นๆ
    front = [c for c in REQUIRED]
    tail = [c for c in df.columns if c not in front]
    df = df[front + tail]
    return df

def _merge_codebook(df: pd.DataFrame, allcode_path: Optional[str]) -> pd.DataFrame:
    """เติมกลุ่ม/หมวด จาก Code2024.xlsx ถ้าให้ path มาและอ่านได้"""
    if not allcode_path:
        return df
    try:
        code = pd.read_excel(allcode_path)
    except Exception:
        return df

    # หา key สำหรับ merge โดยยึดคอลัมน์รหัสใน codebook
    code = code.copy()
    # พยายาม normalize ชื่อ header ของ codebook
    code_cols = {str(c).strip(): c for c in code.columns}
    key_code = None
    for cand in ["รหัส", "code", "incident", "incident code"]:
        cand_real = code_cols.get(cand, None)
        if cand_real is not None:
            key_code = cand_real
            break
    if key_code is None:
        return df

    # หา "กลุ่ม" และ "หมวด" ใน codebook
    group_col = None
    cat_col = None
    for g in ["กลุ่ม", "group"]:
        if g in code_cols:
            group_col = code_cols[g]; break
    for k in ["หมวด", "category"]:
        if k in code_cols:
            cat_col = code_cols[k]; break

    # ถ้าไม่มีคอลัมน์กลุ่ม/หมวด ก็ไม่มีอะไรให้เติม
    if group_col is None and cat_col is None:
        return df

    # สร้างคอลัมน์ 'รหัสหัวข้อ' ใน df ถ้ายังไม่มี
    if "รหัสหัวข้อ" not in df.columns:
        return df

    code[key_code] = code[key_code].astype(str).str.strip()
    df["_merge_key"] = df["รหัสหัวข้อ"].astype(str).str.strip()

    keep_cols = [key_code]
    if group_col: keep_cols.append(group_col)
    if cat_col: keep_cols.append(cat_col)

    code_small = code[keep_cols].drop_duplicates()

    merged = df.merge(code_small, left_on="_merge_key", right_on=key_code, how="left")
    merged.drop(columns=[c for c in [key_code, "_merge_key"] if c in merged.columns], inplace=True)

    if group_col and "กลุ่มอุบัติการณ์" in merged.columns:
        merged["กลุ่มอุบัติการณ์"] = merged["กลุ่มอุบัติการณ์"].fillna(merged[group_col])
        merged.drop(columns=[group_col], inplace=True, errors="ignore")
    if cat_col and "หมวด" in merged.columns:
        merged["หมวด"] = merged["หมวด"].fillna(merged[cat_col])
        merged.drop(columns=[cat_col], inplace=True, errors="ignore")

    return merged

def _coerce_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "วัน-เวลา ที่เกิดเหตุ" in df.columns:
        df["วัน-เวลา ที่เกิดเหตุ"] = pd.to_datetime(df["วัน-เวลา ที่เกิดเหตุ"], errors="coerce")
    return df

def _backfill_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """สร้างคอลัมน์ชื่อเดิมๆ ให้โค้ดส่วนเก่าที่อ้างถึงยังทำงานได้"""
    for legacy_col, std_col in LEGACY_BACKFILL.items():
        if legacy_col not in df.columns and std_col in df.columns:
            df[legacy_col] = df[std_col]
    return df

def normalize_dataframe_columns(df: pd.DataFrame, allcode_path: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    คืนค่า:
      - df_out: DataFrame ที่ normalize ชื่อคอลัมน์ -> มาตรฐาน + เติมกลุ่ม/หมวด + backfill legacy
      - missing: รายการคอลัมน์มาตรฐานที่ยังหายไป (หลัง normalize)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED), REQUIRED.copy()

    out = _normalize_headers(df)
    out = _merge_codebook(out, allcode_path=allcode_path)
    out = _coerce_datetime(out)
    out = _backfill_legacy(out)

    missing = [c for c in REQUIRED if c not in out.columns or out[c].isna().all()]
    return out, missing
