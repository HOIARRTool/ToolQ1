# -*- coding: utf-8 -*-
"""
compat_columns.py

ปรับหัวคอลัมน์จากไฟล์ของผู้ใช้ให้เข้ากับสคีมาตัวแอป
และ (ถ้ามี) merge Code2024.xlsx เพื่อเติมชื่อ/กลุ่ม/หมวด

รองรับหัวตารางชุดนี้เท่านั้น:
In.HCode, วดป.ที่ Import การเกิด, รหัสรายงาน, Incident, ความรุนแรง, สถานะ, ผู้ได้รับผลกระทบ,
ชนิดสถานที่, วดป.ที่เกิด, ช่วงเวลา/เวร, รายละเอียดการเกิด, วดป. ที่ Import การแก้, วดป. ที่แก้ไข,
Resulting Actions, ผลลัพธ์ทางสังคม
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, List

RENAME_MAP = {
    "Incident": "Incident",
    "วดป.ที่เกิด": "Occurrence Date",
    "ความรุนแรง": "Impact",
    "รายละเอียดการเกิด": "รายละเอียดการเกิด",
    "Resulting Actions": "Resulting Actions",
    "ช่วงเวลา/เวร": "ช่วงเวลา/เวร",
    "ชนิดสถานที่": "ชนิดสถานที่",
}

MIN_OUTPUT_COLS = [
    "Incident","Occurrence Date","Impact","รายละเอียดการเกิด","Resulting Actions",
    "ช่วงเวลา/เวร","ชนิดสถานที่","รหัส","ชื่ออุบัติการณ์ความเสี่ยง","กลุ่ม","หมวด","Impact Level"
]

def _build_impact_level(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    map_ai = {"A":"1","B":"1","C":"2","D":"2","E":"3","F":"3","G":"4","H":"4","I":"5"}
    def _map_one(x):
        if x in map_ai: return map_ai[x]
        if x.isdigit() and x in {"1","2","3","4","5"}: return x
        return "N/A"
    return s.map(_map_one)

def _load_codebook(path: str) -> pd.DataFrame:
    p = Path(path) if path else None
    if not p or not p.is_file():
        return pd.DataFrame(columns=["รหัส","ชื่ออุบัติการณ์ความเสี่ยง","กลุ่ม","หมวด"])
    df = pd.read_excel(p)
    cols = ["รหัส","ชื่ออุบัติการณ์ความเสี่ยง","กลุ่ม","หมวด"]
    for c in cols:
        if c not in df.columns:
            return pd.DataFrame(columns=cols)
    out = df[cols].drop_duplicates().copy()
    out["รหัส"] = out["รหัส"].astype(str).str.strip()
    return out

def normalize_dataframe_columns(raw_df, allcode_path=None):
    # กันกรณีรับ None เข้ามา
    if raw_df is None:
        return pd.DataFrame(), ["no_data"]

    # ถ้าเป็น DataFrame เปล่า
    if hasattr(raw_df, "empty") and raw_df.empty:
        return pd.DataFrame(), ["empty_data"]

    # ทำงานต่อได้
    df = raw_df.copy()

    # รีเนมคอลัมน์ที่ใช้งานจริง
    df = df.rename(columns=RENAME_MAP)

    # วันเกิดเหตุ
    if "Occurrence Date" in df.columns:
        df["Occurrence Date"] = pd.to_datetime(df["Occurrence Date"], errors="coerce")

    # สร้าง 'รหัส' จาก Incident (6 ตัวแรก)
    if "Incident" in df.columns:
        df["Incident"] = df["Incident"].astype(str).str.strip()
        df["รหัส"] = df["Incident"].str[:6].str.strip()

    # Impact Level
    if "Impact" in df.columns:
        df["Impact"] = df["Impact"].astype(str).str.strip().str.upper()
        df["Impact Level"] = _build_impact_level(df["Impact"])
    else:
        df["Impact Level"] = "N/A"

    # บังคับคอลัมน์ข้อความ
    for col in ["รายละเอียดการเกิด","Resulting Actions","ช่วงเวลา/เวร","ชนิดสถานที่"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # เตรียมชื่อ/กลุ่ม/หมวด
    df["ชื่ออุบัติการณ์ความเสี่ยง"] = "N/A"
    df["กลุ่ม"] = "N/A"
    df["หมวด"] = "N/A"

    codebook = _load_codebook(allcode_path)
    if not codebook.empty and "รหัส" in df.columns:
        df = df.merge(codebook, on="รหัส", how="left", suffixes=("","_codebook"))
        for c in ["ชื่ออุบัติการณ์ความเสี่ยง","กลุ่ม","หมวด"]:
            df[c] = df[c].fillna("N/A")

    # Missing (เพื่อแจ้งเตือนเบา ๆ)
    output_missing = [c for c in MIN_OUTPUT_COLS if c not in df.columns]

    # จัดลำดับ
    front = ["Occurrence Date","Incident","รหัส","ชื่ออุบัติการณ์ความเสี่ยง",
             "Impact","Impact Level","รายละเอียดการเกิด","Resulting Actions",
             "ช่วงเวลา/เวร","ชนิดสถานที่","กลุ่ม","หมวด"]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]

    return df, output_missing
