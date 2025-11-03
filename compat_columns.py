# -*- coding: utf-8 -*-
"""
compat_columns.py
ยูทิลิตี้สำหรับทำให้ DataFrame จากไฟล์ .xlsx รุ่นเก่า-ใหม่ “เข้ากันได้”
- map คอลัมน์เดิม -> คอลัมน์มาตรฐานที่แอปต้องใช้
- เติม 'กลุ่มอุบัติการณ์' และ 'หมวด' โดย merge กับ Code2024.xlsx ถ้าระบุพาธ
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# คีย์มาตรฐานที่ต้องมี
REQUIRED_COLS = [
    "รหัสหัวข้อ",
    "หัวข้อ",
    "วัน-เวลา ที่เกิดเหตุ",
    "ระดับความรุนแรง",
    "สรุปปัญหา/เหตุการณ์โดยย่อ",
    "กลุ่มอุบัติการณ์",
    "หมวด",
]

# ชื่อคอลัมน์แบบต่าง ๆ ที่พบได้ในไฟล์เดิม ๆ -> ค่าเป้าหมาย
CANDIDATE_MAP: Dict[str, List[str]] = {
    "รหัสหัวข้อ": ["รหัสหัวข้อ", "Incident", "รหัส", "Incident Code", "Code"],
    "หัวข้อ": ["หัวข้อ", "ชื่ออุบัติการณ์ความเสี่ยง", "Incident Name", "Topic"],
    "วัน-เวลา ที่เกิดเหตุ": ["วัน-เวลา ที่เกิดเหตุ", "Occurrence Date", "วันที่เกิด", "วันเวลา"],
    "ระดับความรุนแรง": ["ระดับความรุนแรง", "Impact", "Impact Level", "ความรุนแรง"],
    "สรุปปัญหา/เหตุการณ์โดยย่อ": ["สรุปปัญหา/เหตุการณ์โดยย่อ", "รายละเอียดการเกิด", "รายละเอียด", "Summary"],
    "กลุ่มอุบัติการณ์": ["กลุ่มอุบัติการณ์", "กลุ่ม", "Group"],
    "หมวด": ["หมวด", "หมวดหมู่", "Category"],
}

def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_dataframe_columns(df: pd.DataFrame, allcode_path: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    รับ DataFrame ดิบ -> คืน DataFrame ที่มีคอลัมน์มาตรฐานครบเท่าที่ทำได้
    และลิสต์ชื่อคอลัมน์ที่ยังขาด
    - ถ้า allcode_path ชี้ไปยัง Code2024.xlsx จะ merge เพื่อเติม 'กลุ่มอุบัติการณ์', 'หมวด'
      โดยอ้างอิง key 'รหัสหัวข้อ' -> คอลัมน์ 'รหัส' ใน Code2024.xlsx
    """
    df_norm = df.copy()

    # 1) สร้างคอลัมน์มาตรฐานทีละตัวจาก candidates
    for target, candidates in CANDIDATE_MAP.items():
        if target in df_norm.columns:
            continue
        src = _first_present(df_norm, candidates)
        if src is not None:
            df_norm[target] = df_norm[src]
        else:
            # สร้างเป็นค่าว่างไว้ก่อน
            df_norm[target] = pd.NA

    # 2) จัดชนิดวันที่ให้ 'วัน-เวลา ที่เกิดเหตุ' ถ้าเป็นสตริง
    if df_norm["วัน-เวลา ที่เกิดเหตุ"].notna().any():
        df_norm["วัน-เวลา ที่เกิดเหตุ"] = pd.to_datetime(
            df_norm["วัน-เวลา ที่เกิดเหตุ"], errors="coerce"
        )

    # 3) เติมกลุ่ม/หมวดจาก Code2024.xlsx หากให้พาธมา
    if allcode_path:
        p = Path(allcode_path)
        if p.is_file():
            try:
                code_df = pd.read_excel(p)
                # คาดว่ามีคอลัมน์ 'รหัส', 'กลุ่ม', 'หมวด'
                expected_cols = {"รหัส", "กลุ่ม", "หมวด"}
                if expected_cols.issubset(set(code_df.columns)):
                    code = code_df[["รหัส", "กลุ่ม", "หมวด"]].drop_duplicates()
                    # ทำคีย์ให้เทียบกันได้
                    df_norm["__key__"] = df_norm["รหัสหัวข้อ"].astype(str).str.strip()
                    code["__key__"] = code["รหัส"].astype(str).str.strip()
                    df_norm = df_norm.merge(code[["__key__", "กลุ่ม", "หมวด"]], on="__key__", how="left", suffixes=("", "_from_code"))
                    # ถ้าเดิมไม่มีค่า ให้ใช้จาก codebook
                    for col_src, col_new in [("กลุ่มอุบัติการณ์", "กลุ่ม"), ("หมวด", "หมวด")]:
                        if col_src in df_norm.columns and col_new in df_norm.columns:
                            df_norm[col_src] = df_norm[col_src].fillna(df_norm[col_new])
                    df_norm.drop(columns=[c for c in ["__key__", "กลุ่ม", "หมวด"] if c in df_norm.columns], inplace=True)
                # ถ้าไฟล์ไม่ตรง schema ก็ปล่อยผ่านแบบไม่ทำอะไร
            except Exception:
                pass

    # 4) ระบุรายชื่อที่ยังขาด
    missing = [c for c in REQUIRED_COLS if c not in df_norm.columns or df_norm[c].isna().all()]
    return df_norm, missing
