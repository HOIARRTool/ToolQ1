# compat_columns.py
import pandas as pd
import numpy as np

# ====== ค่ามาตรฐานที่แอปใช้หลัง normalize ======
STANDARD_COLS = [
    "รหัสหัวข้อ",                # from: รหัส
    "หัวข้อ",                    # from: Incident
    "วัน-เวลา ที่เกิดเหตุ",       # from: Occurrence Date
    "ระดับความรุนแรง",           # from: Impact Level (fallback: Impact/ความรุนแรง)
    "สรุปปัญหา/เหตุการณ์โดยย่อ",  # from: รายละเอียดการเกิด
    "กลุ่มอุบัติการณ์",
    "หมวด",
]

# ====== mapping จากหัวตารางของคุณ -> ชื่อมาตรฐาน ======
RENAME_MAP = {
    "รหัส": "รหัสหัวข้อ",
    "Incident": "หัวข้อ",
    "Occurrence Date": "วัน-เวลา ที่เกิดเหตุ",
    "Impact Level": "ระดับความรุนแรง",
    "Impact": "ระดับความรุนแรง",          # เผื่อบางไฟล์ไม่มี Impact Level
    "ความรุนแรง": "ระดับความรุนแรง",      # เผื่อหัวไทย
    "รายละเอียดการเกิด": "สรุปปัญหา/เหตุการณ์โดยย่อ",
    # คอลัมน์อื่น ๆ ที่คุณมี จะถูกเก็บไว้เหมือนเดิม ไม่ทับ
}

# เลือกคอลัมน์รหัสที่ใช้ merge กับ Code2024.xlsx
CODE_KEYS = ["รหัสหัวข้อ", "รหัส", "Code", "รหัสรายงาน"]

def _read_codebook(path: str) -> pd.DataFrame | None:
    try:
        cb = pd.read_excel(path)
    except Exception:
        return None
    if cb is None or cb.empty:
        return None

    # พยายาม normalize หัวตารางของ codebook ให้มี: 'รหัสหัวข้อ','กลุ่มอุบัติการณ์','หมวด'
    cb_cols = {c.strip(): c for c in cb.columns if isinstance(c, str)}
    # เครสที่ codebook ใช้หัว "รหัส"
    if "รหัสหัวข้อ" not in cb_cols and "รหัส" in cb_cols:
        cb.rename(columns={cb_cols["รหัส"]: "รหัสหัวข้อ"}, inplace=True)
    # เครสสะกดแตกต่าง
    for want, aliases in {
        "กลุ่มอุบัติการณ์": ["กลุ่ม", "กลุ่มเหตุการณ์", "ชื่ออุบัติการณ์ความเสี่ยง", "Risk Group", "Group"],
        "หมวด": ["หมวดหมู่", "Category"],
    }.items():
        if want not in cb.columns:
            for a in aliases:
                if a in cb.columns:
                    cb.rename(columns={a: want}, inplace=True)
                    break

    keep = [c for c in ["รหัสหัวข้อ", "กลุ่มอุบัติการณ์", "หมวด"] if c in cb.columns]
    if not keep or "รหัสหัวข้อ" not in keep:
        return None
    return cb[keep].drop_duplicates()

def normalize_dataframe_columns(raw_df: pd.DataFrame | None, allcode_path: str | None = None):
    """รับ DataFrame ที่มีหัวตารางแบบของผู้ใช้ แล้วรีเนม/เติมคอลัมน์ให้เป็นมาตรฐานที่แอปต้องการ
    คืนค่า: (df_norm, missing_cols_after)"""

    # กันเคสไม่มีข้อมูล
    if raw_df is None:
        return pd.DataFrame(), ["no_data"]
    if getattr(raw_df, "empty", False):
        return pd.DataFrame(), ["empty_data"]

    df = raw_df.copy()

    # 1) รีเนมคอลัมน์เท่าที่แมปได้
    # ทำงานแบบ case-insensitive เบา ๆ
    lower_map = {k.lower(): v for k, v in RENAME_MAP.items()}
    rename_dict = {}
    for c in df.columns:
        if isinstance(c, str) and c.strip().lower() in lower_map:
            rename_dict[c] = lower_map[c.strip().lower()]
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    # 2) จัดคอลัมน์ "ระดับความรุนแรง" (ถ้ายังไม่มี)
    if "ระดับความรุนแรง" not in df.columns:
        for alt in ["Impact Level", "Impact", "ความรุนแรง"]:
            if alt in raw_df.columns:
                df["ระดับความรุนแรง"] = raw_df[alt]
                break

    # 3) วันที่: แปลง "วัน-เวลา ที่เกิดเหตุ" ให้เป็น datetime ถ้ามี
    if "วัน-เวลา ที่เกิดเหตุ" in df.columns:
        try:
            df["วัน-เวลา ที่เกิดเหตุ"] = pd.to_datetime(df["วัน-เวลา ที่เกิดเหตุ"], errors="coerce")
        except Exception:
            pass

    # 4) เติมคอลัมน์ที่จำเป็นแต่ขาด ให้เป็นค่าว่างก่อน
    for col in STANDARD_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # 5) merge กลุ่ม/หมวดจาก Code2024.xlsx (ถ้ามี)
    if allcode_path:
        cb = _read_codebook(allcode_path)
        if cb is not None and not cb.empty:
            # หาคีย์สำหรับ join (ตามลำดับ)
            key = None
            for k in CODE_KEYS:
                if k in df.columns:
                    key = k
                    break
            if key is None:
                # บางไฟล์ df ไม่มี 'รหัสหัวข้อ' แต่มี 'รหัส' อยู่เดิม
                for k in ["รหัส", "รหัสรายงาน"]:
                    if k in raw_df.columns:
                        df["รหัสหัวข้อ"] = raw_df[k]
                        key = "รหัสหัวข้อ"
                        break
            if key is not None and "รหัสหัวข้อ" in cb.columns:
                # ทำให้ชื่อคอลัมน์คีย์ตรงกัน
                if key != "รหัสหัวข้อ":
                    df = df.rename(columns={key: "รหัสหัวข้อ"})
                df = df.merge(cb, on="รหัสหัวข้อ", how="left", suffixes=("", "_cb"))
                # ถ้าใน df เดิมมี 'กลุ่มอุบัติการณ์'/'หมวด' ให้อยู่; otherwise ใช้จาก codebook
                if "กลุ่มอุบัติการณ์_cb" in df.columns:
                    df["กลุ่มอุบัติการณ์"] = df["กลุ่มอุบัติการณ์"].fillna(df["กลุ่มอุบัติการณ์_cb"])
                if "หมวด_cb" in df.columns:
                    df["หมวด"] = df["หมวด"].fillna(df["หมวด_cb"])
                # ล้างคอลัมน์ _cb ช่วยความเรียบร้อย
                drop_cols = [c for c in df.columns if c.endswith("_cb")]
                if drop_cols:
                    df.drop(columns=drop_cols, inplace=True)

    # 6) จัดเรียงคอลัมน์มาตรฐานให้อยู่ต้น ๆ (ที่เหลือคงไว้)
    front = [c for c in STANDARD_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    # 7) สร้างรายการคอลัมน์ที่ "ยัง" ขาดหลัง normalize (เน้น 7 ช่องมาตรฐานเท่านั้น)
    missing_after = [c for c in STANDARD_COLS if df[c].isna().all()]

    return df, missing_after
