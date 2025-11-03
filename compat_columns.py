# compat_columns.py
import re
import pandas as pd

def _coerce_code_str(series: pd.Series) -> pd.Series:
    """บังคับคีย์ให้เป็น string แบบสะอาด ใช้ได้กับทั้งตัวเลข/รหัส"""
    def _norm(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip()
        s = re.sub(r"\.0$", "", s)  # ตัด .0 ท้ายค่าที่มาจากเลขทศนิยม Excel
        s = s.replace(",", "")      # กันมี comma
        return s
    # ใช้ dtype 'string' เพื่อให้ NA เป็น <NA> ไม่ใช่ "nan"
    return series.map(_norm).astype("string")

def normalize_dataframe_columns(raw_df: pd.DataFrame, allcode_path: str | None = None):
    df = raw_df.copy()

    # ----- โค้ดแม็พชื่อคอลัมน์เดิมของคุณ -> คอลัมน์มาตรฐาน -----
    # ถ้ายังไม่มี "รหัสหัวข้อ" แต่มี "รหัสรายงาน" ให้ย้ายมาใช้เป็นคีย์
    if "รหัสหัวข้อ" not in df.columns and "รหัสรายงาน" in df.columns:
        df["รหัสหัวข้อ"] = df["รหัสรายงาน"]

    # บังคับให้คีย์เป็นสตริงสะอาดเสมอ
    if "รหัสหัวข้อ" in df.columns:
        df["รหัสหัวข้อ"] = _coerce_code_str(df["รหัสหัวข้อ"])

    missing_cols = []

    # ----- โหลด Code2024.xlsx แล้ว merge กลุ่ม/หมวด -----
    if allcode_path:
        try:
            cb = pd.read_excel(allcode_path, engine="openpyxl")
            # รองรับชื่อคอลัมน์ใน codebook ที่ต่างกันเล็กน้อย
            # ต้องมีคอลัมน์ 'รหัสหัวข้อ' ใน codebook เช่นกัน
            if "รหัสหัวข้อ" not in cb.columns:
                # ถ้าชื่อคีย์ใน codebook เป็น 'รหัสรายงาน' ก็โคลนมา
                if "รหัสรายงาน" in cb.columns:
                    cb["รหัสหัวข้อ"] = cb["รหัสรายงาน"]
                else:
                    cb["รหัสหัวข้อ"] = pd.NA

            cb["รหัสหัวข้อ"] = _coerce_code_str(cb["รหัสหัวข้อ"])

            # ทำความสะอาดชื่อคอลัมน์เป้าหมายที่เราจะดึง เช่น กลุ่มอุบัติการณ์, หมวด
            # ถ้าชื่อในไฟล์คุณต่างไป ให้เพิ่ม mapping ได้
            for col in ["กลุ่มอุบัติการณ์", "หมวด"]:
                if col not in cb.columns:
                    cb[col] = pd.NA

            # ถ้า df ไม่มีคีย์พอ (ว่างทั้งหมด) ให้ข้าม merge
            if "รหัสหัวข้อ" in df.columns and df["รหัสหัวข้อ"].notna().any():
                df = df.merge(cb[["รหัสหัวข้อ", "กลุ่มอุบัติการณ์", "หมวด"]],
                              on="รหัสหัวข้อ", how="left", suffixes=("", "_cb"))
                # เติมค่าว่างจากด้านซ้ายด้วยค่าที่ merge มา
                for col in ["กลุ่มอุบัติการณ์", "หมวด"]:
                    if col in df.columns and f"{col}_cb" in df.columns:
                        df[col] = df[col].fillna(df[f"{col}_cb"])
                        df.drop(columns=[f"{col}_cb"], inplace=True, errors="ignore")
            else:
                # ไม่มีคีย์ให้ merge
                missing_cols.extend(["กลุ่มอุบัติการณ์", "หมวด"])
        except Exception as e:
            # โหลด codebook ไม่ได้ ไม่ต้องล้ม ให้ทำงานต่อและแจ้งเตือน
            print(f"[compat_columns] โหลด {allcode_path} ไม่ได้: {e}")
            missing_cols.extend(["กลุ่มอุบัติการณ์", "หมวด"])
    else:
        # ไม่ได้ระบุ codebook
        missing_cols.extend(["กลุ่มอุบัติการณ์", "หมวด"])

    # คืนค่าตามสัญญาเดิม
    return df, list(dict.fromkeys(missing_cols))  # unique & keep order
                
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
