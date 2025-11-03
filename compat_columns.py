# compat_columns.py
import re
import pandas as pd


def _coerce_code_str(series: pd.Series) -> pd.Series:
    """บังคับคีย์ให้เป็นสตริงสะอาด (ตัด .0 ท้ายค่าตัวเลข, ตัดช่องว่าง)"""
    def _norm(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip()
        s = re.sub(r"\.0$", "", s)  # ตัด .0 จากเลขที่มาจาก Excel
        s = s.replace(",", "")
        return s
    return series.map(_norm).astype("string")


def normalize_dataframe_columns(raw_df: pd.DataFrame, allcode_path: str | None = None):
    """
    - ทำคอลัมน์ให้เข้ามาตรฐานขั้นต่ำสำหรับแอป
    - ถ้ามี codebook (Code2024.xlsx) จะ merge 'กลุ่มอุบัติการณ์' และ 'หมวด'
    คืนค่า: (df, missing_cols)
    """
    df = raw_df.copy()

    # 1) จัดการคอลัมน์คีย์
    if "รหัสหัวข้อ" not in df.columns and "รหัสรายงาน" in df.columns:
        df["รหัสหัวข้อ"] = df["รหัสรายงาน"]

    if "รหัสหัวข้อ" in df.columns:
        df["รหัสหัวข้อ"] = _coerce_code_str(df["รหัสหัวข้อ"])

    # 2) รายการคอลัมน์ที่ถ้ายังไม่มี/ได้ไม่ครบ จะรายงานกลับ (แต่ไม่หยุดทำงาน)
    required_min = [
        "รหัสหัวข้อ", "หัวข้อ", "วัน-เวลา ที่เกิดเหตุ",
        "ระดับความรุนแรง", "สรุปปัญหา/เหตุการณ์โดยย่อ",
        "กลุ่มอุบัติการณ์", "หมวด"
    ]
    missing_cols: list[str] = [c for c in required_min if c not in df.columns]

    # 3) โหลด codebook และ merge กลุ่ม/หมวด (ถ้าให้พาธมา)
    if allcode_path:
        try:
            cb = pd.read_excel(allcode_path, engine="openpyxl")

            # ให้มีคีย์ 'รหัสหัวข้อ' ใน codebook เช่นกัน
            if "รหัสหัวข้อ" not in cb.columns:
                if "รหัสรายงาน" in cb.columns:
                    cb["รหัสหัวข้อ"] = cb["รหัสรายงาน"]
                else:
                    cb["รหัสหัวข้อ"] = pd.NA

            cb["รหัสหัวข้อ"] = _coerce_code_str(cb["รหัสหัวข้อ"])

            # ให้มีคอลัมน์ผลลัพธ์ที่ต้องการ
            for col in ["กลุ่มอุบัติการณ์", "หมวด"]:
                if col not in cb.columns:
                    cb[col] = pd.NA

            # มีคีย์พอให้ merge ไหม?
            if "รหัสหัวข้อ" in df.columns and df["รหัสหัวข้อ"].notna().any():
                df = df.merge(
                    cb[["รหัสหัวข้อ", "กลุ่มอุบัติการณ์", "หมวด"]],
                    on="รหัสหัวข้อ",
                    how="left",
                    suffixes=("", "_cb"),
                )

                # ถ้าด้านซ้ายว่าง ให้เติมค่าที่ merge มา
                for col in ["กลุ่มอุบัติการณ์", "หมวด"]:
                    if col in df.columns and f"{col}_cb" in df.columns:
                        df[col] = df[col].fillna(df[f"{col}_cb"])
                        df.drop(columns=[f"{col}_cb"], inplace=True, errors="ignore")
            else:
                # ไม่มีคีย์จะ merge
                for col in ["กลุ่มอุบัติการณ์", "หมวด"]:
                    if col not in df.columns:
                        df[col] = pd.NA
                missing_cols.extend(["กลุ่มอุบัติการณ์", "หมวด"])

        except Exception as e:
            print(f"[compat_columns] โหลด {allcode_path} ไม่ได้: {e}")
            # ถ้าโหลดไม่ได้ ให้สร้างคอลัมน์เปล่าไว้ก่อน
            for col in ["กลุ่มอุบัติการณ์", "หมวด"]:
                if col not in df.columns:
                    df[col] = pd.NA
            missing_cols.extend(["กลุ่มอุบัติการณ์", "หมวด"])
    else:
        # ไม่ได้ระบุ codebook: สร้างคอลัมน์ว่างไว้ก่อน
        for col in ["กลุ่มอุบัติการณ์", "หมวด"]:
            if col not in df.columns:
                df[col] = pd.NA
        missing_cols.extend(["กลุ่มอุบัติการณ์", "หมวด"])

    # unique และรักษาลำดับ
    missing_cols = list(dict.fromkeys(missing_cols))
    return df, missing_cols
