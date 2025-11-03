# compat_columns.py
import re
import pandas as pd


def _coerce_code_str(series: pd.Series) -> pd.Series:
    def _norm(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip()
        s = re.sub(r"\.0$", "", s)  # ตัด .0 ที่หลงมาจากเลข
        s = s.replace(",", "")
        return s
    return series.map(_norm).astype("string")


def normalize_dataframe_columns(raw_df: pd.DataFrame, allcode_path: str | None = None):
    # ถ้าได้ None เข้ามา ให้คืน DataFrame ว่าง + missing=[] เพื่อไม่ให้โค้ดส่วนอื่น error
    if raw_df is None:
        return pd.DataFrame(), []

    df = raw_df.copy()

    # ----- map คีย์มาตรฐานจากคอลัมน์ของคุณ -----
    # รหัสหัวข้อ: ใช้ 'รหัสรายงาน' ถ้ามี ไม่งั้นลอง 'In.HCode' ถัดไป
    if "รหัสรายงาน" in df.columns:
        df["รหัสหัวข้อ"] = _coerce_code_str(df["รหัสรายงาน"])
    elif "In.HCode" in df.columns:
        df["รหัสหัวข้อ"] = _coerce_code_str(df["In.HCode"])
    else:
        df["รหัสหัวข้อ"] = pd.NA

    # หัวข้อ
    if "Incident" in df.columns:
        df["หัวข้อ"] = df["Incident"].astype("string")
    else:
        df["หัวข้อ"] = pd.NA

    # วัน-เวลา ที่เกิดเหตุ = วดป.ที่เกิด + ช่อง 'ช่วงเวลา/เวร' (ถ้ามี)
    base = df["วดป.ที่เกิด"].astype("string") if "วดป.ที่เกิด" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    if "ช่วงเวลา/เวร" in df.columns:
        df["วัน-เวลา ที่เกิดเหตุ"] = (base.fillna("") + " " + df["ช่วงเวลา/เวร"].astype("string").fillna("")).str.strip()
        df.loc[df["วัน-เวลา ที่เกิดเหตุ"] == "", "วัน-เวลา ที่เกิดเหตุ"] = pd.NA
    else:
        df["วัน-เวลา ที่เกิดเหตุ"] = base

    # ระดับความรุนแรง
    if "ความรุนแรง" in df.columns:
        df["ระดับความรุนแรง"] = df["ความรุนแรง"].astype("string")
    else:
        df["ระดับความรุนแรง"] = pd.NA

    # สรุปปัญหา/เหตุการณ์โดยย่อ
    if "รายละเอียดการเกิด" in df.columns:
        df["สรุปปัญหา/เหตุการณ์โดยย่อ"] = df["รายละเอียดการเกิด"].astype("string")
    else:
        df["สรุปปัญหา/เหตุการณ์โดยย่อ"] = pd.NA

    # สร้างช่องว่างไว้ก่อน (ถ้ายังไม่มี)
    if "กลุ่มอุบัติการณ์" not in df.columns:
        df["กลุ่มอุบัติการณ์"] = pd.NA
    if "หมวด" not in df.columns:
        df["หมวด"] = pd.NA

    # ----- เติมกลุ่ม/หมวดจาก Code2024.xlsx หากระบุไฟล์ -----
    if allcode_path:
        try:
            cb = pd.read_excel(allcode_path, engine="openpyxl")

            # คีย์ codebook: รองรับได้ทั้ง 'รหัสหัวข้อ' หรือ 'รหัสรายงาน' หรือ 'In.HCode'
            if "รหัสหัวข้อ" in cb.columns:
                cb_key = "รหัสหัวข้อ"
            elif "รหัสรายงาน" in cb.columns:
                cb_key = "รหัสรายงาน"
            elif "In.HCode" in cb.columns:
                cb_key = "In.HCode"
            else:
                cb_key = None

            if cb_key:
                cb = cb.copy()
                cb["รหัสหัวข้อ"] = _coerce_code_str(cb[cb_key])

                if "กลุ่มอุบัติการณ์" not in cb.columns:
                    cb["กลุ่มอุบัติการณ์"] = pd.NA
                if "หมวด" not in cb.columns:
                    cb["หมวด"] = pd.NA

                # บังคับฝั่งซ้ายให้เป็น string แล้ว merge
                df["รหัสหัวข้อ"] = _coerce_code_str(df["รหัสหัวข้อ"])
                df = df.merge(
                    cb[["รหัสหัวข้อ", "กลุ่มอุบัติการณ์", "หมวด"]],
                    on="รหัสหัวข้อ",
                    how="left",
                    suffixes=("", "_cb"),
                )
                # ถ้าเดิมไม่มีค่า ให้ใช้ค่าจาก codebook
                for col in ["กลุ่มอุบัติการณ์", "หมวด"]:
                    if f"{col}_cb" in df.columns:
                        df[col] = df[col].fillna(df[f"{col}_cb"])
                        df.drop(columns=[f"{col}_cb"], inplace=True, errors="ignore")
        except Exception:
            # ไม่เตือน/ไม่หยุด — ยึดตามที่ขอ
            pass

    # ไม่สนใจว่าจะ “ขาด” อะไร — คืนค่า df และ missing=[] เสมอ
    return df, []
