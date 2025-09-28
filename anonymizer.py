# anonymizer.py
import re
from pathlib import Path

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from huggingface_hub import snapshot_download

# ====== ค่าตัวแทนเมื่อปกปิดข้อมูล ======
ENTITY_TO_ANONYMIZED_TOKEN_MAP = {
    "HN": "[HN_NUMBER]",
    "PERSON": "[PERSON]",
    "LOCATION": "[LOCATION]",
    "ORGANIZATION": "[ORGANIZATION]",
}

# ====== Regex rules ======
HN_PATTERN = re.compile(r'HN[\s\.\-:]*\d+', re.IGNORECASE)
PLACEHOLDER_PATTERN = re.compile(r'\[[A-Z_]+\]')  # กันไม่ให้แทนซ้ำใน [TOKEN]


@st.cache_resource
def load_ner_model():
    """
    ดาวน์โหลด (ครั้งแรก) และโหลดโมเดล NER จาก Hugging Face
    แสดงสถานะด้วยกล่องเดียว (st.status)
    """
    with st.status("🚀 กำลังโหลดโมเดล NER...", expanded=True) as status:
        try:
            local_dir = Path("model")
            if not local_dir.exists() or not any(local_dir.iterdir()):
                st.write("🔽 กำลังดาวน์โหลดโมเดลจาก Hugging Face...")
                snapshot_download(
                    repo_id="pythainlp/thainer-corpus-v2-base-model",
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                )

            st.write("⚙️ กำลังโหลดโมเดลเข้าหน่วยความจำ...")
            tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
            model = AutoModelForTokenClassification.from_pretrained(str(local_dir))

            ner_pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1,  # CPU
                aggregation_strategy="simple",
            )
            status.update(label="✅ โหลด NER pipeline เรียบร้อยแล้ว", state="complete")
            return ner_pipeline

        except Exception as e:
            status.update(label=f"❌ โหลดโมเดลล้มเหลว: {e}", state="error")
            return None


def anonymize_text(text: str, ner_model):
    """
    ปกปิดข้อมูลในหนึ่งข้อความ: ทำ Regex HN ก่อน แล้วค่อยผ่าน NER
    """
    if not isinstance(text, str) or not text.strip():
        return text

    anonymized = HN_PATTERN.sub(ENTITY_TO_ANONYMIZED_TOKEN_MAP["HN"], text)

    if not ner_model:
        return anonymized

    try:
        protected_spans = [(m.start(), m.end()) for m in PLACEHOLDER_PATTERN.finditer(anonymized)]

        def overlaps(a, b):
            return not (a[1] <= b[0] or b[1] <= a[0])

        ner_results = ner_model(anonymized)

        for ent in sorted(ner_results, key=lambda x: x["start"], reverse=True):
            start, end = ent["start"], ent["end"]
            if any(overlaps((start, end), ps) for ps in protected_spans):
                continue

            group = ent.get("entity_group")
            if group in ENTITY_TO_ANONYMIZED_TOKEN_MAP:
                token = ENTITY_TO_ANONYMIZED_TOKEN_MAP[group]
                anonymized = anonymized[:start] + token + anonymized[end:]
                protected_spans.append((start, start + len(token)))

        return anonymized

    except Exception:
        # ถ้า NER มีปัญหา ให้คืนข้อความที่ทำ Regex แล้ว
        return anonymized


def anonymize_column(df, text_col: str, ner_model, out_col: str = "รายละเอียดการเกิด_Anonymized"):
    """
    ปกปิดทั้งคอลัมน์ พร้อมกล่องสถานะเดียว + progress bar
    """
    if text_col not in df.columns:
        df[out_col] = df.get(text_col, "")
        return df

    with st.status("🔒 กำลังปกปิดข้อมูลส่วนบุคคล…", expanded=True) as status:
        n = len(df)
        pbar = st.progress(0)

        texts = df[text_col].astype(str).tolist()
        out = []
        for i, txt in enumerate(texts, start=1):
            out.append(anonymize_text(txt, ner_model))
            # อัปเดตเป็น % โดยไม่สร้างบรรทัดใหม่
            pbar.progress(int(i * 100 / max(n, 1)))

        df[out_col] = out
        status.update(label="✅ ปกปิดข้อมูลส่วนบุคคลเรียบร้อย", state="complete")
        return df
