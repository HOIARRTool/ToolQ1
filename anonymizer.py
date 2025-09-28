import os
import re
from pathlib import Path

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from huggingface_hub import snapshot_download

# ✅ ตัวแปรแทน entity
ENTITY_TO_ANONYMIZED_TOKEN_MAP = {
    "HN": "[HN_NUMBER]",
    "PERSON": "[PERSON]",
    "LOCATION": "[LOCATION]",
    "ORGANIZATION": "[ORGANIZATION]",
}

# ✅ Regex
HN_PATTERN = re.compile(r'(?<![0-9A-Za-zก-๙])HN[\s\.\-:]*\d{1,}', re.IGNORECASE)
PLACEHOLDER_PATTERN = re.compile(r'\[[A-Z_]+\]')


@st.cache_resource
def load_ner_model():
    """
    โหลด NER model จาก Hugging Face (ดาวน์โหลดครั้งแรกแล้ว cache ไว้)
    """
    local_dir = Path("model")
    if not local_dir.exists() or not any(local_dir.iterdir()):
        st.info("🔽 กำลังดาวน์โหลดโมเดล thainer-corpus-v2-base-model จาก Hugging Face...")
        progress_bar = st.progress(0)

        # snapshot_download ไม่มี callback โดยตรง เราเลยใช้ trick
        # แสดง progress ขั้นตอนแบบ "ประมาณการ"
        progress_bar.progress(20)
        snapshot_download(
            repo_id="pythainlp/thainer-corpus-v2-base-model",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        progress_bar.progress(60)

    try:
        st.info("⚙️ กำลังโหลดโมเดลเข้าหน่วยความจำ...")
        progress_bar = st.progress(80)

        tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
        model = AutoModelForTokenClassification.from_pretrained(str(local_dir))

        ner_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            aggregation_strategy="simple"
        )

        progress_bar.progress(100)
        st.success("✅ โหลด NER pipeline เรียบร้อยแล้ว")
        return ner_pipeline

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดร้ายแรงในการโหลด NER model: {e}")
        return None


def anonymize_text(text, ner_model):
    """
    Anonymizes text by first applying rules (Regex for HN) and then using the NER model.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    anonymized_text = HN_PATTERN.sub(ENTITY_TO_ANONYMIZED_TOKEN_MAP["HN"], text)

    if not ner_model:
        return anonymized_text

    try:
        protected_spans = [(m.start(), m.end()) for m in PLACEHOLDER_PATTERN.finditer(anonymized_text)]

        def overlaps(a, b):
            return not (a[1] <= b[0] or b[1] <= a[0])

        ner_results = ner_model(anonymized_text)

        for entity in sorted(ner_results, key=lambda x: x['start'], reverse=True):
            entity_group = entity['entity_group']
            start, end = entity['start'], entity['end']

            if any(overlaps((start, end), ps) for ps in protected_spans):
                continue

            if entity_group in ENTITY_TO_ANONYMIZED_TOKEN_MAP:
                token = ENTITY_TO_ANONYMIZED_TOKEN_MAP[entity_group]
                anonymized_text = anonymized_text[:start] + token + anonymized_text[end:]
                protected_spans.append((start, start + len(token)))

        return anonymized_text

    except Exception as e:
        print(f"Error during NER anonymization for text: '{text[:100]}...' | Error: {e}")
        return anonymized_text
