import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
from pathlib import Path

# ✅ 1. ย้ายตัวแปรที่ต้องใช้ร่วมกันมาไว้ข้างบนสุด
ENTITY_TO_ANONYMIZED_TOKEN_MAP = {
    "HN": "[HN_NUMBER]",
    "PERSON": "[PERSON]",
    "LOCATION": "[LOCATION]",
    "ORGANIZATION": "[ORGANIZATION]",
    # เพิ่มกฎอื่นๆ ได้ที่นี่ เช่น
    # "IDCARD": "[IDCARD_NUMBER]"
}

# Regex สำหรับ HN
HN_PATTERN = re.compile(r'(?<![0-9A-Za-zก-๙])HN[\s\.\-:]*\d{1,}', re.IGNORECASE)
# Regex สำหรับตรวจหาโทเคนที่แทนแล้ว เช่น [HN_NUMBER]
PLACEHOLDER_PATTERN = re.compile(r'\[[A-Z_]+\]')


@st.cache_resource
def load_ner_model():
    """
    Loads the NER model and tokenizer from a local path and creates the pipeline.
    """
    print("Attempting to load NER model from local path...")
    try:
        model_name = "pythainlp/thainer-corpus-v2-base-model"
        model_folder_name = model_name.split('/')[1]
        model_path = Path(__file__).parent / "models" / model_folder_name

        if not model_path.is_dir():
            st.error(f"Error: Model directory not found at the expected path: {model_path}")
            parent_dir_contents = list(Path(__file__).parent.glob('*'))
            st.warning(f"Contents of the root directory: {parent_dir_contents}")
            return None

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)

        ner_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            aggregation_strategy="simple"
        )

        print("NER pipeline created successfully.")
        return ner_pipeline

    except Exception as e:
        print(f"An unexpected error occurred while loading NER model: {e}")
        st.error(f"เกิดข้อผิดพลาดร้ายแรงในการโหลด NER model: {e}")
        return None


def anonymize_text(text, ner_model):
    """
    Anonymizes text by first applying rules (Regex for HN) and then using the NER model.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # --- ขั้นตอนที่ 1: ใช้ Regex ค้นหาและแทนที่ HN ก่อน ---
    anonymized_text = HN_PATTERN.sub(ENTITY_TO_ANONYMIZED_TOKEN_MAP["HN"], text)

    if not ner_model:
        return anonymized_text

    try:
        # --- กันไม่ให้ NER ไปแทนซ้ำใน [TOKEN] ที่เราวางไว้แล้ว ---
        protected_spans = [(m.start(), m.end()) for m in PLACEHOLDER_PATTERN.finditer(anonymized_text)]

        def overlaps(a, b):
            return not (a[1] <= b[0] or b[1] <= a[0])

        # --- ขั้นตอนที่ 2: ทำ NER ---
        ner_results = ner_model(anonymized_text)

        for entity in sorted(ner_results, key=lambda x: x['start'], reverse=True):
            entity_group = entity['entity_group']
            start, end = entity['start'], entity['end']

            # ข้ามถ้าซ้อนกับโทเคนที่แทนแล้ว
            if any(overlaps((start, end), ps) for ps in protected_spans):
                continue

            if entity_group in ENTITY_TO_ANONYMIZED_TOKEN_MAP:
                token = ENTITY_TO_ANONYMIZED_TOKEN_MAP[entity_group]
                anonymized_text = anonymized_text[:start] + token + anonymized_text[end:]
                # เพิ่มช่วงที่แทนแล้วเข้า protected_spans
                protected_spans.append((start, start + len(token)))

        return anonymized_text

    except Exception as e:
        print(f"Error during NER anonymization for text: '{text[:100]}...' | Error: {e}")
        return anonymized_text
