import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
from pathlib import Path

# ✅ 1. ย้ายตัวแปรที่ต้องใช้ร่วมกันมาไว้ข้างบนสุด
# สร้าง Dictionary สำหรับเก็บ "กฎ" การแปลงค่าต่างๆ
ENTITY_TO_ANONYMIZED_TOKEN_MAP = {
    "HN": "[HN_NUMBER]",
    "PERSON": "[PERSON]",
    "LOCATION": "[LOCATION]",
    "ORGANIZATION": "[ORGANIZATION]",
    # คุณสามารถเพิ่มกฎอื่นๆ ได้ที่นี่ในอนาคต เช่น
    # "IDCARD": "[IDCARD_NUMBER]"
}


@st.cache_resource
def load_ner_model():
    """
    Loads the NER model and tokenizer from a local path and creates the pipeline.
    """
    print("Attempting to load NER model from local path...")
    try:
        # ใช้โมเดลตัวล่าสุดที่เราตกลงกัน
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
    hn_pattern = r'\bHN[\s.]?\d+\b'
    anonymized_text = re.sub(hn_pattern, ENTITY_TO_ANONYMIZED_TOKEN_MAP["HN"], text, flags=re.IGNORECASE)

    if not ner_model:
        return anonymized_text

    try:
        # --- ขั้นตอนที่ 2: ทำ NER กับข้อความที่ผ่าน Regex มาแล้ว ---
        ner_results = ner_model(anonymized_text)

        # คัดกรองและแทนที่ Entity ที่โมเดลหาเจอ
        for entity in sorted(ner_results, key=lambda x: x['start'], reverse=True):
            entity_group = entity['entity_group']
            # ตรวจสอบว่า entity ที่เจอ มีอยู่ในกฎของเราหรือไม่
            if entity_group in ENTITY_TO_ANONYMIZED_TOKEN_MAP:
                start, end = entity['start'], entity['end']
                token = ENTITY_TO_ANONYMIZED_TOKEN_MAP[entity_group]
                anonymized_text = anonymized_text[:start] + token + anonymized_text[end:]

        return anonymized_text

    except Exception as e:
        print(f"Error during NER anonymization for text: '{text[:100]}...' | Error: {e}")
        # ถ้าเกิด Error ระหว่างทำ NER ให้ส่งค่าที่ผ่าน Regex แล้วกลับไปก่อน
        return anonymized_text