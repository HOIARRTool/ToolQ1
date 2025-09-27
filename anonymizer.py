# File: anonymizer.py
import os
import streamlit as st
from transformers import pipeline
import re
from pathlib import Path

@st.cache_resource
def load_ner_model():
    """Loads the NER pipeline model and caches it."""
    print("Attempting to load NER model from local path...")
    
    try:
        # สร้างที่อยู่ของโมเดลแบบเต็ม (Absolute Path)
        # Path(__file__).parent จะได้ที่อยู่ของโฟลเดอร์ที่ไฟล์ anonymizer.py นี้อยู่
        # ซึ่งก็ควรจะเป็นโฟลเดอร์หลักของโปรเจกต์
        model_path = Path(__file__).parent / "models" / "no-name-ner-th"

        model = pipeline(
            "token-classification",
            model=model_path,  # ใช้ที่อยู่แบบเต็มที่สร้างขึ้น
            device=-1,
            token=os.environ.get("HUGGING_FACE_TOKEN")
        )
        print("NER model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading NER model: {e}")
        st.error(f"เกิดข้อผิดพลาดในการโหลด NER model: {e}")
        return None

# กำหนดประเภทของข้อมูลที่จะถูกแทนที่ และ Token ที่จะใช้แทน
ENTITY_TO_ANONYMIZED_TOKEN_MAP = {
    "PERSON": "[PERSON]",
    "PHONE": "[PHONE]",
    "EMAIL": "[EMAIL]",
    "ADDRESS": "[LOCATION]",
    "DATE": "[DATE]",
    "NATIONAL_ID": "[NATIONAL_ID]",
    "HOSPITAL_IDS": "[HOSPITAL_IDS]",
    "HN": "[HOSPITAL_NUMBER]",
    "ORGANIZATION": "[ORGANIZATION]",
    "LOCATION": "[LOCATION]",
    "URL": "[URL]",
}

def anonymize_text(original_text: str, ner_model):
    """
    ใช้โมเดล NER และ RegEx เพื่อปกปิดข้อมูลส่วนบุคคลในข้อความ
    """
    if not isinstance(original_text, str) or not original_text.strip():
        return original_text

    # ==========================================================
    # ✨ ขั้นตอนที่ 1 (เพิ่มใหม่): ใช้ RegEx ค้นหาและแทนที่ HN ก่อน ✨
    # ==========================================================
    # รูปแบบ: ค้นหา HN หรือ hn, ตามด้วยเว้นวรรคหรือไม่ก็ได้ ( \s? ), ตามด้วยตัวเลข 1 ตัวขึ้นไป ( \d+ )
    hn_pattern = r'[Hh][Nn]\s?\d+'
    text_after_regex = re.sub(hn_pattern, ENTITY_TO_ANONYMIZED_TOKEN_MAP["HN"], original_text)

    # ถ้าไม่มี ner_model ก็ให้คืนค่าหลังจากทำ RegEx ไปเลย
    if not ner_model:
        return text_after_regex

    try:
        # ==========================================================
        # ✨ ขั้นตอนที่ 2 (ของเดิม): ทำ NER กับข้อความที่ผ่าน RegEx มาแล้ว ✨
        # ==========================================================
        ner_results = ner_model(text_after_regex)

        if not ner_results:
            return text_after_regex

        # 2. จัดการ Entity ที่ต่อเนื่องกัน
        combined_entities = []
        for entity in ner_results:
            entity_name = re.sub(r'^[BI]-', '', entity['entity'])
            entity['entity'] = entity_name

            if (combined_entities and
                    combined_entities[-1]['entity'] == entity_name and
                    entity['start'] == combined_entities[-1]['end']):
                combined_entities[-1]['word'] += entity['word']
                combined_entities[-1]['end'] = entity['end']
            elif (combined_entities and
                  combined_entities[-1]['entity'] == entity_name and
                  entity['start'] == combined_entities[-1]['end'] + 1 and
                  text_after_regex[entity['start'] - 1].isspace()):
                combined_entities[-1]['word'] += " " + entity['word']
                combined_entities[-1]['end'] = entity['end']
            else:
                combined_entities.append(entity)

        # 3. คัดกรองเฉพาะ entity ที่เราต้องการปกปิด
        entities_to_anonymize = [
            e for e in combined_entities if e['entity'] in ENTITY_TO_ANONYMIZED_TOKEN_MAP
        ]

        # 4. เรียงลำดับจากท้ายมาหน้า
        entities_to_anonymize.sort(key=lambda x: x['start'], reverse=True)

        # 5. แทนที่ข้อความ
        anonymized_text = text_after_regex
        for entity in entities_to_anonymize:
            start, end = entity['start'], entity['end']
            token = ENTITY_TO_ANONYMIZED_TOKEN_MAP.get(entity['entity'])
            if token:
                anonymized_text = anonymized_text[:start] + token + anonymized_text[end:]

        return anonymized_text

    except Exception:
        # หากเกิดข้อผิดพลาดใดๆ ให้คืนค่าข้อความที่ผ่าน RegEx แล้วไปก่อน
        return text_after_regex
