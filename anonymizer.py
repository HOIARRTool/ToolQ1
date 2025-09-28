# File: anonymizer.py
import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
from pathlib import Path

# ใช้ @st.cache_resource เพื่อให้แน่ใจว่าโมเดลจะถูกโหลดแค่ครั้งเดียว
# ซึ่งจะช่วยให้แอปทำงานเร็วขึ้นมากหลังจากการรันครั้งแรก
@st.cache_resource
def load_ner_model():
    """
    Loads the NER model and tokenizer from a local path and creates the pipeline.
    This is the robust way to load local models with Transformers.
    """
    print("Attempting to load NER model from local path...")
    try:
        # 1. ระบุที่อยู่ของโมเดลเหมือนเดิม
        # Path(__file__).parent คือที่อยู่ของโฟลเดอร์ที่ไฟล์ anonymizer.py นี้อยู่
        model_path = Path(__file__).parent / "models" / "no-name-ner-th"

        # 2. ตรวจสอบว่าโฟลเดอร์โมเดลมีอยู่จริง (สำหรับ Debug)
        if not model_path.is_dir():
            st.error(f"Error: Model directory not found at the expected path: {model_path}")
            # แสดงไฟล์และโฟลเดอร์ทั้งหมดในระดับบนสุดเพื่อช่วยหา
            parent_dir_contents = list(Path(__file__).parent.glob('*'))
            st.warning(f"Contents of the root directory: {parent_dir_contents}")
            return None

        # 3. โหลด Tokenizer และ Model แยกกันโดยตรงจาก Path ที่ระบุ
        # คำสั่ง .from_pretrained จะเข้าใจได้ทันทีว่านี่คือการโหลดจากไฟล์ในเครื่อง
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)

        # 4. สร้าง pipeline จาก object ที่โหลดมาแล้ว
        ner_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            aggregation_strategy="simple"  
        )
        
        print("NER pipeline created successfully with aggregation strategy.")
        return ner_pipeline

    except Exception as e:
        print(f"An unexpected error occurred while loading NER model: {e}")
        st.error(f"เกิดข้อผิดพลาดร้ายแรงในการโหลด NER model: {e}")
        return None


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
    hn_pattern = r'[Hh][Nn]([\s.]?\d+|\s[a-zA-Z]{2}\d+)'
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
