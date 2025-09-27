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

        # 4. สร้าง pipeline จาก object ของ model และ tokenizer ที่โหลดมาแล้ว
        ner_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
        )
        
        print("NER model, tokenizer, and pipeline loaded successfully from local path.")
        return ner_pipeline

    except Exception as e:
        print(f"An unexpected error occurred while loading NER model: {e}")
        st.error(f"เกิดข้อผิดพลาดร้ายแรงในการโหลด NER model: {e}")
        return None

def anonymize_text(text, ner_model):
    """Anonymizes text by replacing identified entities with placeholders."""
    if not ner_model:
        return "Error: NER model is not available."
    try:
        ner_results = ner_model(text)
        anonymized_text = text
        # Process results from last to first to keep indices valid
        for entity in sorted(ner_results, key=lambda x: x['start'], reverse=True):
            start, end = entity['start'], entity['end']
            entity_label = f"[{entity['entity_group']}]"
            anonymized_text = anonymized_text[:start] + entity_label + anonymized_text[end:]
        return anonymized_text
    except Exception as e:
        print(f"Error during anonymization: {e}")
        return "Error during text processing."
