from transformers import pipeline
import os

# 1. กำหนดชื่อโมเดลที่ถูกต้องและมีขนาดเล็กกว่า
MODEL_NAME = "pythainlp/thainer-corpus-v2-base-model"

# 2. สร้างที่อยู่สำหรับบันทึกไฟล์โดยอัตโนมัติจากชื่อโมเดล
# จะได้ผลลัพธ์เป็น ./models/thainer-corpus-v2-base-model
SAVE_PATH = f"./models/{MODEL_NAME.split('/')[1]}"

# 3. สร้างโฟลเดอร์ถ้ายังไม่มี
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

print(f"Downloading model '{MODEL_NAME}'...")

# 4. ดาวน์โหลดโมเดล
model = pipeline("token-classification", model=MODEL_NAME)

# 5. บันทึกโมเดลลงในที่อยู่ที่เราสร้างขึ้น
model.save_pretrained(SAVE_PATH)

print(f"Model downloaded and saved successfully to {SAVE_PATH}")