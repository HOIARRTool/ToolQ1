from transformers import pipeline
import os

# สร้างโฟลเดอร์ models ถ้ายังไม่มี
if not os.path.exists("models"):
    os.makedirs("models")

print("Downloading NER model...")

# ดาวน์โหลดและบันทึกโมเดลลงในโฟลเดอร์ชื่อ models/no-name-ner-th
model = pipeline("token-classification", model="loolootech/no-name-ner-th")
model.save_pretrained("./models/no-name-ner-th")

print("Model downloaded and saved successfully to ./models/no-name-ner-th")