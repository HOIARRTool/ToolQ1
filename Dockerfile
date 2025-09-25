# 1. เลือก Base Image เป็น Python 3.11
FROM python:3.11-slim

# 2. ตั้งค่า Working Directory ภายใน Container
WORKDIR /app

# 3. ติดตั้งโปรแกรมเสริมที่จำเป็น (cmake)
RUN apt-get update && apt-get install -y cmake build-essential && rm -rf /var/lib/apt/lists/*

# 4. คัดลอกไฟล์ requirements.txt เข้าไปก่อน
COPY requirements.txt ./

# 5. ติดตั้ง Python libraries ทั้งหมด
RUN pip install --no-cache-dir -r requirements.txt

# 6. คัดลอกไฟล์โค้ดทั้งหมดที่เหลือเข้าไป
COPY . .

# 7. เปิด Port 8080 เพื่อให้คนภายนอกเข้ามาใช้งานได้
EXPOSE 8080

# 8. คำสั่งสำหรับรันแอป Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true", "--server.enableCORS", "false"]
