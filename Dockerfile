# 1. เปลี่ยนจาก slim เป็นเวอร์ชันเต็ม
FROM python:3.11

# 2. ตั้งค่า Working Directory ภายใน Container
WORKDIR /app

# 3. ติดตั้งโปรแกรมเสริมที่จำเป็น
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libpango-1.0-0 \
    libharfbuzz0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. คัดลอกไฟล์ requirements.txt เข้าไปก่อน
COPY requirements.txt ./

# 5. ติดตั้ง Python libraries ทั้งหมด
RUN pip install --no-cache-dir -r requirements.txt

# 6. คัดลอกไฟล์โค้ดทั้งหมดที่เหลือเข้าไป
COPY . .

# 7. เปิด Port 8080
EXPOSE 8080

# 8. คำสั่งสำหรับรันแอป Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true", "--server.enableCORS", "false"]
