# Start with the official Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# ✅ --- START: เพิ่มส่วนติดตั้ง Git LFS ---
# Install system dependencies needed for Git LFS
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs
# ✅ --- END: สิ้นสุดส่วนที่เพิ่ม ---

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# ✅ --- เพิ่มคำสั่ง git lfs pull เพื่อดึงไฟล์ใหญ่ ---
RUN git lfs pull
# --- สิ้นสุด ---

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]