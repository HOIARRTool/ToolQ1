# 1. Start from a full, stable Debian base image
FROM debian:bullseye

# 2. Set environment variables to prevent interactive prompts
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 3. Install Python 3, pip, and all necessary system dependencies for your app
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    cmake \
    build-essential \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory
WORKDIR /app

# 5. Copy and install Python requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application's code
COPY . .

# 7. Expose the port
EXPOSE 8080

# 8. Define the command to run the app using python3
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true", "--server.enableCORS", "false"]
