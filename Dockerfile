FROM python:3.11-slim

WORKDIR /app

# Install cmake just in case another library needs it
RUN apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true", "--server.enableCORS", "false"]
