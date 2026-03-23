FROM python:3.9-slim

WORKDIR /app

# Combine system dependencies and Python dependencies in a single layer to save Docker cache space
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev wget build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Download AI Models
RUN wget -nc -O inswapper_128.onnx https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx
RUN wget -nc -O ESPCN_x2.pb https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb

COPY . .
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
