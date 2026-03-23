FROM python:3.9-slim

# Install system dependencies required by cv2, insightface, and rembg
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the Inswapper model directly into the server image
RUN wget -nc -O inswapper_128.onnx https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx

# Download the Super Resolution (Upscaling) model
RUN wget -nc -O ESPCN_x2.pb https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb

# Copy the rest of the application
COPY . .

# Hugging Face routes external traffic to port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
