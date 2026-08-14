FROM python:3.10-slim


WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files into /app
COPY app.py .
COPY inference_runner.py .
COPY util.py .
COPY model.py .

# Bake in default weights
COPY model.pt /app/model.pt

ENV HOME=/app
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# Setup permissions
RUN mkdir -p /tmp/Ultralytics && \
    chown -R 8080:8080 /app /tmp/Ultralytics && \
    chmod -R 777 /tmp

USER 8080
EXPOSE 8080

CMD ["python", "app.py"]