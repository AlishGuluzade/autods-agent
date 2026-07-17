FROM python:3.11-slim

WORKDIR /app

# System deps needed by lightgbm/xgboost wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV UPLOAD_DIR=/tmp/autods_uploads
ENV ARTIFACTS_DIR=/tmp/autods_artifacts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
