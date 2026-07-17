# Combined single-container image -- this is the one to use for Hugging
# Face Spaces (which runs exactly one container per Space) or any other
# platform that only exposes one port. For local development, prefer
# `docker-compose up` instead, which runs backend/frontend separately with
# hot reload.
#
# Build: docker build -t autods-agent .
# Run:   docker run -p 7860:7860 -e GROQ_API_KEY=... autods-agent

# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend, serving the built frontend too ----
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY --from=frontend-build /frontend/dist ./frontend_dist

ENV FRONTEND_DIST=/app/frontend_dist
ENV UPLOAD_DIR=/tmp/autods_uploads
ENV ARTIFACTS_DIR=/tmp/autods_artifacts
ENV PORT=7860

# Cloud Run injects $PORT at runtime and requires the container to listen on
# it; Hugging Face Spaces expects 7860 by default.
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
