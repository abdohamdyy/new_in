FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY config.py .
COPY drug_search.py .
COPY app.py .

RUN pip install --no-cache-dir \
    flask gunicorn \
    chromadb \
    langchain-huggingface \
    transformers sentencepiece tokenizers \
    sentence-transformers \
    torch

EXPOSE 5543

CMD ["gunicorn", "--bind", "0.0.0.0:5543", "--workers", "2", "--threads", "4", "--timeout", "180", "app:app"]
