FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# أدوات لازمة لمكتبات ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# انسخ السورس المهم كله
COPY config.py .
COPY drug_search.py .
COPY equivalents_search.py .
COPY app.py .

# باكدجات البايثون (CPU)
RUN pip install --no-cache-dir \
    flask gunicorn \
    chromadb \
    langchain-huggingface \
    transformers sentencepiece tokenizers \
    sentence-transformers \
    torch

EXPOSE 5543

# شغّل بـ gunicorn على 5543 مع لوجات واضحة
CMD ["gunicorn", "--bind", "0.0.0.0:5543", "--workers", "2", "--threads", "4", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
