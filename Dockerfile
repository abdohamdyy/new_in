FROM python:3.11-slim

# بيئة تشغيل
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# أدوات نظام أساسية
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# تثبيت باكدجات بايثون
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# نسخ الكود
COPY . /app

EXPOSE 5543

# أمر تشغيل افتراضي (تقدر تغيّره من docker-compose برضه)
CMD gunicorn -w 4 -k gthread --threads 8 \
    -b 0.0.0.0:5543 \
    app:app \
    --access-logfile - \
    --error-logfile - \
    --timeout 120
