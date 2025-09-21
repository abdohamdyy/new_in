# استخدم صورة Python خفيفة
FROM python:3.11-slim

# بيئة تشغيل
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# حزم نظام مطلوبة لبعض البايناريات (hnswlib/torch) والـ curl للـ healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# مجلد العمل
WORKDIR /app

# نسخ requirements وتثبيت
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# نسخ كود التطبيق
COPY . /app

# المنفذ
EXPOSE 5543

# أمر التشغيل (Gunicorn)
# 4 عُمّال + 8 ثريدز — عدّل زي ما تحب حسب السيرفر
CMD gunicorn -w 4 -k gthread --threads 8 \
    -b 0.0.0.0:${PORT:-5543} \
    app:app \
    --access-logfile - \
    --error-logfile - \
    --timeout 120
