# ===== Base =====
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# اختياري: أدوات بناء خفيفة (لو مكتبتك تحتاج)
# تقدر تشيل build-essential لو مش محتاج
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# نزّل المتطلبات لو موجودة (مرن: لو مفيش requirements مش هيكسر)
COPY requirements*.txt /tmp/
RUN python -m pip install --upgrade pip && \
    if [ -f /tmp/requirements.txt ]; then pip install -r /tmp/requirements.txt; fi && \
    if [ -f /tmp/requirements-dev.txt ]; then pip install -r /tmp/requirements-dev.txt; fi

EXPOSE 5543

# ===== Development =====
FROM base AS dev
# في التطوير هنربط الكود بـ bind mount من الـ compose،
# لكن بنعمل COPY احتياطي لو حد شغّل الصورة من غير compose.
COPY . /app

# ===== Production =====
FROM base AS prod
COPY . /app
# الافتراضي للإنتاج (compose في التطوير بيستبدل الـ command)
CMD ["gunicorn", "-w", "4", "-k", "gthread", "--threads", "8", "-b", "0.0.0.0:5543", "app:app", "--access-logfile", "-", "--error-logfile", "-", "--timeout", "120"]
