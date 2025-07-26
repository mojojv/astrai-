# Dockerfile para ASTRAI Cancer Detection
FROM python:3.11-slim

# Metadatos
LABEL maintainer="ASTRAI Team"
LABEL description="ASTRAI Cancer Detection System - LLM + CNN for Thyroid Analysis"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgdcm-tools \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN useradd --create-home --shell /bin/bash astrai
USER astrai
WORKDIR /home/astrai

# Copiar requirements y instalar dependencias Python
COPY --chown=astrai:astrai requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY --chown=astrai:astrai . .

# Crear directorios necesarios
RUN mkdir -p logs models datasets reports configs

# Exponer puerto para API
EXPOSE 8000

# Comando por defecto
CMD ["python", "-m", "src.api.main"]

