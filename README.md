# ASTRAI Cancer Detection System

![ASTRAI Logo](docs/images/astrai_logo.png)

[![CI/CD Pipeline](https://github.com/astrai-team/cancer-detection/workflows/CI/badge.svg)](https://github.com/astrai-team/cancer-detection/actions)
[![Code Coverage](https://codecov.io/gh/astrai-team/cancer-detection/branch/main/graph/badge.svg)](https://codecov.io/gh/astrai-team/cancer-detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Descripción General

ASTRAI (Artificial Intelligence System for Thyroid Risk Assessment and Interpretation) es un sistema avanzado de inteligencia artificial diseñado para la detección y análisis automatizado de cáncer tiroideo mediante la combinación de modelos de visión por computadora (CNNs) y modelos de lenguaje grandes (LLMs). El sistema proporciona análisis médico automatizado, clasificación TI-RADS, y generación de reportes médicos profesionales.

### Características Principales

- **Segmentación Automática**: Identificación precisa de glándula tiroides, nódulos y calcificaciones
- **Clasificación TI-RADS**: Evaluación automática según estándares médicos internacionales
- **Análisis LLM**: Interpretación médica en lenguaje natural y generación de reportes
- **Soporte Multi-formato**: Compatible con DICOM, PNG, JPG y otros formatos médicos
- **API REST**: Interfaz programática para integración con sistemas hospitalarios
- **GUI Intuitiva**: Interfaz gráfica para uso clínico directo
- **Despliegue Escalable**: Arquitectura containerizada para producción

## Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Model Layer    │    │ Service Layer   │
│                 │    │                 │    │                 │
│ • DICOM Handler │    │ • CNN Models    │    │ • REST API      │
│ • Preprocessor  │    │ • LLM Models    │    │ • GUI           │
│ • Dataset       │    │ • Fusion Logic  │    │ • Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Instalación Rápida

### Requisitos del Sistema

- Python 3.9 o superior
- CUDA 11.8+ (opcional, para aceleración GPU)
- Docker y Docker Compose (para despliegue)
- 8GB RAM mínimo (16GB recomendado)
- 50GB espacio en disco para modelos

### Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/astrai-team/cancer-detection.git
cd cancer-detection

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con sus configuraciones

# Ejecutar pruebas
pytest tests/

# Iniciar la aplicación
python -m src.main --demo all
```

### Instalación con Docker

```bash
# Clonar el repositorio
git clone https://github.com/astrai-team/cancer-detection.git
cd cancer-detection

# Construir y ejecutar con Docker Compose
docker-compose up -d

# Verificar el estado
docker-compose ps

# Ver logs
docker-compose logs -f astrai-api
```

## Uso Básico

### API REST

```python
import requests

# Subir imagen para análisis
with open('thyroid_image.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/analyze',
        files={'image': f},
        data={'patient_id': '12345'}
    )

result = response.json()
print(f"TI-RADS: {result['tirads']}")
print(f"Análisis: {result['medical_analysis']}")
```

### Interfaz Gráfica

```bash
# Ejecutar GUI
python -m src.gui.main_window
```

### Línea de Comandos

```bash
# Analizar una imagen
python -m src.main analyze --image path/to/image.dcm --output report.pdf

# Entrenar modelo personalizado
python -m src.training.train_cnn --config configs/training_configs/cnn_config.yaml

# Evaluar modelo
python -m src.evaluation.evaluate --model models/best_model.pth --dataset test_data/
```

## Estructura del Proyecto

```
ASTRAI-Cancer-Detection/
├── src/                          # Código fuente principal
│   ├── core/                     # Componentes centrales
│   ├── data/                     # Procesamiento de datos
│   ├── models/                   # Modelos de IA
│   ├── api/                      # API REST
│   └── gui/                      # Interfaz gráfica
├── tests/                        # Pruebas automatizadas
├── configs/                      # Archivos de configuración
├── deployment/                   # Configuraciones de despliegue
├── docs/                         # Documentación
├── scripts/                      # Scripts de utilidad
└── datasets/                     # Datos de entrenamiento
```

## Configuración

### Variables de Entorno

```bash
# Configuración de la aplicación
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Configuración de modelos
MODEL_CACHE_DIR=./models
USE_GPU=true
MAX_BATCH_SIZE=8

# Configuración de base de datos
DATABASE_URL=postgresql://user:pass@localhost/astrai
REDIS_URL=redis://localhost:6379
```

### Configuración de Modelos

Los modelos se configuran mediante archivos YAML en `configs/model_configs/`:

```yaml
# cnn_config.yaml
model:
  name: "ThyroidSegmentationModel"
  backbone: "resnet50"
  num_classes: 4
  pretrained: true

training:
  batch_size: 16
  learning_rate: 0.001
  epochs: 100
  optimizer: "adam"
```

## Desarrollo

### Configuración del Entorno de Desarrollo

```bash
# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# Configurar pre-commit hooks
pre-commit install

# Ejecutar linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/

# Ejecutar pruebas con cobertura
pytest tests/ --cov=src --cov-report=html
```

### Contribuir

1. Fork el repositorio
2. Crear una rama para su feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit sus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear un Pull Request

### Estándares de Código

- Seguir PEP 8 para estilo de Python
- Usar type hints en todas las funciones
- Documentar funciones con docstrings
- Mantener cobertura de pruebas > 80%
- Usar conventional commits para mensajes

## Despliegue en Producción

### Kubernetes

```bash
# Aplicar manifiestos de Kubernetes
kubectl apply -f deployment/kubernetes/

# Verificar el despliegue
kubectl get pods -n astrai
kubectl get services -n astrai

# Escalar el despliegue
kubectl scale deployment astrai-api --replicas=5 -n astrai
```

### Monitoreo

El sistema incluye monitoreo completo con:

- **Prometheus**: Métricas de sistema y aplicación
- **Grafana**: Dashboards y visualizaciones
- **Alertmanager**: Alertas automáticas
- **Jaeger**: Trazas distribuidas

Acceder a los dashboards:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## Documentación

- [Guía de Instalación](docs/installation.md)
- [Manual de Usuario](docs/user_guide.md)
- [Documentación de API](docs/api_reference.md)
- [Guía de Desarrollo](docs/development.md)
- [Arquitectura del Sistema](docs/architecture.md)

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

