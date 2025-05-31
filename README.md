# Proyecto MLOps: Predicción de Precios Inmobiliarios

Este proyecto implementa un sistema completo de MLOps para la predicción de precios de propiedades inmobiliarias, utilizando datos de Realtor.com. El sistema incluye recolección automática de datos, procesamiento, entrenamiento de modelos, y despliegue continuo.

## Arquitectura del Sistema

El proyecto está compuesto por los siguientes componentes principales:

- **Airflow**: Orquestación de pipelines de datos y entrenamiento
- **MLflow**: Registro y seguimiento de experimentos
- **FastAPI**: API para servir predicciones
- **Streamlit**: Interfaz de usuario para predicciones y visualización
- **Prometheus & Grafana**: Monitoreo y observabilidad
- **PostgreSQL**: Almacenamiento de datos y metadatos
- **MinIO**: Almacenamiento de artefactos

## Estructura del Proyecto

```
.
├── airflow/
│   ├── dags/
│   │   └── real_estate_pipeline.py
│   └── Dockerfile
├── api/
│   ├── app/
│   │   └── main.py
│   └── Dockerfile
├── streamlit/
│   ├── app/
│   │   └── main.py
│   └── Dockerfile
├── mlflow/
│   └── Dockerfile
├── monitoring/
│   ├── grafana/
│   │   └── provisioning/
│   └── prometheus/
│       └── prometheus.yml
├── docker-compose.yml
└── requirements.txt
```

## Requisitos

- Docker y Docker Compose
- Python 3.9+
- Acceso a la API de datos inmobiliarios

## Configuración

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd real-estate-mlops
```

2. Crear y configurar el archivo .env:
```bash
cp .env.example .env
# Editar .env con las credenciales necesarias
```

3. Iniciar los servicios:
```bash
docker-compose up -d
```

## Acceso a los Servicios

- Airflow UI: http://localhost:8080
- MLflow UI: http://localhost:5000
- FastAPI Docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## Pipeline de Datos

1. **Recolección**: Datos obtenidos de la API de Realtor.com
2. **Procesamiento**: Limpieza y transformación de datos
3. **Entrenamiento**: Evaluación y entrenamiento automático de modelos
4. **Despliegue**: Actualización automática del modelo en producción

## Monitoreo y Observabilidad

- Métricas de rendimiento del modelo
- Latencia de predicciones
- Distribución de datos
- Alertas configurables

## Desarrollo

Para contribuir al proyecto:

1. Crear un nuevo branch:
```bash
git checkout -b feature/nueva-funcionalidad
```

2. Realizar cambios y commits:
```bash
git add .
git commit -m "Descripción del cambio"
```

3. Crear pull request

## CI/CD

El proyecto utiliza GitHub Actions para:

- Pruebas automáticas
- Construcción de imágenes Docker
- Publicación en DockerHub
- Despliegue con Argo CD

## Licencia

[MIT License](LICENSE) # Proyecto_Final_MLOPS
