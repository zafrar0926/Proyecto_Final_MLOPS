# MLOps Real Estate Price Prediction Project

Este proyecto implementa un sistema MLOps completo para la predicción de precios de bienes raíces, utilizando datos de Realtor.com. El sistema incluye recolección automática de datos, procesamiento, entrenamiento continuo, y despliegue automatizado de modelos.

## Arquitectura del Sistema

El proyecto está organizado en varios microservicios desplegados en Kubernetes:

### Componentes Principales
- **FastAPI**: API REST para servir predicciones en tiempo real
- **MLflow**: Gestión y tracking de experimentos ML
- **Airflow**: Orquestación de pipelines de datos
- **Streamlit**: UI para predicciones e interpretabilidad
- **PostgreSQL**: Almacenamiento de datos (Raw y Clean)
- **MinIO**: Almacenamiento de artefactos ML
- **Grafana & Prometheus**: Monitoreo y métricas

### Bases de Datos
- **Raw Data DB**: Almacena datos crudos de la API
- **Clean Data DB**: Almacena datos procesados
- **MLflow DB**: Metadatos de experimentos
- **Airflow DB**: Metadatos de DAGs

## Flujo de Trabajo

1. **Recolección de Datos**:
   - Airflow orquesta la recolección periódica desde la API externa
   - Los datos se almacenan en Raw Data DB

2. **Procesamiento**:
   - Pipeline de limpieza y transformación
   - Datos procesados almacenados en Clean Data DB

3. **Entrenamiento**:
   - Evaluación automática de necesidad de reentrenamiento
   - Registro de experimentos en MLflow
   - Selección automática del mejor modelo

4. **Despliegue**:
   - Modelos registrados en MLflow
   - FastAPI sirve el modelo en producción
   - Streamlit proporciona interfaz de usuario

5. **Monitoreo**:
   - Métricas de API recolectadas por Prometheus
   - Visualización en dashboards de Grafana

## Requisitos

- Kubernetes (Minikube o similar)
- kubectl
- Python 3.8+
- Docker

## Despliegue

1. Clonar el repositorio:
```bash
git clone https://github.com/zafrar0926/Proyecto_Final_MLOPS.git
cd Proyecto_Final_MLOPS
```

2. Crear namespace:
```bash
kubectl create namespace mlops
```

3. Desplegar componentes:
```bash
kubectl apply -f kubernetes/base -n mlops
kubectl apply -f kubernetes/airflow -n mlops
kubectl apply -f kubernetes/api -n mlops
```

## Acceso a los Servicios

Los servicios están disponibles en los siguientes puertos:

- Streamlit: Puerto 8501
- FastAPI: Puerto 8000
- MLflow: Puerto 5000
- Airflow: Puerto 8080
- Grafana: Puerto 3000

## Estructura del Proyecto

```
.
├── airflow/               # DAGs y configuración
├── api/                   # Código FastAPI
├── kubernetes/            # Manifiestos K8s
│   ├── airflow/          # Configuración Airflow
│   ├── api/              # Configuración FastAPI
│   ├── base/             # Configuraciones base
│   └── monitoring/       # Grafana y Prometheus
├── models/               # Código de modelos ML
└── streamlit/            # Código Streamlit UI
```

## Monitoreo y Métricas

El sistema incluye métricas clave:
- Latencia de predicciones
- Tasa de solicitudes
- Precisión del modelo
- Uso de recursos

## Interpretabilidad

La interfaz de Streamlit incluye:
- Visualizaciones SHAP para interpretabilidad
- Historial de modelos
- Comparación de versiones
- Métricas de rendimiento

## Contribuir

Para contribuir al proyecto:
1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit cambios (`git commit -m 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Crear Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
