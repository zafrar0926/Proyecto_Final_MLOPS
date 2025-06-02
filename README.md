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

Una vez ejecutado `./deploy.sh`, los servicios deberían estar accesibles a través de `kubectl port-forward` que el script intenta iniciar automáticamente. Las URL son:

- **Airflow**: `http://localhost:8080/`
- **MLflow**: `http://localhost:5000/`
- **Streamlit**: `http://localhost:8501/`
- **FastAPI Docs**: `http://localhost:8001/docs` (el puerto base para la API es `8001`)
- **Grafana**: `http://localhost:3000/`

El script `deploy.sh` intenta iniciar estos port-forwards en segundo plano. Los logs de cada port-forward se guardan en `/tmp/<nombre_servicio>_pf.log`.

Si necesitas iniciar un port-forward manualmente (por ejemplo, si el script falla o si lo detienes), puedes usar los siguientes comandos en terminales separadas:

```bash
# Para Airflow
kubectl port-forward svc/airflow-webserver -n mlops 8080:8080

# Para MLflow
kubectl port-forward svc/mlflow -n mlops 5000:5000

# Para Streamlit
kubectl port-forward svc/streamlit -n mlops 8501:8501

# Para FastAPI (acceso en puerto local 8001)
kubectl port-forward svc/fastapi -n mlops 8001:80 

# Para Grafana
kubectl port-forward svc/grafana -n mlops 3000:3000
```

Si deseas detener todos los port-forwards iniciados por el script o manualmente que sigan el patrón de `kubectl port-forward`:
```bash
pkill -f "kubectl port-forward"
```

Anteriormente, el proyecto intentaba usar Ingress para exponer los servicios, pero debido a problemas persistentes con `minikube tunnel`, se ha simplificado el acceso mediante port-forwarding directo por ahora.

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
