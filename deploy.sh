#!/bin/bash

# Función para esperar a que un pod esté listo
wait_for_pod() {
    local label=$1
    local namespace=$2
    echo "Esperando a que el pod $label esté listo..."
    kubectl wait --for=condition=ready pod -l $label -n $namespace --timeout=300s
}

# Iniciar Minikube si no está corriendo
if ! minikube status >/dev/null 2>&1; then
    echo "Iniciando Minikube..."
    minikube start
fi

# Crear namespace si no existe
kubectl create namespace mlops 2>/dev/null || true

# Aplicar ConfigMaps y Secrets
echo "Aplicando ConfigMaps y Secrets..."
kubectl apply -f kubernetes/config/ -n mlops

# Crear PV y PVC para datos raw
echo "Creando volúmenes persistentes..."
kubectl apply -f kubernetes/storage/raw-data-pv.yaml -n mlops

# Aplicar bases de datos
echo "Aplicando bases de datos..."
kubectl apply -f kubernetes/databases/raw-data-db.yaml -n mlops
kubectl apply -f kubernetes/databases/clean-data-db.yaml -n mlops

# Esperar a que la base de datos raw esté lista
wait_for_pod "app=raw-data-db" "mlops"

# Crear tabla raw_properties
echo "Creando tabla raw_properties..."
kubectl apply -f kubernetes/jobs/create-table.yaml -n mlops

# Cargar datos iniciales
echo "Cargando datos iniciales..."
kubectl apply -f kubernetes/jobs/load-data.yaml -n mlops

# Aplicar data-api
echo "Aplicando data-api..."
kubectl apply -f kubernetes/data-api/data-api.yaml -n mlops

# Aplicar componentes de monitoreo
echo "Aplicando componentes de monitoreo..."
kubectl apply -f kubernetes/monitoring/ -n mlops

# Aplicar componentes de MLflow
echo "Aplicando componentes de MLflow..."
kubectl apply -f kubernetes/mlflow/ -n mlops

# Aplicar componentes de Airflow
echo "Aplicando componentes de Airflow..."
kubectl apply -f kubernetes/airflow/ -n mlops

# Aplicar componentes de FastAPI
echo "Aplicando componentes de FastAPI..."
kubectl apply -f kubernetes/api/ -n mlops

# Aplicar componentes de Streamlit
echo "Aplicando componentes de Streamlit..."
kubectl apply -f kubernetes/streamlit/ -n mlops

# Aplicar Ingress
echo "Aplicando Ingress..."
kubectl apply -f kubernetes/ingress/ -n mlops

# Esperar a que los servicios principales estén listos
echo "Esperando a que los servicios estén listos..."
wait_for_pod "app=raw-data-db" "mlops"
wait_for_pod "app=clean-data-db" "mlops"
wait_for_pod "app=data-api" "mlops"
wait_for_pod "app=mlflow" "mlops"
wait_for_pod "app=airflow" "mlops"
wait_for_pod "app=fastapi" "mlops"
wait_for_pod "app=streamlit" "mlops"
wait_for_pod "app=grafana" "mlops"

echo "Deployment completado!"

echo "Iniciando port-forwards en segundo plano..."
echo "Esto puede tardar unos segundos en estabilizarse."

# Detener port-forwards anteriores
echo "Deteniendo port-forwards existentes..."
pkill -f "kubectl port-forward" || true
sleep 5

echo "Iniciando nuevos port-forwards..."
kubectl port-forward svc/airflow-webserver -n mlops 8080:8080 > /tmp/airflow_pf.log 2>&1 &
kubectl port-forward svc/mlflow -n mlops 5000:5000 > /tmp/mlflow_pf.log 2>&1 &
kubectl port-forward svc/streamlit -n mlops 8501:8501 > /tmp/streamlit_pf.log 2>&1 &
kubectl port-forward svc/fastapi -n mlops 8001:80 > /tmp/fastapi_pf.log 2>&1 &
kubectl port-forward svc/data-api -n mlops 8002:80 > /tmp/data_api_pf.log 2>&1 &
kubectl port-forward svc/grafana -n mlops 3000:3000 > /tmp/grafana_pf.log 2>&1 &
kubectl port-forward svc/raw-data-db -n mlops 5432:5432 > /tmp/raw_db_pf.log 2>&1 &

echo "Los logs de port-forward están en /tmp/*_pf.log"
echo "Espera unos segundos y luego accede a los servicios:"
echo "- Airflow: http://localhost:8080/"
echo "- MLflow: http://localhost:5000/"
echo "- Streamlit: http://localhost:8501/"
echo "- FastAPI Docs: http://localhost:8001/docs"
echo "- Data API: http://localhost:8002/docs"
echo "- Grafana: http://localhost:3000/"
echo "- Raw Database: localhost:5432"
echo "Si un servicio no carga, revisa su log de port-forward en /tmp/."
echo "Puedes detener todos los port-forwards con: pkill -f 'kubectl port-forward'" 