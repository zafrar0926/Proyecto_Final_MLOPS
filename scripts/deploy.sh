#!/bin/bash

# Function to check if a pod is ready
wait_for_pod() {
    local pod_prefix=$1
    local namespace=${2:-default}
    local timeout=${3:-300}  # 5 minutes timeout by default
    
    echo "Waiting for pod $pod_prefix to be ready..."
    local start_time=$(date +%s)
    
    while true; do
        if [ $(($(date +%s) - start_time)) -gt $timeout ]; then
            echo "Timeout waiting for $pod_prefix"
            return 1
        fi
        
        if kubectl get pods -n $namespace | grep $pod_prefix | grep -q "1/1.*Running"; then
            echo "$pod_prefix is ready!"
            return 0
        fi
        sleep 5
    done
}

# Function to wait for job completion
wait_for_job() {
    local job_name=$1
    local namespace=${2:-default}
    local timeout=${3:-300}  # 5 minutes timeout by default
    
    echo "Waiting for job $job_name to complete..."
    local start_time=$(date +%s)
    
    while true; do
        if [ $(($(date +%s) - start_time)) -gt $timeout ]; then
            echo "Timeout waiting for $job_name"
            return 1
        fi
        
        if kubectl get job $job_name -n $namespace -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' | grep -q "True"; then
            echo "Job $job_name completed successfully!"
            return 0
        fi
        sleep 5
    done
}

# Clean up everything
echo "🧹 Cleaning up environment..."
kubectl delete --all jobs --namespace=default
kubectl delete --all pods --namespace=default
kubectl delete --all deployments --namespace=default
kubectl delete --all statefulsets --namespace=default
kubectl delete --all services --namespace=default
kubectl delete --all configmaps --namespace=default
kubectl delete --all secrets --namespace=default
kubectl delete --all ingress --namespace=default
kubectl delete --all pvc --namespace=default
kubectl delete --all pv --namespace=default

# Clean Docker
echo "🐳 Cleaning Docker environment..."
docker system prune -af --volumes

# Start Minikube if not running
if ! minikube status | grep -q "Running"; then
    echo "🚀 Starting Minikube..."
    minikube start
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/{raw,clean}

# Deploy components in order
echo "🚀 Deploying components..."

# 1. Deploy Storage Components
echo "📦 Deploying Storage Components..."
kubectl apply -f kubernetes/databases/
wait_for_pod "postgresql"
wait_for_pod "raw-data-db"
wait_for_pod "clean-data-db"

# 2. Deploy MinIO
echo "📦 Deploying MinIO..."
kubectl apply -f kubernetes/base/minio/
wait_for_pod "minio"

# 3. Deploy MLflow
echo "🔬 Deploying MLflow..."
kubectl apply -f kubernetes/mlflow/
wait_for_pod "mlflow"

# 4. Deploy and Initialize Airflow
echo "🌪️ Deploying Airflow..."
kubectl apply -f kubernetes/airflow/init-job.yaml
wait_for_job "airflow-init"

kubectl apply -f kubernetes/airflow/
wait_for_pod "airflow-scheduler"
wait_for_pod "airflow-webserver"

# 5. Deploy FastAPI
echo "🚀 Deploying FastAPI..."
kubectl apply -f kubernetes/api/
wait_for_pod "fastapi"

# 6. Deploy Monitoring
echo "📊 Deploying Monitoring..."
kubectl apply -f kubernetes/monitoring/
wait_for_pod "prometheus"
wait_for_pod "grafana"

# 7. Deploy Streamlit
echo "🎨 Deploying Streamlit..."
kubectl apply -f kubernetes/streamlit/
wait_for_pod "streamlit"

# 8. Deploy Ingress
echo "🌐 Deploying Ingress..."
kubectl apply -f kubernetes/base/ingress/

# Make script executable
chmod +x deploy.sh

echo "✅ Deployment completed!"
echo "🔍 Checking pod status..."
kubectl get pods

echo "
Services should be available at:
- MLflow: http://mlflow.local
- Airflow: http://airflow.local
- FastAPI: http://api.local
- Streamlit: http://streamlit.local
- Grafana: http://grafana.local
" 