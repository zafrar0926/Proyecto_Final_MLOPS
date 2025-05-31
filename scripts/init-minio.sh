#!/bin/bash

# Esperar a que MinIO esté disponible
echo "Esperando a que MinIO esté disponible..."
until curl -sf http://minio:9000/minio/health/live; do
    echo "MinIO no está listo - esperando..."
    sleep 5
done

# Crear bucket de MLflow
echo "Creando bucket de MLflow..."
mc config host add minio http://minio:9000 minioadmin minioadmin
mc mb minio/mlflow

echo "Configuración de MinIO completada." 