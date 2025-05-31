#!/bin/bash

# Start MLflow server with environment variables
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${MLFLOW_TRACKING_URI}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" 