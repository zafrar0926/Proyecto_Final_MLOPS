#!/bin/bash

# Inicializar la base de datos de Airflow
airflow db init

# Crear usuario admin
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Crear las conexiones necesarias
airflow connections add 'postgres_raw' \
    --conn-type 'postgres' \
    --conn-login 'rawdata' \
    --conn-password 'rawdata123' \
    --conn-host 'raw_data_db' \
    --conn-port '5432' \
    --conn-schema 'raw_data'

airflow connections add 'postgres_clean' \
    --conn-type 'postgres' \
    --conn-login 'cleandata' \
    --conn-password 'cleandata123' \
    --conn-host 'clean_data_db' \
    --conn-port '5432' \
    --conn-schema 'clean_data'

airflow connections add 'mlflow_connection' \
    --conn-type 'http' \
    --conn-host 'mlflow' \
    --conn-port '5000'

# Crear variables necesarias
airflow variables set 'group_number' '1'
airflow variables set 'class_day' 'Wednesday'

echo "Inicialización de Airflow completada." 