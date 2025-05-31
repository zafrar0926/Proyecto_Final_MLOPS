from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import mlflow
from sqlalchemy import create_engine
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar nuestros scripts
from data_collector import DataCollector
from train import main as train_model

# Configuración por defecto para los DAGs
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

def generate_and_save_data(**context):
    """Genera datos sintéticos y los guarda en la base de datos RAW."""
    try:
        # Crear instancia del colector
        collector = DataCollector()
        
        # Generar datos
        df = collector.generate_synthetic_data(n_samples=1000)
        
        # Limpiar datos
        df = collector.clean_data()
        
        # Guardar en la base de datos RAW
        engine = create_engine('postgresql://rawdata:rawdata123@raw_data_db:5432/raw_data')
        collector.save_to_db(engine, 'raw_properties')
        
        # Guardar también en formato parquet
        os.makedirs("/opt/airflow/data/raw", exist_ok=True)
        output_path = "/opt/airflow/data/raw/synthetic_batch_1.parquet"
        df.to_parquet(output_path, index=False)
        
        context['task_instance'].xcom_push(key='raw_data_path', value=output_path)
        return output_path
        
    except Exception as e:
        logger.error(f"Error en la generación de datos: {e}")
        raise

def process_and_save_data(**context):
    """Procesa los datos y los guarda en la base de datos CLEAN."""
    try:
        # Obtener path de los datos crudos
        raw_data_path = context['task_instance'].xcom_pull(task_ids='generate_data')
        
        # Si estamos en modo test y no hay valor en XCom, usar el path por defecto
        if raw_data_path is None:
            raw_data_path = "/opt/airflow/data/raw/synthetic_batch_1.parquet"
        
        # Leer datos
        df = pd.read_parquet(raw_data_path)
        
        # Guardar en la base de datos CLEAN
        engine = create_engine('postgresql://cleandata:cleandata123@clean_data_db:5432/clean_data')
        df.to_sql('clean_properties', engine, if_exists='replace', index=False)
        
        # Guardar también en formato parquet
        os.makedirs("/opt/airflow/data/processed", exist_ok=True)
        processed_path = "/opt/airflow/data/processed/clean_batch_1.parquet"
        df.to_parquet(processed_path, index=False)
        
        return processed_path
        
    except Exception as e:
        logger.error(f"Error en el procesamiento de datos: {e}")
        raise

def train_new_model(**context):
    """Entrena un nuevo modelo usando los datos procesados."""
    try:
        # Entrenar modelo
        model, scaler, test_r2 = train_model("postgresql://cleandata:cleandata123@clean_data_db:5432/clean_data")
        
        # Guardar métricas en XCom
        context['task_instance'].xcom_push(key='test_r2', value=test_r2)
        
        return test_r2
        
    except Exception as e:
        logger.error(f"Error en el entrenamiento del modelo: {e}")
        raise

# Crear el DAG
with DAG(
    'real_estate_pipeline',
    default_args=default_args,
    description='Pipeline para entrenamiento de modelo inmobiliario',
    schedule_interval='@daily',
    catchup=False
) as dag:
    
    # Tarea 1: Generar y guardar datos
    generate_data = PythonOperator(
        task_id='generate_data',
        python_callable=generate_and_save_data,
    )
    
    # Tarea 2: Procesar y guardar datos
    process_data = PythonOperator(
        task_id='process_data',
        python_callable=process_and_save_data,
    )
    
    # Tarea 3: Entrenar modelo
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_new_model,
    )
    
    # Definir el orden de las tareas
    generate_data >> process_data >> train_model_task 