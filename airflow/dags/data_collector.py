import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine, text
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        # Conexión a la base de datos raw
        self.raw_db_url = "postgresql://rawdata:rawdata123@raw-data-db:5432/raw_data"
        self.data = None
        
    def fetch_from_db(self, limit=None):
        """Obtiene datos desde la base de datos raw."""
        try:
            # Crear conexión a la base de datos
            engine = create_engine(self.raw_db_url)
            
            # Construir query
            query = "SELECT * FROM raw_properties"
            if limit:
                query += f" LIMIT {limit}"
            
            # Ejecutar query y cargar en DataFrame
            with engine.connect() as conn:
                self.data = pd.read_sql(query, conn)
                
            logger.info(f"Datos obtenidos exitosamente de la base de datos. Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error al obtener datos de la base de datos: {str(e)}")
            raise
    
    def clean_data(self):
        """Limpia y preprocesa los datos."""
        if self.data is None:
            raise ValueError("No hay datos para limpiar. Obtén datos primero.")
            
        try:
            # Eliminar duplicados
            self.data = self.data.drop_duplicates()
            
            # Eliminar valores nulos
            self.data = self.data.dropna()
            
            # Convertir tipos de datos
            numeric_columns = ['bed', 'bath', 'price', 'house_size', 'acre_lot', 'brokered_by', 'street', 'zip_code']
            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Convertir fecha si existe
            if 'prev_sold_date' in self.data.columns:
                self.data['prev_sold_date'] = pd.to_datetime(self.data['prev_sold_date'], errors='coerce')
            
            logger.info(f"Datos limpiados exitosamente. Shape final: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error en la limpieza de datos: {str(e)}")
            raise

    def process_data(self):
        """Procesa los datos para análisis."""
        if self.data is None:
            raise ValueError("No hay datos para procesar. Obtén datos primero.")
        
        try:
            # Aquí agregaremos más procesamiento según sea necesario
            # Por ejemplo, feature engineering, normalización, etc.
            
            # Por ahora, solo verificamos que tenemos los datos
            logger.info(f"Datos listos para análisis. Shape: {self.data.shape}")
            logger.info(f"Columnas disponibles: {self.data.columns.tolist()}")
            
            # Calcular algunas estadísticas básicas
            stats = {
                'precio_promedio': self.data['price'].mean(),
                'precio_mediano': self.data['price'].median(),
                'tamaño_promedio': self.data['house_size'].mean(),
                'habitaciones_promedio': self.data['bed'].mean(),
                'baños_promedio': self.data['bath'].mean()
            }
            
            logger.info(f"Estadísticas calculadas: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error en el procesamiento de datos: {str(e)}")
            raise

# Instanciar el procesador de datos
processor = DataProcessor()

# Funciones para las tareas del DAG
def process_raw_data():
    """Función que ejecuta el proceso completo de procesamiento de datos."""
    try:
        processor.fetch_from_db()
        processor.clean_data()
        stats = processor.process_data()
        logger.info("Procesamiento de datos completado exitosamente")
        return stats
    except Exception as e:
        logger.error(f"Error en el proceso de datos: {str(e)}")
        raise

# Configuración del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 31),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Crear el DAG
dag = DAG(
    'real_estate_pipeline',
    default_args=default_args,
    description='Pipeline para procesamiento de datos inmobiliarios',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Tarea de procesamiento de datos
process_data = PythonOperator(
    task_id='process_data',
    python_callable=process_raw_data,
    dag=dag,
)

# Definir el flujo de tareas
process_data 