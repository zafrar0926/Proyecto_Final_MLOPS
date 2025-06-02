from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import logging
import json
from sqlalchemy import create_engine

# Agregar la carpeta scripts al path
sys.path.append("/opt/airflow/scripts")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar nuestros scripts
from data_collector import DataCollector

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

class DataProcessor:
    def __init__(self):
        # Conexión a la base de datos raw
        self.raw_db_url = "postgresql://rawdata:rawdata123@raw-data-db:5432/raw_data"
        self.clean_db_url = "postgresql://cleandata:cleandata123@clean-data-db:5432/clean_data"
        self.data = None
        
    def fetch_from_db(self, limit=None):
        """Obtiene datos desde la base de datos raw."""
        try:
            engine = create_engine(self.raw_db_url)
            query = "SELECT * FROM raw_properties"
            if limit:
                query += f" LIMIT {limit}"
            
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
            numeric_columns = ['bed', 'bath', 'price', 'house_size', 'acre_lot']
            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Convertir fecha si existe
            if 'prev_sold_date' in self.data.columns:
                self.data['prev_sold_date'] = pd.to_datetime(self.data['prev_sold_date'], errors='coerce')
            
            # Calcular total_rooms si no existe
            if 'total_rooms' not in self.data.columns and 'bed' in self.data.columns:
                self.data['total_rooms'] = self.data['bed']
            
            logger.info(f"Datos limpiados exitosamente. Shape final: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error en la limpieza de datos: {str(e)}")
            raise

    def save_to_clean_db(self):
        """Guarda los datos limpios en la base de datos clean."""
        if self.data is None:
            raise ValueError("No hay datos para guardar.")
            
        try:
            engine = create_engine(self.clean_db_url)
            self.data.to_sql('clean_properties', engine, if_exists='replace', index=False)
            logger.info(f"Datos guardados exitosamente en la base clean. Registros: {len(self.data)}")
        except Exception as e:
            logger.error(f"Error al guardar en la base de datos clean: {str(e)}")
            raise

# Instanciar el procesador
processor = DataProcessor()

def process_raw_data(**context):
    """Procesa los datos raw y los guarda en la base clean."""
    try:
        # Obtener y procesar datos
        processor.fetch_from_db()
        processor.clean_data()
        
        # Calcular estadísticas
        stats = {
            'precio_promedio': processor.data['price'].mean(),
            'precio_mediano': processor.data['price'].median(),
            'tamaño_promedio': processor.data['house_size'].mean(),
            'habitaciones_promedio': processor.data['bed'].mean(),
            'baños_promedio': processor.data['bath'].mean()
        }
        
        # Guardar en la base clean
        processor.save_to_clean_db()
        
        # Guardar estadísticas en XCom
        context['task_instance'].xcom_push(key='data_stats', value=json.dumps(stats))
        logger.info("Procesamiento y guardado de datos completado exitosamente")
        return stats
        
    except Exception as e:
        logger.error(f"Error en el proceso de datos: {str(e)}")
        raise

def train_model(**context):
    """Entrena el modelo usando los datos procesados."""
    try:
        from train import main as train_model
        
        # Entrenar modelo usando la base de datos clean
        model, scaler, metrics = train_model("postgresql://cleandata:cleandata123@clean-data-db:5432/clean_data")
        
        # Guardar métricas en XCom
        context['task_instance'].xcom_push(key='model_metrics', value=metrics)
        logger.info(f"Entrenamiento completado. R2 Score: {metrics}")
        
        return metrics
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {str(e)}")
        raise

# Crear el DAG
with DAG(
    'real_estate_pipeline',
    default_args=default_args,
    description='Pipeline para procesamiento y entrenamiento de modelo inmobiliario',
    schedule_interval='@daily',
    catchup=False
) as dag:
    
    # Tarea 1: Obtener y guardar datos
    collect_data = PythonOperator(
        task_id='collect_data',
        python_callable=collect_and_save_data,
    )
    
    # Tarea 2: Procesar y guardar datos
    process_data = PythonOperator(
        task_id='process_data',
        python_callable=process_raw_data,
    )
    
    # Tarea 3: Entrenar modelo
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    # Definir el orden de las tareas
    collect_data >> process_data >> train_model_task 