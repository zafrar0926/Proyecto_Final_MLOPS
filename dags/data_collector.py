import requests
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Day(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

class DataCollector:
    def __init__(self, api_url="http://data-api:80"):
        self.api_url = api_url
        self.data = None

    def fetch_data(self, group_id):
        """Obtiene datos de la API externa."""
        try:
            # Hacer la petición a la API
            response = requests.get(f"{self.api_url}/data/{group_id}")
            
            if response.status_code == 200:
                # Convertir la respuesta a DataFrame
                self.data = pd.DataFrame(response.json())
                logger.info(f"Datos obtenidos exitosamente. Shape: {self.data.shape}")
                return self.data
            else:
                logger.error(f"Error al obtener datos: {response.status_code} - {response.text}")
                raise Exception(f"Error al obtener datos: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error en la petición a la API: {str(e)}")
            raise

    def clean_data(self):
        """Limpia los datos obtenidos."""
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
            
            # Agregar timestamp de recolección
            self.data['collected_at'] = datetime.now()
            
            logger.info(f"Datos limpiados exitosamente. Shape final: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error en la limpieza de datos: {str(e)}")
            raise

    def save_to_db(self, db_connection, table_name):
        """Guarda los datos en la base de datos."""
        if self.data is None:
            raise ValueError("No hay datos para guardar. Obtén datos primero.")
        
        try:
            self.data.to_sql(table_name, db_connection, if_exists='append', index=False)
            logger.info(f"Datos guardados exitosamente en la tabla {table_name}")
        except Exception as e:
            logger.error(f"Error al guardar en la base de datos: {str(e)}")
            raise 