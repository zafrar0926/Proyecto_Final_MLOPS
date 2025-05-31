import requests
import pandas as pd
from typing import Dict, Any, List, Union
import logging
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from enum import Enum

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class Day(str, Enum):
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"

class DataCollector:
    def __init__(self, group_number: int = 1, day: Day = Day.WEDNESDAY):
        """
        Inicializa el colector de datos.
        Args:
            group_number: Número del grupo (por defecto 1)
            day: Día de la clase (Tuesday o Wednesday)
        """
        self.api_url = os.getenv('API_SOURCE_URL', 'http://10.43.101.108:80')
        self.group_number = group_number
        self.day = day
        self.raw_data_path = os.path.join('data', 'raw')
        
        # Crear directorio si no existe
        os.makedirs(self.raw_data_path, exist_ok=True)
        
        logger.info(f"Inicializando DataCollector para el Grupo {self.group_number} - {self.day}")
    
    def test_connection(self) -> bool:
        """Prueba la conexión con la API usando el endpoint health."""
        try:
            response = requests.get(f"{self.api_url}/health")
            response.raise_for_status()
            data = response.json()
            status = data.get('status', 'Unknown')
            logger.info(f"Estado de la API: {status}")
            return status == "OK"
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al verificar la salud de la API: {e}")
            return False
    
    def restart_data_generation(self) -> bool:
        """Reinicia la generación de datos para el grupo."""
        try:
            params = {
                'group_number': self.group_number,
                'day': self.day
            }
            response = requests.get(f"{self.api_url}/restart_data_generation", params=params)
            response.raise_for_status()
            logger.info(f"Generación de datos reiniciada para Grupo {self.group_number}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al reiniciar la generación de datos: {e}")
            return False
    
    def collect_data(self) -> Dict[str, Any]:
        """
        Recolecta datos de la API para el grupo específico.
        Los datos incluyen información sobre propiedades inmobiliarias.
        """
        try:
            params = {
                'group_number': self.group_number,
                'day': self.day
            }
            response = requests.get(f"{self.api_url}/data", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or not isinstance(data, dict):
                logger.warning("No se recibieron datos válidos de la API")
                return {}
            
            batch_number = data.get('batch_number', 0)
            properties_data = data.get('data', [])
            
            logger.info(f"Datos recolectados exitosamente - Batch #{batch_number}")
            logger.info(f"Número de propiedades en este batch: {len(properties_data)}")
            
            if properties_data:
                logger.info("Ejemplo de propiedad:")
                logger.info(json.dumps(properties_data[0], indent=2))
            
            return data
            
        except requests.exceptions.RequestException as e:
            if "recolectaron todos los datos" in str(e):
                logger.info("Se han recolectado todos los datos disponibles para este grupo")
            else:
                logger.error(f"Error al recolectar datos: {e}")
            return {}
    
    def save_raw_data(self, data: Dict[str, Any], filename: str = None) -> None:
        """
        Guarda los datos crudos en formato JSON y crea un DataFrame con las propiedades.
        Args:
            data: Datos a guardar
            filename: Nombre del archivo (opcional)
        """
        try:
            if not data:
                logger.warning("No hay datos para guardar")
                return
            
            batch_number = data.get('batch_number', 0)
            properties_data = data.get('data', [])
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"raw_data_group_{self.group_number}_{self.day}_batch_{batch_number}_{timestamp}.json"
            
            file_path = os.path.join(self.raw_data_path, filename)
            
            # Guardar datos completos como JSON
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Datos completos guardados en {file_path}")
            
            # Crear y analizar DataFrame de propiedades
            if properties_data:
                df = pd.DataFrame(properties_data)
                
                logger.info("\nResumen del DataFrame de propiedades:")
                logger.info(f"- Dimensiones: {df.shape}")
                logger.info(f"- Columnas: {df.columns.tolist()}")
                
                # Verificar tipos de datos
                logger.info("\nTipos de datos:")
                for col in df.columns:
                    logger.info(f"- {col}: {df[col].dtype}")
                
                # Estadísticas básicas para columnas numéricas
                numeric_cols = ['price', 'bed', 'bath', 'acre_lot', 'house_size']
                logger.info("\nEstadísticas básicas:")
                for col in numeric_cols:
                    if col in df.columns:
                        stats = df[col].describe()
                        logger.info(f"\n{col}:")
                        logger.info(f"- Media: {stats['mean']:.2f}")
                        logger.info(f"- Min: {stats['min']:.2f}")
                        logger.info(f"- Max: {stats['max']:.2f}")
            
        except Exception as e:
            logger.error(f"Error al guardar los datos: {e}")

if __name__ == "__main__":
    # Prueba básica del colector - Grupo 1, Miércoles
    collector = DataCollector(group_number=1, day=Day.WEDNESDAY)
    
    # Verificar estado de la API
    if collector.test_connection():
        # Reiniciar generación de datos
        if collector.restart_data_generation():
            # Recolectar datos
            data = collector.collect_data()
            if data:
                collector.save_raw_data(data) 