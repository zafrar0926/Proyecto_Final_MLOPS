import pandas as pd
import json
from sqlalchemy import create_engine
import logging
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Leer el archivo JSON directamente
    logger.info("Leyendo archivo JSON...")
    file_path = 'data/raw/raw_data_group_1_Wednesday_batch_0_20250531_114346.json'
    logger.info(f"Intentando leer archivo: {file_path}")
    
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    
    # Convertir los datos a DataFrame
    logger.info("Convirtiendo datos a DataFrame...")
    df = pd.DataFrame(json_data['data'])
    
    logger.info(f"Datos cargados del JSON: {len(df)} registros")
    logger.info(f"Columnas: {df.columns.tolist()}")
    
    # Crear conexión a la base de datos
    logger.info("Conectando a la base de datos...")
    engine = create_engine('postgresql://rawdata:rawdata123@localhost:5432/raw_data')
    
    # Guardar en la base de datos
    logger.info("Guardando datos en la base de datos...")
    df.to_sql('raw_properties', engine, if_exists='append', index=False)
    
    logger.info("Datos guardados exitosamente en la base de datos")
    
    # Verificar la cantidad de registros
    with engine.connect() as conn:
        result = conn.execute("SELECT COUNT(*) FROM raw_properties")
        count = result.fetchone()[0]
        logger.info(f"Total de registros en la tabla: {count}")
    
except Exception as e:
    logger.error(f"Error: {str(e)}")
    raise 