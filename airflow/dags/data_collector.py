from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np

class Day(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

class DataCollector:
    def __init__(self):
        self.data = None

    def generate_synthetic_data(self, n_samples=100):
        """Genera datos sintéticos para propiedades inmobiliarias."""
        np.random.seed(42)
        
        data = {
            'bed': np.random.randint(1, 7, n_samples),
            'bath': np.random.randint(1, 5, n_samples),
            'acre_lot': np.random.uniform(0.1, 2.0, n_samples),
            'house_size': np.random.uniform(800, 5000, n_samples),
            'price': np.random.uniform(100000, 1000000, n_samples),
            'total_rooms': np.random.randint(3, 12, n_samples),
            'created_at': [datetime.now() - timedelta(days=x) for x in range(n_samples)]
        }
        
        self.data = pd.DataFrame(data)
        return self.data

    def clean_data(self):
        """Limpia los datos generados."""
        if self.data is None:
            raise ValueError("No hay datos para limpiar. Genera datos primero.")
        
        # Eliminar duplicados
        self.data = self.data.drop_duplicates()
        
        # Eliminar valores nulos
        self.data = self.data.dropna()
        
        # Asegurar tipos de datos correctos
        self.data['bed'] = self.data['bed'].astype(int)
        self.data['bath'] = self.data['bath'].astype(int)
        self.data['total_rooms'] = self.data['total_rooms'].astype(int)
        
        return self.data

    def save_to_db(self, db_connection, table_name):
        """Guarda los datos en la base de datos."""
        if self.data is None:
            raise ValueError("No hay datos para guardar. Genera datos primero.")
        
        self.data.to_sql(table_name, db_connection, if_exists='append', index=False) 