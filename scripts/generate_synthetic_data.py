import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(n_samples=1000):
    """Genera datos sintéticos para propiedades inmobiliarias."""
    np.random.seed(42)
    
    # Generar características base
    data = {
        'bed': np.random.randint(1, 7, n_samples),
        'bath': np.random.randint(1, 5, n_samples),
        'acre_lot': np.random.uniform(0.1, 2.0, n_samples),
        'house_size': np.random.uniform(800, 5000, n_samples),
        'status': np.random.choice(['for_sale', 'ready_to_build', 'foreclosure'], n_samples),
        'street': [f'Street {i}' for i in range(n_samples)],
        'city': np.random.choice(['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento'], n_samples),
        'state': 'CA',
        'zip_code': np.random.randint(90001, 96162, n_samples),
        'brokered_by': np.random.randint(1, 100, n_samples)
    }
    
    # Generar precios basados en las características
    base_price = 200000
    price = (
        base_price +
        data['bed'] * 50000 +
        data['bath'] * 30000 +
        data['acre_lot'] * 100000 +
        data['house_size'] * 200
    )
    
    # Añadir algo de ruido al precio
    data['price'] = price * np.random.uniform(0.8, 1.2, n_samples)
    
    # Generar fechas de venta previas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    data['prev_sold_date'] = dates
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Calcular características adicionales
    df['total_rooms'] = df['bed'] + df['bath']
    df['price_per_sqft'] = df['price'] / df['house_size']
    
    return df

if __name__ == "__main__":
    # Generar datos
    df = generate_synthetic_data(1000)
    
    # Guardar en formato parquet
    output_path = "/opt/airflow/data/raw/synthetic_batch_1.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Datos sintéticos guardados en: {output_path}")
    print(f"Forma del dataset: {df.shape}")
    print("\nPrimeras 5 filas:")
    print(df.head())
    print("\nEstadísticas descriptivas:")
    print(df.describe()) 