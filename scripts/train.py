import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import logging
import sys
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_features(df):
    """Prepara las características para el modelo."""
    try:
        # Seleccionar características numéricas relevantes
        features = ['bed', 'bath', 'acre_lot', 'house_size', 'total_rooms']
        target = 'price'
        
        # Separar features y target
        X = df[features]
        y = df[target]
        
        # Escalar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, features, scaler
        
    except Exception as e:
        logger.error(f"Error en la preparación de características: {e}")
        raise

def train_model(data_path, params=None):
    """Entrena un modelo de regresión lineal y lo registra en MLflow."""
    try:
        # Configurar MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("real_estate_prediction")
        
        # Cargar datos
        df = pd.read_parquet(data_path)
        logger.info(f"Datos cargados: {df.shape[0]} registros")
        
        # Preparar datos
        X, y, features, scaler = prepare_features(df)
        
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Iniciar run de MLflow
        with mlflow.start_run() as run:
            # Entrenar modelo
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Logging de parámetros y métricas
            mlflow.log_params({
                "model_type": "LinearRegression",
                "test_size": 0.2,
                "random_state": 42,
                "features": features
            })
            
            mlflow.log_metrics({
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            })
            
            # Registrar modelo
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="real_estate_model"
            )
            
            # Registrar scaler como artefacto
            mlflow.sklearn.log_model(
                scaler,
                "scaler"
            )
            
            # Transicionar modelo a producción si el R² es mejor que 0.7
            if r2 > 0.7:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name="real_estate_model",
                    version=1,  # La versión más reciente
                    stage="Production"
                )
                logger.info("Modelo promovido a producción")
            
            logger.info(f"Entrenamiento completado. Métricas: RMSE={rmse:.2f}, R2={r2:.2f}")
            
            return {
                "rmse": rmse,
                "r2": r2,
                "mae": mae,
                "run_id": run.info.run_id
            }
            
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        train_model(data_path)
    else:
        logger.error("Debe proporcionar la ruta a los datos de entrenamiento") 