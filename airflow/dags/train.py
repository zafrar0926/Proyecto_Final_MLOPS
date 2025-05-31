import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import mlflow
from mlflow.tracking import MlflowClient
import logging
from sqlalchemy import create_engine

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_best_run_metric(client, experiment_name, metric_name):
    """Obtiene el mejor valor de una métrica de todos los experimentos anteriores."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return float('-inf')
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1
    )
    
    if not runs:
        return float('-inf')
    
    return runs[0].data.metrics.get(metric_name, float('-inf'))

def main(db_uri):
    """
    Función principal para entrenar el modelo con GridSearch.
    """
    try:
        logger.info("Iniciando entrenamiento del modelo con GridSearch...")
        
        # Configurar MLflow
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'
        os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        experiment_name = "real_estate_prediction_gridsearch"
        mlflow.set_experiment(experiment_name)
        
        # Crear cliente MLflow
        client = MlflowClient()
        
        # Cargar datos
        logger.info("Cargando datos...")
        engine = create_engine(db_uri)
        df = pd.read_sql_table('clean_properties', engine)
        
        # Preparar datos
        X = df[['bed', 'bath', 'acre_lot', 'house_size', 'total_rooms']]
        y = df['price']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Escalar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Definir parámetros para GridSearch
        param_grid = {
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        }
        
        # Crear modelo base
        model = ElasticNet(random_state=42, max_iter=1000)
        
        # Crear GridSearch
        logger.info("Iniciando GridSearch...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # Entrenar GridSearch
        grid_search.fit(X_train_scaled, y_train)
        
        # Obtener el mejor modelo
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Mejores parámetros encontrados: {best_params}")
        
        # Evaluar mejor modelo
        y_pred = best_model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Métricas del mejor modelo:")
        logger.info(f"R2 Score: {test_r2}")
        logger.info(f"MSE: {test_mse}")
        logger.info(f"MAE: {test_mae}")
        
        # Obtener el mejor R2 de experimentos anteriores
        best_previous_r2 = get_best_run_metric(client, experiment_name, "test_r2")
        logger.info(f"Mejor R2 anterior: {best_previous_r2}")
        
        # Registrar el experimento en MLflow
        with mlflow.start_run(run_name="grid_search_best_model"):
            # Registrar parámetros
            mlflow.log_params(best_params)
            
            # Registrar métricas
            mlflow.log_metrics({
                "test_r2": test_r2,
                "test_mse": test_mse,
                "test_mae": test_mae
            })
            
            # Solo registrar el modelo si es mejor que el anterior
            if test_r2 > best_previous_r2:
                logger.info("¡Nuevo mejor modelo encontrado! Registrando en MLflow...")
                # Registrar el modelo
                mlflow.sklearn.log_model(
                    best_model, 
                    "model",
                    registered_model_name="real_estate_elasticnet"
                )
                
                # Registrar el scaler como artefacto
                mlflow.sklearn.log_model(
                    scaler,
                    "scaler"
                )
            else:
                logger.info("El modelo actual no supera al mejor modelo anterior.")
        
        return best_model, scaler, test_r2
        
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main("postgresql://cleandata:cleandata123@clean_data_db:5432/clean_data") 