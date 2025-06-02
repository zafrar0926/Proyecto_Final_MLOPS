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
import shap
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

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

def generate_shap_plots(model, X_train_scaled, X_test_scaled, feature_names):
    """Genera y guarda plots de SHAP para interpretabilidad del modelo."""
    try:
        # Calcular valores SHAP
        explainer = shap.LinearExplainer(model, X_train_scaled)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Guardar resumen de importancia de características
        plt.figure()
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
        plt.savefig('shap_summary.png')
        plt.close()
        
        # Guardar gráfico de dependencia para cada característica
        shap_interaction_values = None
        try:
            shap_interaction_values = explainer.shap_interaction_values(X_test_scaled)
        except:
            logger.warning("No se pudieron calcular los valores de interacción SHAP")
        
        feature_importance = {}
        for i, feature in enumerate(feature_names):
            importance = np.abs(shap_values[:, i]).mean()
            feature_importance[feature] = float(importance)
            
            # Gráfico de dependencia individual
            plt.figure()
            shap.dependence_plot(i, shap_values, X_test_scaled, feature_names=feature_names, show=False)
            plt.savefig(f'shap_dependence_{feature}.png')
            plt.close()
        
        return {
            'feature_importance': feature_importance,
            'shap_values': shap_values.tolist(),
            'interaction_values': shap_interaction_values.tolist() if shap_interaction_values is not None else None
        }
        
    except Exception as e:
        logger.error(f"Error generando plots SHAP: {e}")
        return None

def transition_model_to_production(client, model_name, version, metrics):
    """Transiciona un modelo a producción si cumple los criterios."""
    try:
        # Obtener la versión actual en producción
        prod_model = client.get_latest_versions(model_name, stages=["Production"])
        
        if not prod_model:
            logger.info("No hay modelo en producción. Transitioning nuevo modelo...")
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            return True
            
        # Comparar métricas
        current_metrics = json.loads(prod_model[0].description or '{}')
        if metrics['test_r2'] > current_metrics.get('test_r2', float('-inf')):
            # Archivar modelo actual
            client.transition_model_version_stage(
                name=model_name,
                version=prod_model[0].version,
                stage="Archived"
            )
            
            # Promover nuevo modelo
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error transitioning modelo a producción: {e}")
        return False

def prepare_data(df):
    """Prepara los datos para el entrenamiento."""
    try:
        # Seleccionar features
        features = ['bed', 'bath', 'acre_lot', 'house_size', 'total_rooms']
        target = 'price'
        
        # Separar features y target
        X = df[features]
        y = df[target]
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features
        
    except Exception as e:
        logger.error(f"Error preparando datos: {e}")
        raise

def train_model(df):
    """Entrena un modelo de Random Forest."""
    try:
        # Preparar datos
        X_train, X_test, y_train, y_test, scaler, features = prepare_data(df)
        
        # Iniciar run de MLflow
        with mlflow.start_run() as run:
            # Entrenar modelo
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Hacer predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred))
            }
            
            # Registrar parámetros y métricas en MLflow
            mlflow.log_params({
                'n_estimators': 100,
                'max_depth': 10,
                'features': features
            })
            
            mlflow.log_metrics(metrics)
            
            # Registrar modelo
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="real_estate_model"
            )
            
            # Registrar scaler
            mlflow.sklearn.log_model(
                scaler,
                "scaler",
                registered_model_name="real_estate_scaler"
            )
            
            logger.info(f"Modelo entrenado exitosamente. Métricas: {metrics}")
            return model, metrics
            
    except Exception as e:
        logger.error(f"Error entrenando modelo: {e}")
        raise

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
        
        # Entrenar modelo
        model, metrics = train_model(df)
        
        # Generar explicabilidad SHAP
        shap_data = generate_shap_plots(model, X_train_scaled, X_test_scaled, features)
        
        # Registrar el experimento en MLflow
        with mlflow.start_run(run_name="grid_search_best_model") as run:
            # Registrar parámetros
            mlflow.log_params({
                'n_estimators': 100,
                'max_depth': 10,
                'features': features
            })
            
            # Registrar métricas
            mlflow.log_metrics(metrics)
            
            # Registrar plots SHAP
            if os.path.exists('shap_summary.png'):
                mlflow.log_artifact('shap_summary.png', 'shap_plots')
            
            for feature in features:
                if os.path.exists(f'shap_dependence_{feature}.png'):
                    mlflow.log_artifact(f'shap_dependence_{feature}.png', 'shap_plots')
            
            if shap_data:
                mlflow.log_dict(shap_data, 'shap_values.json')
            
            # Registrar el modelo
            model_info = mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="real_estate_model"
            )
            
            # Registrar el scaler
            mlflow.sklearn.log_model(
                scaler,
                "scaler",
                registered_model_name="real_estate_scaler"
            )
            
            # Intentar transicionar a producción
            if transition_model_to_production(client, "real_estate_model", model_info.version, metrics):
                logger.info("Modelo promovido a producción")
            else:
                logger.info("Modelo registrado pero no promovido a producción")
        
        # Limpiar archivos temporales
        for file in ['shap_summary.png'] + [f'shap_dependence_{f}.png' for f in features]:
            if os.path.exists(file):
                os.remove(file)
        
        return model, scaler, metrics
        
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main("postgresql://cleandata:cleandata123@clean_data_db:5432/clean_data") 