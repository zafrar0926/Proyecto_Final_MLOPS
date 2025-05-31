import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import logging
import json
from datetime import datetime
import os
import shap
import matplotlib.pyplot as plt

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path):
    """Carga y preprocesa los datos para el entrenamiento."""
    try:
        # Cargar datos
        df = pd.read_parquet(data_path)
        logger.info(f"Datos cargados: {df.shape}")
        
        # Separar características y objetivo
        X = df.drop(['price'], axis=1)
        y = df['price']
        
        # Preprocesamiento
        # 1. Convertir fechas a características numéricas
        X['prev_sold_date'] = pd.to_datetime(X['prev_sold_date'])
        X['days_since_prev_sale'] = (datetime.now() - X['prev_sold_date']).dt.days
        X = X.drop('prev_sold_date', axis=1)
        
        # 2. Codificación de variables categóricas
        categorical_cols = ['status', 'city', 'state']
        X = pd.get_dummies(X, columns=categorical_cols)
        
        # 3. Escalar características numéricas
        numeric_cols = ['bed', 'bath', 'acre_lot', 'house_size', 'days_since_prev_sale']
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        
        return X, y, scaler
        
    except Exception as e:
        logger.error(f"Error en el preprocesamiento: {e}")
        raise

def train_model(X, y, params=None):
    """Entrena el modelo con los parámetros especificados."""
    try:
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Parámetros por defecto si no se especifican
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 5,
                'learning_rate': 0.1,
                'loss': 'squared_error'
            }
        
        # Entrenar modelo
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Calcular importancia de características
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Generar explicaciones SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        return model, metrics, feature_importance, (X_test, y_test, y_pred, shap_values)
        
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        raise

def log_model_to_mlflow(model, metrics, feature_importance, evaluation_data, params, run_name):
    """Registra el modelo y sus métricas en MLflow."""
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("real_estate_prediction")
        
        with mlflow.start_run(run_name=run_name):
            # Registrar parámetros
            mlflow.log_params(params)
            
            # Registrar métricas
            mlflow.log_metrics(metrics)
            
            # Registrar modelo
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="real_estate_model"
            )
            
            # Guardar importancia de características
            feature_importance_path = "feature_importance.csv"
            feature_importance.to_csv(feature_importance_path, index=False)
            mlflow.log_artifact(feature_importance_path)
            
            # Generar y guardar gráficos de evaluación
            X_test, y_test, y_pred, shap_values = evaluation_data
            
            # 1. Gráfico de dispersión predicción vs real
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Precio Real')
            plt.ylabel('Precio Predicho')
            plt.title('Predicción vs Real')
            plt.savefig("prediction_vs_real.png")
            mlflow.log_artifact("prediction_vs_real.png")
            
            # 2. Gráfico SHAP
            shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig("shap_summary.png")
            mlflow.log_artifact("shap_summary.png")
            
            # Registrar información adicional
            mlflow.log_dict(
                {
                    "feature_names": X_test.columns.tolist(),
                    "training_timestamp": datetime.now().isoformat(),
                    "data_shape": X_test.shape
                },
                "model_info.json"
            )
            
        logger.info(f"Modelo registrado exitosamente en MLflow con métricas: {metrics}")
        
    except Exception as e:
        logger.error(f"Error al registrar en MLflow: {e}")
        raise

def main(data_path, params=None):
    """Función principal que ejecuta el pipeline de entrenamiento."""
    try:
        # 1. Cargar y preprocesar datos
        X, y, scaler = load_and_preprocess_data(data_path)
        
        # 2. Entrenar modelo
        model, metrics, feature_importance, evaluation_data = train_model(X, y, params)
        
        # 3. Registrar en MLflow
        run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_model_to_mlflow(
            model,
            metrics,
            feature_importance,
            evaluation_data,
            params or {},
            run_name
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error en el pipeline de entrenamiento: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python train.py <path_to_data>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    metrics = main(data_path)
    print(f"Entrenamiento completado. Métricas: {json.dumps(metrics, indent=2)}") 