import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import mlflow
from datetime import datetime
import shap
import os
from .config import settings

# Configuración de la página
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Configuración de MLflow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = settings.MINIO_ENDPOINT
os.environ['AWS_ACCESS_KEY_ID'] = settings.AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = settings.AWS_SECRET_ACCESS_KEY
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

def get_model_history():
    """Get model history from MLflow"""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(settings.MLFLOW_MODEL_NAME)
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.r2 DESC"]
        )
        return runs
    return []

def make_prediction(features):
    """Make prediction using FastAPI endpoint"""
    try:
        response = requests.post(
            f"{settings.API_URL}/predict",
            json=features
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def get_shap_explanation(features):
    """Get SHAP explanation for prediction"""
    try:
        # Load the production model from MLflow
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(settings.MLFLOW_MODEL_NAME)
        if experiment:
            production_runs = [run for run in client.search_runs(
                experiment_ids=[experiment.experiment_id]
            ) if run.data.tags.get("production") == "true"]
            
            if production_runs:
                run = production_runs[0]
                model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
                
                # Create SHAP explainer
                explainer = shap.LinearExplainer(model, pd.DataFrame([features]))
                shap_values = explainer.shap_values(pd.DataFrame([features]))
                
                # Create feature importance plot
                feature_importance = pd.DataFrame({
                    'Feature': list(features.keys()),
                    'Importance': np.abs(shap_values).mean(0)
                })
                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                
                return feature_importance
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {str(e)}")
    return None

def plot_model_performance(model_history):
    """Genera gráficos de rendimiento de modelos."""
    fig = go.Figure()
    
    for metric in ['rmse', 'mae', 'r2']:
        if metric in model_history.columns:
            fig.add_trace(go.Scatter(
                x=model_history['creation_timestamp'],
                y=model_history[metric],
                name=metric.upper(),
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title="Evolución del Rendimiento de Modelos",
        xaxis_title="Tiempo",
        yaxis_title="Valor de Métrica",
        hovermode='x unified'
    )
    
    return fig

def get_model_explanation(model, features):
    """Genera explicaciones SHAP para la predicción."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    fig = shap.plots.waterfall(shap_values[0])
    return fig

def main():
    st.title("🏠 Real Estate Price Predictor")
    st.markdown("""
    Esta aplicación te ayuda a predecir el precio de una propiedad basándose en sus características.
    Utiliza un modelo de Machine Learning entrenado con datos reales del mercado inmobiliario.
    """)
    
    # Sidebar - Model Information
    st.sidebar.title("Model Information")
    runs = get_model_history()
    if runs:
        production_runs = [run for run in runs if run.data.tags.get("production") == "true"]
        if production_runs:
            current_model = production_runs[0]
            st.sidebar.success(f"Current Production Model")
            st.sidebar.metric("R² Score", f"{current_model.data.metrics.get('r2', 0):.4f}")
            st.sidebar.metric("MSE", f"{current_model.data.metrics.get('mse', 0):.4f}")
            
            # Model parameters
            st.sidebar.subheader("Model Parameters")
            params = current_model.data.params
            for param, value in params.items():
                st.sidebar.text(f"{param}: {value}")
    
    # Main content - Prediction Interface
    st.header("Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bed = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
        acre_lot = st.number_input("Lot Size (acres)", min_value=0.1, max_value=10.0, value=0.5)
        
    with col2:
        house_size = st.number_input("House Size (sq ft)", min_value=500, max_value=10000, value=2000)
        total_rooms = st.number_input("Total Rooms", min_value=2, max_value=20, value=5)
    
    if st.button("Predict Price"):
        features = {
            "bed": float(bed),
            "bath": float(bath),
            "acre_lot": float(acre_lot),
            "house_size": float(house_size),
            "total_rooms": float(total_rooms)
        }
        
        result = make_prediction(features)
        if result:
            st.success(f"Predicted Price: ${result['predicted_price']:,.2f}")
            st.info(f"Model Version: {result['model_version']}")
            st.info(f"Model Stage: {result['model_stage']}")
            
            # Get and display SHAP explanation
            st.subheader("Feature Importance (SHAP)")
            feature_importance = get_shap_explanation(features)
            if feature_importance is not None:
                fig = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance'
                )
                st.plotly_chart(fig)
    
    # Model History
    st.header("Model History")
    if runs:
        history_data = []
        for run in runs:
            history_data.append({
                "Run ID": run.info.run_id[:8],
                "R² Score": run.data.metrics.get("r2", 0),
                "MSE": run.data.metrics.get("mse", 0),
                "Status": "Production" if run.data.tags.get("production") == "true" else "Archived",
                "Timestamp": pd.to_datetime(run.info.start_time).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df)
        
        # Plot model performance history
        fig = px.line(history_df, x="Timestamp", y="R² Score", title="Model Performance History")
        st.plotly_chart(fig)
    else:
        st.warning("No model history available")

    # Información adicional
    st.sidebar.header("Acerca de")
    st.sidebar.markdown("""
    Esta aplicación es parte de un proyecto MLOps que incluye:
    - Pipeline de datos automatizado
    - Entrenamiento continuo del modelo
    - Monitoreo de rendimiento
    - API REST para predicciones
    """)

    # Mostrar timestamp
    st.sidebar.markdown(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 