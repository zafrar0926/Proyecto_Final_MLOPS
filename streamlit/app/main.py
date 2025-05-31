import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import mlflow
import shap
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from .config import settings

# Configuración de la página
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.25rem 0.5rem rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar para navegación
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Make Prediction", "Model History", "Model Explainability"])

# Configuración de MLflow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = settings.MINIO_ENDPOINT
os.environ['AWS_ACCESS_KEY_ID'] = settings.AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = settings.AWS_SECRET_ACCESS_KEY
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

def get_model_history():
    """Get model history from MLflow"""
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    
    models_info = []
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id)
        for run in runs:
            models_info.append({
                "run_id": run.info.run_id,
                "experiment_name": exp.name,
                "metrics": run.data.metrics,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "tags": run.data.tags
            })
    return models_info

def make_prediction(data):
    """Make prediction using FastAPI endpoint"""
    try:
        response = requests.post(
            f"{settings.API_URL}/predict",
            json=data
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

if page == "Make Prediction":
    st.title("🏠 Real Estate Price Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Details")
        beds = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        baths = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
        house_size = st.number_input("House Size (sq ft)", min_value=500, max_value=10000, value=1500)
        acre_lot = st.number_input("Lot Size (acres)", min_value=0.1, max_value=10.0, value=0.5)
        
    with col2:
        st.subheader("Location Details")
        zip_code = st.text_input("ZIP Code", "12345")
        city = st.text_input("City", "Sample City")
        state = st.text_input("State", "CA")
        
    if st.button("Predict Price"):
        with st.spinner("Calculating prediction..."):
            prediction_data = {
                "bed": beds,
                "bath": baths,
                "house_size": house_size,
                "acre_lot": acre_lot,
                "zip_code": zip_code,
                "city": city,
                "state": state
            }
            
            try:
                result = make_prediction(prediction_data)
                st.success("Prediction Complete!")
                
                st.markdown("### Predicted Price")
                st.markdown(f"""
                <div class="prediction-box">
                    <h1 style='text-align: center; color: #4CAF50;'>${result['prediction']:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

elif page == "Model History":
    st.title("📈 Model History and Performance")
    
    try:
        models = get_model_history()
        
        # Crear DataFrame con la información de los modelos
        df_models = pd.DataFrame(models)
        
        # Mostrar métricas a lo largo del tiempo
        st.subheader("Model Performance Over Time")
        fig = go.Figure()
        
        for metric in ['rmse', 'mae', 'r2']:
            if metric in df_models.columns:
                fig.add_trace(go.Scatter(
                    x=df_models['start_time'],
                    y=df_models[metric],
                    name=metric.upper(),
                    mode='lines+markers'
                ))
        
        fig.update_layout(
            title="Metrics Evolution",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla detallada de modelos
        st.subheader("Detailed Model Information")
        st.dataframe(df_models)
        
    except Exception as e:
        st.error(f"Error fetching model history: {str(e)}")

elif page == "Model Explainability":
    st.title("🔍 Model Explainability (SHAP)")
    
    st.info("This section shows the SHAP values for the current production model, helping understand how each feature impacts the predictions.")
    
    try:
        # Aquí implementaremos la explicabilidad SHAP
        # Por ahora es un placeholder
        st.write("SHAP analysis will be implemented here")
        
    except Exception as e:
        st.error(f"Error generating SHAP analysis: {str(e)}")

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