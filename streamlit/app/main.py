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
import matplotlib.pyplot as plt

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

def get_shap_explanation(features, model=None):
    """Get SHAP explanation for prediction"""
    try:
        # Load the production model from MLflow if not provided
        if model is None:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(settings.MLFLOW_MODEL_NAME)
            if experiment:
                production_runs = [run for run in client.search_runs(
                    experiment_ids=[experiment.experiment_id]
                ) if run.data.tags.get("production") == "true"]
                
                if production_runs:
                    run = production_runs[0]
                    model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
                else:
                    st.error("No production model found")
                    return None
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Create SHAP explainer based on model type
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, features_df)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features_df)
        
        # If shap_values is a list (for multi-output models), take the first element
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'Feature': list(features.keys()),
            'Importance': np.abs(shap_values).mean(0),
            'Value': shap_values[0]
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        return {
            'feature_importance': feature_importance,
            'shap_values': shap_values,
            'features_df': features_df,
            'explainer': explainer
        }
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
    
    st.info("This section shows how different features impact the model's predictions using SHAP (SHapley Additive exPlanations) values.")
    
    try:
        # Create sample data for explanation
        sample_data = {
            "bed": 3,
            "bath": 2,
            "house_size": 1500,
            "acre_lot": 0.5,
            "zip_code": "12345",
            "city": "Sample City",
            "state": "CA"
        }
        
        # Get SHAP explanation
        shap_explanation = get_shap_explanation(sample_data)
        
        if shap_explanation:
            st.subheader("Feature Importance")
            
            # Plot feature importance
            fig_importance = px.bar(
                shap_explanation['feature_importance'],
                x='Importance',
                y='Feature',
                orientation='h',
                title='Global Feature Importance'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # SHAP Summary Plot
            st.subheader("SHAP Summary Plot")
            fig_summary = plt.figure()
            shap.summary_plot(
                shap_explanation['shap_values'],
                shap_explanation['features_df'],
                show=False
            )
            st.pyplot(fig_summary)
            
            # Feature Impact Analysis
            st.subheader("Feature Impact Analysis")
            for idx, row in shap_explanation['feature_importance'].iterrows():
                impact = "positive" if row['Value'] > 0 else "negative"
                magnitude = abs(row['Value'])
                
                st.markdown(f"""
                **{row['Feature']}**:
                - Impact: {impact.title()}
                - Magnitude: {magnitude:.4f}
                - Current Value: {sample_data[row['Feature']]}
                """)
            
            # Interactive Feature Analysis
            st.subheader("Interactive Feature Analysis")
            selected_feature = st.selectbox(
                "Select a feature to analyze:",
                shap_explanation['feature_importance']['Feature'].tolist()
            )
            
            if selected_feature:
                feature_idx = list(sample_data.keys()).index(selected_feature)
                st.write(f"Analyzing the impact of {selected_feature}:")
                
                # Create dependence plot
                fig_dependence = plt.figure()
                shap.dependence_plot(
                    feature_idx,
                    shap_explanation['shap_values'],
                    shap_explanation['features_df'],
                    show=False
                )
                st.pyplot(fig_dependence)
        
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