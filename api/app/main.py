import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, Table, Column, Float, DateTime, MetaData
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import logging

from .config import settings
from .schemas import PropertyFeatures, PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MLflow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = settings.MINIO_ENDPOINT
os.environ['AWS_ACCESS_KEY_ID'] = settings.AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = settings.AWS_SECRET_ACCESS_KEY
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

# Initialize FastAPI app
app = FastAPI(
    title="Real Estate Price Prediction API",
    description="API for predicting real estate prices using MLflow models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus metrics
PREDICTION_REQUEST_COUNT = Counter(
    'prediction_request_count', 
    'Number of prediction requests received'
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds', 
    'Time spent processing prediction requests'
)

# Initialize database connection and table
engine = create_engine(settings.RAW_DATA_DB_URI)
metadata = MetaData()

predictions_table = Table(
    'predictions', 
    metadata,
    Column('timestamp', DateTime, primary_key=True),
    Column('bed', Float),
    Column('bath', Float),
    Column('acre_lot', Float),
    Column('house_size', Float),
    Column('total_rooms', Float),
    Column('predicted_price', Float)
)

# Create table if it doesn't exist
metadata.create_all(engine)

# Cache for model and scaler
model_cache = {
    "model": None,
    "scaler": None,
    "run_id": None
}

def load_model():
    """Load the best model from the experiment"""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(settings.MLFLOW_MODEL_NAME)
        
        if experiment is None:
            raise HTTPException(status_code=500, detail=f"Experiment {settings.MLFLOW_MODEL_NAME} not found")
        
        # Get the best run based on R2 score
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_r2 DESC"],
            max_results=1
        )
        
        if not runs:
            raise HTTPException(status_code=500, detail="No runs found in the experiment")
        
        best_run = runs[0]
        
        # Only reload if run_id has changed
        if model_cache["run_id"] != best_run.info.run_id:
            logger.info(f"Loading model from run: {best_run.info.run_id}")
            model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
            scaler = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/scaler")
            
            model_cache["model"] = model
            model_cache["scaler"] = scaler
            model_cache["run_id"] = best_run.info.run_id
        
        return model_cache["model"], model_cache["scaler"], best_run.info.run_id
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/metrics")
def metrics():
    """Endpoint for Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(features: PropertyFeatures):
    """
    Make a prediction for house price based on input features
    """
    PREDICTION_REQUEST_COUNT.inc()
    
    with PREDICTION_LATENCY.time():
        try:
            # Load model and scaler
            model, scaler, run_id = load_model()
            
            # Prepare features
            feature_dict = features.model_dump()
            df = pd.DataFrame([feature_dict])
            
            # Scale features
            scaled_features = scaler.transform(df)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            
            # Store prediction in database
            with engine.connect() as conn:
                conn.execute(
                    predictions_table.insert().values(
                        timestamp=datetime.now(),
                        **feature_dict,
                        predicted_price=prediction
                    )
                )
                conn.commit()
            
            return PredictionResponse(
                predicted_price=prediction,
                model_version=run_id,
                model_stage=settings.MODEL_STAGE
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 