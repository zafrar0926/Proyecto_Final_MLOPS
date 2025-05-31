from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_URL: str = "http://fastapi:8000"
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    MLFLOW_MODEL_NAME: str = "real_estate_prediction_gridsearch"
    MINIO_ENDPOINT: str = "http://minio:9000"
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadmin"

settings = Settings() 