from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    MLFLOW_MODEL_NAME: str = "real_estate_prediction_gridsearch"
    MINIO_ENDPOINT: str = "http://minio:9000"
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadmin"
    RAW_DATA_DB_URI: str = "postgresql://rawdata:rawdata123@raw_data_db:5432/raw_data"
    MODEL_STAGE: str = "None"

settings = Settings()