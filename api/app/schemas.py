from pydantic import BaseModel, Field
from typing import Optional

class PropertyFeatures(BaseModel):
    bed: float = Field(..., description="Number of bedrooms")
    bath: float = Field(..., description="Number of bathrooms")
    acre_lot: float = Field(..., description="Lot size in acres")
    house_size: float = Field(..., description="House size in square feet")
    total_rooms: float = Field(..., description="Total number of rooms")

class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str
    model_stage: str 