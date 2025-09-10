from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Delivery Time Prediction API",
    description="API for predicting delivery times using ML models",
    version="1.0.0"
)


MODELS_DIR = Path("/models")
MODELS_DIR.mkdir(exist_ok=True)

active_model_name: Optional[str] = None
active_model_pipeline = None


class PredictionRequest(BaseModel):
    """Input features for delivery time prediction"""
    qty: int = Field(..., description="Quantity of items")
    weight_kg: float = Field(..., description="Weight in kilograms")
    unit_price_usd: float = Field(..., description="Unit price in USD")
    distance_km: float = Field(..., description="Distance in kilometers")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    weekday: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    ship_country: str = Field(..., description="Shipping country code")
    category: str = Field(..., description="Product category")
    carrier: str = Field(..., description="Shipping carrier name")

    class Config:
        json_schema_extra = {
            "example": {
                "qty": 2,
                "weight_kg": 0.8,
                "unit_price_usd": 12.5,
                "distance_km": 350.0,
                "hour_of_day": 14,
                "weekday": 2,
                "ship_country": "IT",
                "category": "Widgets",
                "carrier": "DHL"
            }
        }


class PredictionResponse(BaseModel):
    """Response with predicted delivery days"""
    delivery_days: float = Field(..., description="Predicted delivery time in days")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service health status")


class ModelInfo(BaseModel):
    """Information about a stored model"""
    name: str
    active: bool
    file_size_mb: float


@app.get("/health/", response_model=HealthResponse, tags=["Health"])
async def health_check():
   
    return {"status": "alive"}


@app.post("/model/", tags=["Model Management"])
async def upload_model(
    name: str = Form(..., pattern="^[a-zA-Z0-9_]+$"),
    file: UploadFile = File(...),
    activate: bool = Form(True)
):
  
    global active_model_name, active_model_pipeline
    
    if not file.filename.endswith('.joblib'):
        raise HTTPException(status_code=400, detail="Only .joblib files are accepted")
    
    model_path = MODELS_DIR / f"{name}.joblib"
    
    try:
        with model_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        test_model = joblib.load(model_path)
        

        if activate:
            active_model_name = name
            active_model_pipeline = test_model
            logger.info(f"Model '{name}' uploaded and activated")
        else:
            logger.info(f"Model '{name}' uploaded")
        
        return {"message": f"Model '{name}' uploaded successfully", "activated": activate}
    
    except Exception as e:
        # Clean up on failure
        if model_path.exists():
            model_path.unlink()
        raise HTTPException(status_code=400, detail=f"Failed to process model: {str(e)}")


@app.get("/model/", response_model=List[ModelInfo], tags=["Model Management"])
async def list_models():

    models = []
    
    for model_file in MODELS_DIR.glob("*.joblib"):
        name = model_file.stem
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        models.append(ModelInfo(
            name=name,
            active=(name == active_model_name),
            file_size_mb=round(file_size_mb, 2)
        ))
    
    return models


@app.put("/model/{name}/activate", tags=["Model Management"])
async def activate_model(name: str):
    global active_model_name, active_model_pipeline
    
    model_path = MODELS_DIR / f"{name}.joblib"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    
    try:
        active_model_pipeline = joblib.load(model_path)
        active_model_name = name
        logger.info(f"Model '{name}' activated")
        return {"message": f"Model '{name}' activated successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {str(e)}")


@app.delete("/model/{name}", tags=["Model Management"])
async def delete_model(name: str):
    """
    Delete a stored model.
    Cannot delete the currently active model.
    """
    global active_model_name, active_model_pipeline
    
    if name == active_model_name:
        raise HTTPException(status_code=400, detail="Cannot delete the active model")
    
    model_path = MODELS_DIR / f"{name}.joblib"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    
    try:
        model_path.unlink()
        logger.info(f"Model '{name}' deleted")
        return {"message": f"Model '{name}' deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


# Prediction endpoint
@app.post("/predict/", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):

    global active_model_pipeline
    
    # Check if there's an active model
    if active_model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="No active model. Please upload and activate a model first."
        )
    
    try:
        # Convert request to DataFrame with exact feature order
        input_data = pd.DataFrame([{
            'qty': request.qty,
            'weight_kg': request.weight_kg,
            'unit_price_usd': request.unit_price_usd,
            'distance_km': request.distance_km,
            'hour_of_day': request.hour_of_day,
            'weekday': request.weekday,
            'ship_country': request.ship_country,
            'category': request.category,
            'carrier': request.carrier
        }])
        
        # Make prediction
        prediction = active_model_pipeline.predict(input_data)
        
        # Return prediction
        return {"delivery_days": float(prediction[0])}
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


# Startup event - try to load an existing model
@app.on_event("startup")
async def startup_event():

    global active_model_name, active_model_pipeline
    
    # Check for existing models
    model_files = list(MODELS_DIR.glob("*.joblib"))
    
    if model_files:
        # Load the first model found
        first_model = model_files[0]
        name = first_model.stem
        
        try:
            active_model_pipeline = joblib.load(first_model)
            active_model_name = name
            logger.info(f"Auto-loaded model '{name}' on startup")
        except Exception as e:
            logger.error(f"Failed to auto-load model: {str(e)}")
    else:
        logger.info("No models found on startup. Please upload a model.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)