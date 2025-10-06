from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List
import uvicorn
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predictor.pipelines.prediction_pipeline import PredictionPipeline, CustomData
from src.predictor.exception import TenYearChdException
from src.predictor.logger import logging

# Initialize FastAPI app
app = FastAPI(
    title="10-Year CHD Risk Prediction API",
    description="API for predicting 10-year Coronary Heart Disease risk",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction pipeline
prediction_pipeline = PredictionPipeline()

# Pydantic models for request/response
class PatientData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 45,
                "sex": "M",
                "is_smoking": "YES",
                "cigsperday": 20.0,
                "bpmeds": 0.0,
                "prevalentstroke": 0,
                "prevalenthyp": 0,
                "diabetes": 0,
                "totchol": 250.0,
                "sysbp": 140.0,
                "diabp": 90.0,
                "bmi": 28.5,
                "heartrate": 75.0,
                "glucose": 100.0,
                "education": 2
            }
        }
    )

    age: int = Field(..., ge=20, le=100, description="Age of the patient (20-100)")
    sex: str = Field(..., description="Sex of the patient (M/F)")
    is_smoking: str = Field(..., description="Smoking status (YES/NO)")
    cigsperday: float = Field(..., ge=0, le=100, description="Cigarettes per day (0-100)")
    bpmeds: float = Field(..., ge=0, le=1, description="On BP medication (0/1)")
    prevalentstroke: int = Field(..., ge=0, le=1, description="Previous stroke (0/1)")
    prevalenthyp: int = Field(..., ge=0, le=1, description="Prevalent hypertension (0/1)")
    diabetes: int = Field(..., ge=0, le=1, description="Diabetes status (0/1)")
    totchol: float = Field(..., ge=100, le=600, description="Total cholesterol (100-600)")
    sysbp: float = Field(..., ge=80, le=300, description="Systolic BP (80-300)")
    diabp: float = Field(..., ge=40, le=200, description="Diastolic BP (40-200)")
    bmi: float = Field(..., ge=10, le=70, description="Body Mass Index (10-70)")
    heartrate: float = Field(..., ge=40, le=200, description="Heart rate (40-200)")
    glucose: float = Field(..., ge=40, le=500, description="Glucose level (40-500)")
    education: int = Field(..., ge=1, le=4, description="Education level (1-4)")

    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v: str) -> str:
        if v.upper() not in ['M', 'F']:
            raise ValueError('Sex must be M or F')
        return v.upper()

    @field_validator('is_smoking')
    @classmethod
    def validate_smoking(cls, v: str) -> str:
        if v.upper() not in ['YES', 'NO']:
            raise ValueError('is_smoking must be YES or NO')
        return v.upper()


class PredictionResponse(BaseModel):
    success: bool
    prediction: int
    risk_level: str
    probability_low_risk: Optional[float] = None
    probability_high_risk: Optional[float] = None
    message: str


# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint with API documentation
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>10-Year CHD Risk Prediction API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            h1 {
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                opacity: 0.9;
                margin-bottom: 30px;
            }
            .endpoints {
                background: rgba(255, 255, 255, 0.15);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .endpoint {
                margin: 15px 0;
                padding: 15px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            .method {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 5px;
                font-weight: bold;
                margin-right: 10px;
            }
            .get { background: #4CAF50; }
            .post { background: #2196F3; }
            code {
                background: rgba(0, 0, 0, 0.3);
                padding: 2px 6px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
            }
            a {
                color: #ffeb3b;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 10-Year CHD Risk Prediction API</h1>
            <p class="subtitle">Predict Coronary Heart Disease Risk using Machine Learning</p>
            
            <div class="endpoints">
                <h2>📚 Available Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <code>/</code>
                    <p>This page - API documentation</p>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <code>/health</code>
                    <p>Check API health status</p>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <code>/predict</code>
                    <p>Make a prediction for CHD risk</p>
                    <p><strong>Required fields:</strong> age, sex, is_smoking, cigsperday, bpmeds, prevalentstroke, prevalenthyp, diabetes, totchol, sysbp, diabp, bmi, heartrate, glucose, education</p>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <code>/batch-predict</code>
                    <p>Make predictions for multiple patients</p>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <code>/model-info</code>
                    <p>Get information about the loaded model</p>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <p>📖 Interactive API Documentation: <a href="/docs">/docs</a></p>
                <p>📊 Alternative API Docs: <a href="/redoc">/redoc</a></p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check if model and preprocessor are loaded
        if prediction_pipeline.model is None:
            prediction_pipeline.load_artifacts()
        
        return {
            "status": "healthy",
            "model_loaded": prediction_pipeline.model is not None,
            "preprocessor_loaded": prediction_pipeline.preprocessor is not None,
            "message": "API is running successfully"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "message": "API is running but model loading failed"
            }
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """
    Predict 10-year CHD risk for a single patient
    """
    try:
        logging.info("Received prediction request")
        
        # Create CustomData object
        custom_data = CustomData(
            age=patient.age,
            sex=patient.sex,
            is_smoking=patient.is_smoking,
            cigsperday=patient.cigsperday,
            bpmeds=patient.bpmeds,
            prevalentstroke=patient.prevalentstroke,
            prevalenthyp=patient.prevalenthyp,
            diabetes=patient.diabetes,
            totchol=patient.totchol,
            sysbp=patient.sysbp,
            diabp=patient.diabp,
            bmi=patient.bmi,
            heartrate=patient.heartrate,
            glucose=patient.glucose,
            education=patient.education
        )
        
        # Convert to DataFrame
        input_df = custom_data.get_data_as_dataframe()
        
        # Make prediction
        prediction, probabilities = prediction_pipeline.predict(input_df)
        
        # Prepare response
        result = {
            "success": True,
            "prediction": int(prediction[0]),
            "risk_level": "High Risk" if prediction[0] == 1 else "Low Risk",
            "message": "Prediction completed successfully"
        }
        
        if probabilities is not None:
            result["probability_low_risk"] = float(probabilities[0][0])
            result["probability_high_risk"] = float(probabilities[0][1])
        
        logging.info(f"Prediction result: {result['risk_level']}")
        
        return result
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(patients: List[PatientData]):
    """
    Predict 10-year CHD risk for multiple patients
    """
    try:
        logging.info(f"Received batch prediction request for {len(patients)} patients")
        
        results = []
        
        for idx, patient in enumerate(patients):
            try:
                # Create CustomData object
                custom_data = CustomData(
                    age=patient.age,
                    sex=patient.sex,
                    is_smoking=patient.is_smoking,
                    cigsperday=patient.cigsperday,
                    bpmeds=patient.bpmeds,
                    prevalentstroke=patient.prevalentstroke,
                    prevalenthyp=patient.prevalenthyp,
                    diabetes=patient.diabetes,
                    totchol=patient.totchol,
                    sysbp=patient.sysbp,
                    diabp=patient.diabp,
                    bmi=patient.bmi,
                    heartrate=patient.heartrate,
                    glucose=patient.glucose,
                    education=patient.education
                )
                
                # Convert to DataFrame
                input_df = custom_data.get_data_as_dataframe()
                
                # Make prediction
                prediction, probabilities = prediction_pipeline.predict(input_df)
                
                # Prepare result
                result = {
                    "patient_id": idx + 1,
                    "success": True,
                    "prediction": int(prediction[0]),
                    "risk_level": "High Risk" if prediction[0] == 1 else "Low Risk"
                }
                
                if probabilities is not None:
                    result["probability_low_risk"] = float(probabilities[0][0])
                    result["probability_high_risk"] = float(probabilities[0][1])
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "patient_id": idx + 1,
                    "success": False,
                    "error": str(e)
                })
        
        logging.info(f"Batch prediction completed for {len(results)} patients")
        
        return {
            "total_patients": len(patients),
            "successful_predictions": sum(1 for r in results if r.get("success", False)),
            "results": results
        }
        
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model-info")
async def model_info():
    """
    Get information about the loaded model
    """
    try:
        if prediction_pipeline.model is None:
            prediction_pipeline.load_artifacts()
        
        model_type = type(prediction_pipeline.model).__name__
        
        # Try to get feature importances if available
        feature_importance = None
        if hasattr(prediction_pipeline.model, 'feature_importances_'):
            feature_importance = "Available"
        
        return {
            "model_type": model_type,
            "model_path": prediction_pipeline.config.model_path,
            "preprocessor_path": prediction_pipeline.config.preprocessor_path,
            "feature_importance": feature_importance,
            "model_loaded": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "message": "An internal error occurred"
        }
    )


if __name__ == "__main__":
    # Note: reload=True works only when running with uvicorn command
    # Use: uvicorn app:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)