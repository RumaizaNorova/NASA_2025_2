"""
FastAPI Backend for Shark Habitat Prediction Dashboard
Provides API endpoints for model predictions and data access
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
from dotenv import load_dotenv
from openai_service import openai_service

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Shark Habitat Prediction API",
    description="API for predicting shark foraging behavior using satellite data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
scaler = None
feature_columns = None
shark_data = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    datetime: str
    species: Optional[str] = None
    sst: Optional[float] = None
    chlorophyll_a: Optional[float] = None
    primary_productivity: Optional[float] = None
    ssh_anomaly: Optional[float] = None

class PredictionResponse(BaseModel):
    foraging_probability: float
    confidence: float
    prediction: int
    features_used: List[str]
    model_info: Dict[str, Any]

class SharkTrack(BaseModel):
    id: str
    name: str
    species: str
    latitude: float
    longitude: float
    datetime: str
    foraging_behavior: int
    foraging_probability: Optional[float] = None

class ModelPerformance(BaseModel):
    model_name: str
    auc_score: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

class InsightRequest(BaseModel):
    prediction_data: Dict[str, Any]
    shark_data: Dict[str, Any]

class QuestionRequest(BaseModel):
    question: str
    context_data: Optional[Dict[str, Any]] = None

class ReportRequest(BaseModel):
    analysis_data: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Load model and data on startup"""
    global model, scaler, feature_columns, shark_data
    
    try:
        # Load the best model (GradientBoosting)
        model_path = "./results_full/models/gradientboosting_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            # Try alternative path
            alt_model_path = "/app/results_full/models/gradientboosting_model.pkl"
            if os.path.exists(alt_model_path):
                model = joblib.load(alt_model_path)
                logger.info(f"Loaded model from {alt_model_path}")
            else:
                logger.error(f"Model file not found: {model_path} or {alt_model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path} or {alt_model_path}")
        
        # Load feature columns (we'll need to reconstruct this from the training data)
        # For now, we'll use the feature list from the integrated data
        data_path = "./integrated_data_full.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            # Get feature columns (exclude target and metadata columns)
            exclude_cols = ['active', 'datetime', 'id', 'name', 'gender', 'species', 'weight', 
                           'length', 'tagDate', 'dist_total', 'foraging_behavior']
            feature_columns = [col for col in df.columns if col not in exclude_cols]
            logger.info(f"Loaded {len(feature_columns)} feature columns")
            
            # Load a sample of shark data for visualization
            shark_data = df.sample(min(1000, len(df))).to_dict('records')
            logger.info(f"Loaded {len(shark_data)} shark track records")
        else:
            # Try alternative path
            alt_data_path = "/app/integrated_data_full.csv"
            if os.path.exists(alt_data_path):
                df = pd.read_csv(alt_data_path)
                exclude_cols = ['active', 'datetime', 'id', 'name', 'gender', 'species', 'weight', 
                               'length', 'tagDate', 'dist_total', 'foraging_behavior']
                feature_columns = [col for col in df.columns if col not in exclude_cols]
                logger.info(f"Loaded {len(feature_columns)} feature columns")
                
                shark_data = df.sample(min(1000, len(df))).to_dict('records')
                logger.info(f"Loaded {len(shark_data)} shark track records")
            else:
                logger.error(f"Data file not found: {data_path} or {alt_data_path}")
                raise FileNotFoundError(f"Data file not found: {data_path} or {alt_data_path}")
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

def prepare_features(request: PredictionRequest) -> np.ndarray:
    """Prepare features for prediction from request data"""
    
    # Parse datetime
    dt = pd.to_datetime(request.datetime)
    
    # Calculate temporal features
    hour = dt.hour
    month = dt.month
    day_of_year = dt.timetuple().tm_yday
    year = dt.year
    day_of_week = dt.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Calculate cyclical features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365)
    
    # Use provided environmental data or defaults
    sst = request.sst if request.sst is not None else 20.0
    chlorophyll_a = request.chlorophyll_a if request.chlorophyll_a is not None else 0.5
    primary_productivity = request.primary_productivity if request.primary_productivity is not None else 0.5
    ssh_anomaly = request.ssh_anomaly if request.ssh_anomaly is not None else 0.0
    
    # Calculate distance to coast (simplified - using distance from 0,0)
    distance_to_coast = np.sqrt(request.latitude**2 + request.longitude**2) * 111000  # rough conversion to meters
    
    # Calculate anomalies (simplified - using mean values)
    sst_anomaly = sst - 20.0  # simplified anomaly
    chl_anomaly = chlorophyll_a - 0.5
    pp_anomaly = primary_productivity - 0.5
    
    # Create feature vector in the same order as training
    features = np.array([
        request.latitude,
        request.longitude,
        hour,
        month,
        day_of_year,
        year,
        day_of_week,
        is_weekend,
        hour_sin,
        hour_cos,
        month_sin,
        month_cos,
        day_of_year_sin,
        day_of_year_cos,
        sst,
        distance_to_coast,
        chlorophyll_a,
        primary_productivity,
        ssh_anomaly,
        sst_anomaly,
        chl_anomaly,
        pp_anomaly,
        0.0,  # weight_length_ratio (not available)
        0.0,  # lat_diff (not available for single point)
        0.0,  # lon_diff (not available for single point)
        0.0,  # distance_moved (not available for single point)
        0.0,  # movement_speed (not available for single point)
        0.0   # time_diff_hours (not available for single point)
    ])
    
    return features.reshape(1, -1)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Shark Habitat Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/predict",
            "/shark-tracks",
            "/model-performance",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": shark_data is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_habitat(request: PredictionRequest):
    """Predict shark foraging probability for given location and conditions"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare features
        features = prepare_features(request)
        
        # Make prediction
        probability = model.predict_proba(features)[0][1]  # Probability of foraging
        prediction = model.predict(features)[0]
        
        # Calculate confidence based on probability
        confidence = abs(probability - 0.5) * 2  # Distance from 0.5, scaled to 0-1
        
        return PredictionResponse(
            foraging_probability=float(probability),
            confidence=float(confidence),
            prediction=int(prediction),
            features_used=feature_columns,
            model_info={
                "model_type": "GradientBoosting",
                "auc_score": 0.972,
                "training_samples": 64942
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/shark-tracks", response_model=List[SharkTrack])
async def get_shark_tracks(
    species: Optional[str] = Query(None, description="Filter by species"),
    limit: int = Query(1000, description="Maximum number of tracks to return"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get shark tracking data for visualization"""
    
    if shark_data is None:
        raise HTTPException(status_code=500, detail="Shark data not loaded")
    
    try:
        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(shark_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Apply filters
        if species:
            df = df[df['species'].str.contains(species, case=False, na=False)]
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['datetime'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['datetime'] <= end_dt]
        
        # Limit results
        df = df.head(limit)
        
        # Convert to response format
        tracks = []
        for _, row in df.iterrows():
            tracks.append(SharkTrack(
                id=str(row['id']),
                name=row['name'],
                species=row['species'],
                latitude=float(row['latitude']),
                longitude=float(row['longitude']),
                datetime=row['datetime'].isoformat(),
                foraging_behavior=int(row['foraging_behavior']),
                foraging_probability=None  # Could be calculated if needed
            ))
        
        return tracks
        
    except Exception as e:
        logger.error(f"Error getting shark tracks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get shark tracks: {str(e)}")

@app.get("/model-performance", response_model=List[ModelPerformance])
async def get_model_performance():
    """Get model performance metrics"""
    
    try:
        # Load performance data
        perf_path = "./results_full/model_performance_full.json"
        if os.path.exists(perf_path):
            import json
            with open(perf_path, 'r') as f:
                perf_data = json.load(f)
        else:
            # Try alternative path
            alt_perf_path = "/app/results_full/model_performance_full.json"
            if os.path.exists(alt_perf_path):
                import json
                with open(alt_perf_path, 'r') as f:
                    perf_data = json.load(f)
            else:
                raise HTTPException(status_code=404, detail="Performance data not found")
        
        performance = []
        for model_name, metrics in perf_data.items():
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                performance.append(ModelPerformance(
                    model_name=model_name,
                    auc_score=metrics['auc_score'],
                    accuracy=report['accuracy'],
                    precision=report['weighted avg']['precision'],
                    recall=report['weighted avg']['recall'],
                    f1_score=report['weighted avg']['f1-score']
                ))
        
        return performance
            
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance data: {str(e)}")

@app.get("/species")
async def get_species():
    """Get list of shark species in the dataset"""
    
    if shark_data is None:
        raise HTTPException(status_code=500, detail="Shark data not loaded")
    
    try:
        df = pd.DataFrame(shark_data)
        species_list = df['species'].unique().tolist()
        return {"species": species_list}
        
    except Exception as e:
        logger.error(f"Error getting species: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get species: {str(e)}")

@app.get("/stats")
async def get_dataset_stats():
    """Get dataset statistics"""
    
    if shark_data is None:
        raise HTTPException(status_code=500, detail="Shark data not loaded")
    
    try:
        df = pd.DataFrame(shark_data)
        
        stats = {
            "total_records": len(df),
            "unique_sharks": df['name'].nunique(),
            "species_count": df['species'].nunique(),
            "date_range": {
                "start": df['datetime'].min(),
                "end": df['datetime'].max()
            },
            "foraging_distribution": {
                "foraging": int(df['foraging_behavior'].sum()),
                "not_foraging": int((df['foraging_behavior'] == 0).sum())
            },
            "species_distribution": df['species'].value_counts().to_dict()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/generate-insights")
async def generate_insights(request: InsightRequest):
    """Generate AI-powered insights about predictions"""
    try:
        insights = await openai_service.generate_insights(
            request.prediction_data, 
            request.shark_data
        )
        return insights
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    """Answer natural language questions about shark data"""
    try:
        # Get context data if not provided
        context_data = request.context_data
        if context_data is None:
            context_data = await get_dataset_stats()
        
        answer = await openai_service.answer_question(
            request.question, 
            context_data
        )
        return answer
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")

@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    """Generate comprehensive analysis report"""
    try:
        report = await openai_service.generate_report(request.analysis_data)
        return report
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
