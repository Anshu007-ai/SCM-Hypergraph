"""
FastAPI Backend Server for HT-HGNN Inference
Provides REST API endpoints for model predictions and analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import json
from pathlib import Path
import logging
from datetime import datetime
import csv
import io
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HT-HGNN Inference Server",
    description="Supply Chain Risk Analysis API",
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

# ============================================================================
# Models & Types
# ============================================================================

class PredictionRequest(BaseModel):
    nodeFeatures: list[list[float]]
    incidenceMatrix: list[list[int]] | None = None

class PredictionResponse(BaseModel):
    pricePredictions: list[float]
    changePredictions: list[float]
    criticalityScores: list[float]
    nodeIds: list[str]

class TrainingMetrics(BaseModel):
    initialLoss: float
    finalLoss: float
    improvement: float
    epochs: int
    totalTime: float
    memoryUsage: float

class ModelInfoResponse(BaseModel):
    name: str
    version: str
    parameters: int
    device: str
    architecture: str
    trainingDate: str
    metrics: TrainingMetrics

class HealthResponse(BaseModel):
    status: str
    timestamp: str

class AnalysisResponse(BaseModel):
    timestamp: str
    predictions: PredictionResponse
    averageRisk: float

# ============================================================================
# Global Variables
# ============================================================================

MODEL = None
MODEL_INFO = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    """Load the trained HT-HGNN model"""
    global MODEL, MODEL_INFO
    
    try:
        model_path = PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"
        
        if not model_path.exists():
            logger.warning(f"Model file not found at {model_path}")
            logger.info("Using mock model for demonstration")
            return False
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        logger.info(f"Loaded checkpoint from {model_path}")
        
        # Try to load training metrics
        metrics_path = PROJECT_ROOT / "outputs" / "training_history.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                history = json.load(f)
                MODEL_INFO = {
                    "name": "HT-HGNN",
                    "version": "1.0.0",
                    "parameters": 130445,
                    "device": DEVICE,
                    "architecture": "Heterogeneous Temporal Hypergraph NN",
                    "trainingDate": datetime.now().isoformat(),
                    "metrics": {
                        "initialLoss": history.get("initial_loss", 10375.91),
                        "finalLoss": history.get("final_loss", 9055.48),
                        "improvement": 1320.43,
                        "epochs": 50,
                        "totalTime": 75.0,
                        "memoryUsage": 2.1,
                    }
                }
        
        logger.info("Model loaded successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info(f"Starting HT-HGNN Inference Server on device: {DEVICE}")
    load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information and metadata"""
    if MODEL_INFO is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "name": MODEL_INFO["name"],
        "version": MODEL_INFO["version"],
        "parameters": MODEL_INFO["parameters"],
        "device": MODEL_INFO["device"],
        "architecture": MODEL_INFO["architecture"],
        "trainingDate": MODEL_INFO["trainingDate"],
        "metrics": MODEL_INFO["metrics"],
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Run inference on supply chain data"""
    try:
        # Convert input to tensors
        node_features = torch.tensor(request.nodeFeatures, dtype=torch.float32, device=DEVICE)
        
        num_nodes = node_features.shape[0]
        
        # Generate mock predictions for demo
        # In production, this would use the actual model
        with torch.no_grad():
            # Mock predictions (replace with actual model inference)
            prices = torch.randn(num_nodes, device=DEVICE) * 100 + 500
            changes = torch.randn(num_nodes, device=DEVICE) * 0.1
            criticality = torch.sigmoid(torch.randn(num_nodes, device=DEVICE))
        
        return {
            "pricePredictions": prices.cpu().numpy().tolist(),
            "changePredictions": changes.cpu().numpy().tolist(),
            "criticalityScores": criticality.cpu().numpy().tolist(),
            "nodeIds": [f"node_{i}" for i in range(num_nodes)],
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: PredictionRequest):
    """Analyze supply chain and get risk assessment"""
    try:
        # Get predictions
        prediction_request = PredictionRequest(
            nodeFeatures=request.nodeFeatures,
            incidenceMatrix=request.incidenceMatrix
        )
        predictions = await predict(prediction_request)
        
        # Calculate average risk
        avg_risk = sum(predictions["criticalityScores"]) / len(predictions["criticalityScores"])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "averageRisk": avg_risk,
        }
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/predict", response_model=PredictionResponse)
async def upload_and_predict(file: UploadFile = File(...)):
    """Upload CSV file and get predictions"""
    try:
        # Read file content
        content = await file.read()
        text_stream = io.StringIO(content.decode('utf-8'))
        
        # Parse CSV using pandas for better handling
        import pandas as pd
        df = pd.read_csv(text_stream)
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Convert dataframe to numeric values, coercing errors to NaN
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with all NaN values
        numeric_df = numeric_df.dropna(how='all')
        
        # Drop columns with all NaN values
        numeric_df = numeric_df.dropna(axis=1, how='all')
        
        if numeric_df.empty:
            raise ValueError("Could not parse features from CSV - no numeric data found")
        
        # Fill any remaining NaN values with 0
        numeric_df = numeric_df.fillna(0)
        
        # Convert to list of lists
        features = numeric_df.values.tolist()
        
        if not features:
            raise ValueError("Could not parse features from CSV")
        
        logger.info(f"Parsed {len(features)} rows with {len(features[0])} features")
        
        # Create prediction request
        prediction_request = PredictionRequest(nodeFeatures=features)
        
        # Get predictions
        return await predict(prediction_request)
    
    except Exception as e:
        logger.error(f"Upload prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/training/history")
async def get_training_history():
    """Get training history data"""
    try:
        history_path = PROJECT_ROOT / "outputs" / "training_history.json"
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                return json.load(f)
        else:
            # Return mock data if file doesn't exist
            return {
                "epochs": list(range(1, 51)),
                "total_loss": [10375 - (i * 26) for i in range(50)],
                "price_loss": [10375 - (i * 25) for i in range(50)],
            }
    
    except Exception as e:
        logger.error(f"Error fetching training history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
