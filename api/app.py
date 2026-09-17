from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
import mlflow

app = FastAPI(title="fMRI Age Prediction API", version="1.0.0")

model = None
MODEL_URI = "models:/xgboost_fmri/Production"

class PredictionRequest(BaseModel):
    features: list[float]

@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"Loaded model from {MODEL_URI}")
    except Exception as e:
        print(f"Failed to load MLflow model on startup: {e}")
        # Note: in a real deployment, we might fail to start up if model is missing.
        # We catch it here so the API can still launch for testing without an active MLflow tracking server.

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        # Assuming model expects a 2D array / DataFrame
        X = pd.DataFrame([request.features])
        pred = model.predict(X)
        return {"predicted_age": float(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
