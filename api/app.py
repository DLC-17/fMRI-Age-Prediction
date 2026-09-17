from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
import os

app = FastAPI(title="fMRI Age Prediction API", version="1.0.0")

model = None

class PredictionRequest(BaseModel):
    features: list[float]

@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("MODEL_PATH")
    if model_path and os.path.exists(model_path):
        try:
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print("No model file found. API is running in demo mode.")

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Deploy a trained model file and set MODEL_PATH.",
        )

    try:
        import xgboost as xgb
        X = xgb.DMatrix([request.features])
        pred = model.predict(X)
        return {"predicted_age": float(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
