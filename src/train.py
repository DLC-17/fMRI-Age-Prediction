import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def train_xgboost(X_train, y_train, random_state=42):
    """
    Trains an XGBoost Regressor for Age prediction.
    """
    with mlflow.start_run(run_name="XGBoost_Age_Train", nested=True):
        model = xgb.XGBRegressor(random_state=random_state, objective="reg:squarederror")
        model.fit(X_train, y_train)
        
        mlflow.log_param("model_type", "XGBoostRegressor")
        mlflow.xgboost.log_model(model, "model")
        
        return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the regression model and returns RMSE and R2.
    """
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    if mlflow.active_run():
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
    return rmse, r2
