import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Trains a Random Forest Regressor.
    """
    with mlflow.start_run(run_name="RandomForest_Train", nested=True):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.sklearn.log_model(model, "model")
        return model

def train_xgboost(X_train, y_train, random_state=42):
    """
    Trains an XGBoost Regressor.
    """
    with mlflow.start_run(run_name="XGBoost_Train", nested=True):
        model = xgb.XGBRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.log_param("model_type", "XGBoost")
        mlflow.xgboost.log_model(model, "model")
        return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns R^2 and MSE.
    """
    from sklearn.metrics import mean_squared_error, r2_score
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    if mlflow.active_run():
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
    return mse, r2
