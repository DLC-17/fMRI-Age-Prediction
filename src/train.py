import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def train_xgboost(X_train, y_train, random_state=42):
    """
    Trains an XGBoost Classifier for ADHD Outcome prediction.
    """
    with mlflow.start_run(run_name="XGBoost_Train", nested=True):
        model = xgb.XGBClassifier(random_state=random_state, eval_metric="logloss")
        model.fit(X_train, y_train)
        
        mlflow.log_param("model_type", "XGBoostClassifier")
        mlflow.xgboost.log_model(model, "model")
        
        return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the classification model and returns Accuracy and ROC AUC.
    """
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else predictions
    
    acc = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probs)
    
    print(classification_report(y_test, predictions))
    
    if mlflow.active_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        
    return acc, auc
