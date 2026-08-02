import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
# ... import lainnya

# ✅ JANGAN GUNAKAN: dagshub.login()
# Sebagai gantinya, set MLflow tracking secara langsung

# Ambil environment variables yang sudah di-set di workflow
mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME')
mlflow_password = os.getenv('MLFLOW_TRACKING_PASSWORD')

# Set MLflow tracking uri
mlflow.set_tracking_uri(mlflow_uri)

# Untuk basic auth, set credentials di URI
if mlflow_username and mlflow_password:
    mlflow.set_tracking_uri(
        f"https://{mlflow_username}:{mlflow_password}@dagshub.com/{mlflow_username}/Eksperimen_SML_Iswahyudi.mlflow"
    )

# Set experiment
mlflow.set_experiment("model-tuning")

# Jalankan training dan logging
with mlflow.start_run():
    # Code untuk training model
    # ... hyperparameter tuning code ...
    
    # Log metrics, params, models, etc
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy_score)
    mlflow.sklearn.log_model(model, "model")
