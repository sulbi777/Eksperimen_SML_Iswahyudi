import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Ambil environment variables
mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME')
mlflow_password = os.getenv('MLFLOW_TRACKING_PASSWORD')

# Set MLflow tracking uri (jika perlu, gunakan kredensial basic auth)
if mlflow_username and mlflow_password:
    mlflow.set_tracking_uri(
        f"https://{mlflow_username}:{mlflow_password}@dagshub.com/{mlflow_username}/Eksperimen_SML_Iswahyudi.mlflow"
    )
elif mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)

mlflow.set_experiment("model-tuning")

# Contoh alur training sederhana (sesuaikan dengan data & preprocessing Anda)
# Pastikan file data/variabel berikut benar adanya di repo
data = pd.read_csv("namadataset_preprocessing/train.csv")  # contoh path
X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# contoh model / grid search (sesuaikan hyperparameter Anda)
model = RandomForestClassifier(random_state=42)
param_grid = {"n_estimators": [50, 100]}

gsearch = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
gsearch.fit(X_train, y_train)

best_model = gsearch.best_estimator_

with mlflow.start_run():
    # log params
    mlflow.log_param("best_n_estimators", gsearch.best_params_.get("n_estimators"))
    # prediksi & hitung metrik
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    # log model
    mlflow.sklearn.log_model(best_model, "model")
