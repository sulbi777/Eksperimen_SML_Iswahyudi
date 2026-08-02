from pathlib import Path
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "namadataset_preprocessing"
def main():
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    X_train, y_train = train.drop(columns="y"), train.y
    X_test, y_test = test.drop(columns="y"), test.y
    mlflow.set_experiment("bank-marketing-baseline")
    mlflow.sklearn.autolog(log_models=True)
    with mlflow.start_run(run_name="logistic-regression-baseline"):
        model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        pred, proba = model.predict(X_test), model.predict_proba(X_test)[:, 1]
        metrics = {"accuracy": accuracy_score(y_test,pred), "precision": precision_score(y_test,pred),
                   "recall": recall_score(y_test,pred), "f1": f1_score(y_test,pred), "roc_auc": roc_auc_score(y_test,proba)}
        mlflow.log_metrics(metrics)
        print(metrics)
if __name__ == "__main__": main()
