import argparse, json, os
from pathlib import Path
import dagshub, joblib, matplotlib.pyplot as plt, mlflow, mlflow.sklearn, pandas as pd, seaborn as sns
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "namadataset_preprocessing"
ART = Path(__file__).parent / "artifacts"; ART.mkdir(exist_ok=True)
def configure_tracking():
    user, repo = os.getenv("DAGSHUB_USER"), os.getenv("DAGSHUB_REPO")
    if user and repo:
        dagshub.init(repo_owner=user, repo_name=repo, mlflow=True)
    else: mlflow.set_tracking_uri("file:" + str((BASE / "mlruns").resolve()))
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--n-iter",type=int,default=12); args=parser.parse_args()
    configure_tracking(); mlflow.set_experiment("bank-marketing-advanced")
    tr, te = pd.read_csv(DATA/"train.csv"), pd.read_csv(DATA/"test.csv")
    X,y=tr.drop(columns="y"),tr.y; Xte,yte=te.drop(columns="y"),te.y
    grid={"n_estimators":[200,350,500],"max_depth":[8,14,None],"min_samples_leaf":[1,2,5],"max_features":["sqrt",0.5]}
    search=RandomizedSearchCV(RandomForestClassifier(class_weight="balanced",random_state=42,n_jobs=-1),grid,n_iter=args.n_iter,scoring="average_precision",cv=StratifiedKFold(4,shuffle=True,random_state=42),n_jobs=-1,random_state=42,refit=True)
    with mlflow.start_run(run_name="random-forest-tuned"):
        search.fit(X,y); model=search.best_estimator_; pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
        metrics={"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred),"recall":recall_score(yte,pred),"f1_score":f1_score(yte,pred),"roc_auc":roc_auc_score(yte,prob),"average_precision":average_precision_score(yte,prob),"best_cv_average_precision":search.best_score_}
        mlflow.log_params({**search.best_params_,"n_iter":args.n_iter,"cv_folds":4,"random_state":42}); mlflow.log_metrics(metrics)
        report=classification_report(yte,pred,output_dict=True); (ART/"metrics.json").write_text(json.dumps({"metrics":metrics,"report":report},indent=2))
        plt.figure(figsize=(5,4)); sns.heatmap(confusion_matrix(yte,pred),annot=True,fmt="d",cmap="Blues"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(ART/"confusion_matrix.png",dpi=160); plt.close()
        importance=pd.DataFrame({"feature":X.columns,"importance":model.feature_importances_}).sort_values("importance",ascending=False); importance.to_csv(ART/"feature_importance.csv",index=False)
        joblib.dump(model,ART/"model.joblib")
        mlflow.log_artifacts(str(ART)); mlflow.sklearn.log_model(model,"model",signature=infer_signature(Xte,model.predict(Xte)),input_example=Xte.head(3))
        print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
