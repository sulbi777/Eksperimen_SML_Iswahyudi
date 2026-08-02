"""Repeatable preprocessing for UCI Bank Marketing. Run from repository root."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "namadataset_raw" / "bank_marketing_raw.csv"
OUT = BASE / "namadataset_preprocessing"
TARGET = "y"
RANDOM_STATE = 42

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    df = df.replace("unknown", np.nan)
    # -1 means never previously contacted; retaining it as its own valid value.
    if "pdays" in df:
        df["pdays_never_contacted"] = (df["pdays"] == -1).astype(int)
    return df

def main():
    if not RAW.exists():
        raise FileNotFoundError(f"{RAW} not found. Run download_raw_data.py first.")
    raw = pd.read_csv(RAW)
    df = clean(raw).drop(columns=["duration"])
    y = (df.pop(TARGET).astype(str).str.lower() == "yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    num = X_train.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X_train.columns if c not in num]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ], verbose_feature_names_out=False)
    train_array = preprocessor.fit_transform(X_train)
    test_array = preprocessor.transform(X_test)
    names = preprocessor.get_feature_names_out().tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_array, columns=names).assign(y=y_train.to_numpy()).to_csv(OUT / "train.csv", index=False)
    pd.DataFrame(test_array, columns=names).assign(y=y_test.to_numpy()).to_csv(OUT / "test.csv", index=False)
    joblib.dump(preprocessor, OUT / "preprocessor.joblib")
    (OUT / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    report = {"raw_rows": len(raw), "rows_after_dedup": len(df), "train_rows": len(X_train),
              "test_rows": len(X_test), "n_numeric": len(num), "n_categorical": len(cat),
              "encoded_features": len(names), "target_positive_rate": round(float(y.mean()), 5),
              "random_state": RANDOM_STATE}
    (OUT / "processing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
