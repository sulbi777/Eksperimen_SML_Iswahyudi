from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

out = Path(__file__).parent / "namadataset_raw" / "bank_marketing_raw.csv"
out.parent.mkdir(parents=True, exist_ok=True)
data = fetch_ucirepo(id=222)
df = pd.concat([data.data.features, data.data.targets], axis=1)
df.to_csv(out, index=False)
print(f"Saved {df.shape[0]:,} rows and {df.shape[1]} columns to {out}")
