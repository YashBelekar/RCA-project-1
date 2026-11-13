from pathlib import Path
import re
import pandas as pd
from src.config import DATA_PATH, TARGET_COLUMN

TARGET_CANDIDATES = [
    "avg_online_spend", "average_online_spend", "avg_store_spend",
    "average_store_spend", "purchase_amount", "purchase_amount_usd",
    "average_purchase_amount", "avg_purchase_amount", "spending_amount"
]

def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

def load_data(path=DATA_PATH):
    if not Path(path).exists():
        raise FileNotFoundError("Dataset not found. Run: python download_dataset.py")
    df = pd.read_csv(path)
    df.columns = [normalize_name(c) for c in df.columns]
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("Columns:", ", ".join(df.columns))
    return df

def resolve_target(df):
    if TARGET_COLUMN:
        key = normalize_name(TARGET_COLUMN)
        if key in df.columns:
            return key
        raise KeyError(f"TARGET_COLUMN={key!r} is not present. Available: {list(df.columns)}")
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    # Conservative fallback: only select an unambiguous column with spend/amount in its name.
    matches = [c for c in df.columns if ("spend" in c or "purchase_amount" in c) and pd.api.types.is_numeric_dtype(df[c])]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        "Could not identify a unique numeric spending target. Inspect the column list above, "
        "then set TARGET_COLUMN in src/config.py to the exact source column name. "
        f"Candidate matches: {matches}"
    )

def prepare_features(df, target):
    work = df.copy()
    work = work.drop_duplicates()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])
    # Exclude identifiers and near-unique text fields that do not generalize.
    drop_terms = ("id", "email", "name", "phone", "address", "customer_code")
    excluded = [c for c in work.columns if c != target and any(term in c for term in drop_terms)]
    X = work.drop(columns=[target] + excluded)
    # Drop columns with > 60% missing values and constant columns.
    X = X.loc[:, X.isna().mean() <= 0.60]
    X = X.loc[:, X.nunique(dropna=True) > 1]
    # Exclude dates in this baseline; richer date engineering can be added separately.
    date_like = [c for c in X.columns if "date" in c or "timestamp" in c]
    X = X.drop(columns=date_like, errors="ignore")
    # Convert numeric-looking object columns when most values parse successfully.
    for col in X.select_dtypes(include="object").columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().mean() >= 0.90:
            X[col] = converted
    return X, work[target], work
