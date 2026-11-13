import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.config import DATA_PATH, OUTPUT_DIR
from src.data_utils import load_data, resolve_target, prepare_features

st.set_page_config(page_title="Shopping Spend Predictor", page_icon="🛍️", layout="wide")
st.title("🛍️ Consumer Shopping Trends — Purchase Amount Prediction")
st.caption("Explore the public consumer shopping dataset and estimate spending with a trained regression model.")

@st.cache_data
def get_data():
    raw = load_data()
    target = resolve_target(raw)
    X, y, clean = prepare_features(raw, target)
    return raw, target, X, y, clean

if not DATA_PATH.exists():
    st.warning("Dataset not found. Run `python download_dataset.py` from the project folder, then refresh this page.")
    st.stop()

try:
    raw, target, X, y, clean = get_data()
except Exception as exc:
    st.error(str(exc)); st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Records used", f"{len(clean):,}")
c2.metric("Predictor columns", f"{X.shape[1]}")
c3.metric("Prediction target", target.replace("_", " ").title())
st.subheader("Dataset preview")
st.dataframe(clean.head(20), use_container_width=True)
st.subheader("Target distribution")
st.bar_chart(clean[target].dropna().value_counts(bins=20, sort=False))

model_path = OUTPUT_DIR / "multiple_linear_regression.joblib"
if not model_path.exists():
    st.info("Train the model first: `python src/train.py`")
    st.stop()

model = joblib.load(model_path)
st.subheader("Predict spending for a customer")
st.write("Enter values for the predictors. Categorical fields use their most common observed category.")
row = {}
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        row[col] = st.number_input(col.replace("_", " ").title(),
                                   value=float(X[col].median()) if X[col].notna().any() else 0.0)
    else:
        options = X[col].dropna().astype(str).value_counts().head(30).index.tolist()
        row[col] = st.selectbox(col.replace("_", " ").title(), options or ["Unknown"])
if st.button("Predict purchase amount", type="primary"):
    try:
        prediction = float(model.predict(pd.DataFrame([row]))[0])
        st.success(f"Estimated {target.replace('_', ' ')}: {prediction:,.2f}")
        st.caption("This is a model estimate, not a guaranteed customer outcome.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
