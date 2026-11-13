import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import OUTPUT_DIR, FIGURE_DIR, RANDOM_STATE, TEST_SIZE
from src.data_utils import load_data, resolve_target, prepare_features

def build_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])

def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    target = resolve_target(df)
    X, y, clean = prepare_features(df, target)
    if X.shape[1] == 0:
        raise ValueError("No usable predictor columns remain after cleaning.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        raise ValueError("Simple Linear Regression requires at least one numeric predictor.")
    # Pick the strongest absolute training-set correlation for the simple model.
    corr = X_train[numeric_cols].corrwith(y_train).abs().dropna().sort_values(ascending=False)
    simple_feature = corr.index[0]
    models = {
        "Simple Linear Regression": (LinearRegression(), [simple_feature]),
        "Multiple Linear Regression": (LinearRegression(), X.columns.tolist()),
        "Ridge Regression": (Ridge(alpha=1.0), X.columns.tolist()),
        "Random Forest Regressor": (RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ), X.columns.tolist())
    }
    rows, fitted = [], {}
    for name, (estimator, features) in models.items():
        preprocessor = build_preprocessor(X[features])
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train[features], y_train)
        pred = pipe.predict(X_test[features])
        rows.append({"Model": name, **metrics(y_test, pred)})
        fitted[name] = pipe
        if name == "Multiple Linear Regression":
            best_pred = pred
    results = pd.DataFrame(rows).sort_values("RMSE")
    print("\nTarget:", target)
    print("Rows used:", len(clean), "| Predictors:", X.shape[1])
    print("Simple regression feature:", simple_feature)
    print("\nModel comparison:\n", results.to_string(index=False))
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    joblib.dump(fitted["Multiple Linear Regression"], OUTPUT_DIR / "multiple_linear_regression.joblib")
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps({
        "target": target, "rows_used": int(len(clean)), "predictor_count": int(X.shape[1]),
        "simple_feature": simple_feature, "metrics": rows
    }, indent=2), encoding="utf-8")

    # Evaluation plots for the multiple linear regression baseline.
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=best_pred, alpha=.65)
    lo, hi = float(min(y_test.min(), best_pred.min())), float(max(y_test.max(), best_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual spending"); plt.ylabel("Predicted spending")
    plt.title("Actual vs Predicted — Multiple Linear Regression")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "actual_vs_predicted.png", dpi=160); plt.close()

    residuals = y_test.to_numpy() - best_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residual (actual − predicted)"); plt.title("Residual Distribution")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "residual_distribution.png", dpi=160); plt.close()

    plt.figure(figsize=(9, 6))
    sns.heatmap(clean.select_dtypes(include=np.number).corr(), cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=160); plt.close()
    print(f"\nSaved model, metrics and figures in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
