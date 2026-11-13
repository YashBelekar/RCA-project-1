# Consumer Shopping Trends — Purchase Amount Prediction

A reproducible machine-learning project inspired by the workflow of [ecommerce-customer-spending-prediction](https://github.com/HardikRavriya/ecommerce-customer-spending-prediction), but using a **different public dataset** and a different prediction target.

## Project objective
Predict a customer's average purchase amount from shopping, demographic and behavioural attributes. The project demonstrates:
- dataset acquisition and validation
- data quality checks and exploratory data analysis
- categorical/numerical preprocessing with scikit-learn pipelines
- Simple Linear Regression (single strongest numeric feature)
- Multiple Linear Regression (all eligible predictors)
- optional Ridge and Random Forest benchmark models
- MAE, MSE, RMSE and R² evaluation
- actual-vs-predicted and residual visualizations
- an interactive Streamlit prediction dashboard

## Dataset
**Consumer Shopping Trends 2026**, published in the public GitHub repository by SahilYadav-op.

- Dataset page: https://github.com/SahilYadav-op/Consumer-shopping-trends-analysis
- Direct CSV: https://raw.githubusercontent.com/SahilYadav-op/Consumer-shopping-trends-analysis/main/Consumer_Shopping_Trends_2026%20%286%29.csv
- The source repository describes approximately 11,790 records and 25 variables, including age, income, digital behaviour, shopping activity, loyalty and spending.
- Target: `avg_online_spend` (or a compatible average-spend field detected by the loader). If your downloaded CSV uses a different column label, edit `TARGET_COLUMN` in `src/config.py` after checking the printed column list.

**Attribution:** Dataset remains the work of its original publisher. This repository contains original analysis code and project documentation; it does not claim ownership of the dataset.

## Folder structure
```text
Consumer_Shopping_Trends_Purchase_Amount_Prediction/
├── data/
│   └── README.md
├── notebooks/
│   └── Consumer_Shopping_Trends_Purchase_Amount_Prediction.ipynb
├── src/
│   ├── config.py
│   ├── data_utils.py
│   └── train.py
├── outputs/
│   └── figures/
├── dashboard.py
├── download_dataset.py
├── requirements.txt
├── run_project.bat
├── run_dashboard.bat
├── .gitignore
├── LICENSE
└── README.md
```

## Run on Windows (PowerShell)
1. Extract this ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal → PowerShell.
4. Run these commands one by one:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_dataset.py
python src/train.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

To open the notebook:
```powershell
jupyter notebook
```
Open `notebooks/Consumer_Shopping_Trends_Purchase_Amount_Prediction.ipynb` and run cells from top to bottom.

To launch the dashboard:
```powershell
streamlit run dashboard.py
```

## Methodology
1. Load the source CSV locally (the downloader saves it to `data/`).
2. Inspect dimensions, data types, missing values, duplicates and summary statistics.
3. Detect the spending target and numeric/categorical predictors.
4. Remove identifiers and leakage-prone columns.
5. Split training/testing data before fitting preprocessing.
6. Impute missing values, one-hot encode categorical columns and scale numeric columns inside a Pipeline.
7. Train Simple Linear Regression and Multiple Linear Regression; compare Ridge and Random Forest as additional benchmarks.
8. Evaluate using MAE, MSE, RMSE and R² on held-out test data.
9. Save the fitted model and metrics under `outputs/`.

## Important notes
- No model scores are hard-coded. They are computed from the downloaded data and current train/test split.
- The source dataset may evolve. The loader prints all column names and gives a clear error if the expected spending target cannot be identified.
- This is a learning project, not a production credit/pricing decision system.
- Do not commit personal/customer identifiers if you create a modified dataset. The supplied analysis drops likely identifier fields before modelling.

## License
Original code in this project is provided under the MIT License. Dataset licensing and attribution remain governed by the original dataset repository; verify its terms before redistribution.
