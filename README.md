# StockSense AI — final prototype

## Features
- Streamlit dashboard
- Inventory CSV upload and validation
- Revenue, estimated gross profit and inventory valuation
- Stock status and preliminary reorder threshold suggestions
- Random Forest demand forecast from dated sales history
- Chronological holdout evaluation using MAE and RMSE
- Download analyzed inventory

## Windows setup
Open this folder in VS Code. In the terminal, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, use CMD and run:
```bat
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Data files
- `inventory.csv`: example current stock and recent 30-day sales
- `sales_history.csv`: 90 days of **fictional demo sales** for testing the ML workflow

Replace demo data with real records before making business claims. Synthetic data does not demonstrate real-world forecasting accuracy.

## Inventory CSV columns
`product,stock,reorder_level,selling_price,cost_price,units_sold_30_days`

## Sales history CSV columns
`date,product,units_sold`

Use one row per product per date. Missing dates are treated as zero by the demo model, so confirm that assumption is valid for your business.

## ML notes
The model uses Random Forest regression with lagged sales and calendar features. Evaluation is chronological and reports MAE/RMSE on a holdout period. Compare these metrics against a naive baseline before claiming the ML model is better. A short history, synthetic data, promotions, holidays, stockouts, or missing records can make forecasts unreliable.
