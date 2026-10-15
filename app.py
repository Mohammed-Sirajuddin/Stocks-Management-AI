from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# StockSense AI — inventory analytics + ML demand forecasting
# ============================================================

st.set_page_config(
    page_title="StockSense AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
INVENTORY_FILE = BASE_DIR / "inventory.csv"
SALES_FILE = BASE_DIR / "sales_history.csv"

INVENTORY_COLUMNS = [
    "product", "stock", "reorder_level", "selling_price",
    "cost_price", "units_sold_30_days",
]
SALES_COLUMNS = ["date", "product", "units_sold"]
NUMERIC_INVENTORY = [
    "stock", "reorder_level", "selling_price",
    "cost_price", "units_sold_30_days",
]

SAMPLE_INVENTORY = pd.DataFrame({
    "product": ["Rice", "Cooking Oil", "Biscuits", "Soap", "Shampoo", "Toothpaste"],
    "stock": [12, 5, 40, 8, 18, 7],
    "reorder_level": [15, 10, 20, 12, 10, 12],
    "selling_price": [60, 150, 20, 35, 120, 80],
    "cost_price": [50, 130, 14, 25, 95, 60],
    "units_sold_30_days": [80, 65, 10, 45, 25, 35],
})

# --------------------------- Styling ---------------------------
st.markdown("""
<style>
.stApp { background: #F4F7FB; }
h1, h2, h3 { color: #17324D; }
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #DFE7F0;
    border-radius: 12px; padding: 14px;
}
[data-testid="stSidebar"] { background: #10243A; }
[data-testid="stSidebar"] * { color: #FFFFFF; }
div[data-testid="stAlert"] { border-radius: 10px; }
.small-note { color: #5C6B7A; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# --------------------------- Data helpers ---------------------------
def load_inventory(uploaded_file):
    """Read and validate an inventory CSV."""
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
        source = f"Uploaded: {uploaded_file.name}"
    elif INVENTORY_FILE.exists():
        raw = pd.read_csv(INVENTORY_FILE)
        source = "inventory.csv"
    else:
        raw = SAMPLE_INVENTORY.copy()
        source = "Built-in sample data (fictional)"

    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    missing = [c for c in INVENTORY_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("Inventory CSV is missing columns: " + ", ".join(missing))

    df = raw[INVENTORY_COLUMNS].copy()
    df["product"] = df["product"].astype("string").str.strip()

    for col in NUMERIC_INVENTORY:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid = (
        df["product"].isna()
        | df["product"].eq("").fillna(True)
        | df[NUMERIC_INVENTORY].isna().any(axis=1)
        | df[NUMERIC_INVENTORY].lt(0).any(axis=1)
    )
    for col in ["stock", "reorder_level", "units_sold_30_days"]:
        invalid |= df[col].notna() & (df[col] % 1 != 0)

    removed = int(invalid.sum())
    df = df.loc[~invalid].copy()
    if df.empty:
        raise ValueError("No valid inventory rows remain. Check the CSV values.")
    if df["product"].duplicated().any():
        raise ValueError("Each product must appear only once in inventory.csv.")
    return df.reset_index(drop=True), source, removed


def calculate_inventory(df):
    """Add business metrics and transparent rule-based recommendations."""
    out = df.copy()
    out["estimated_revenue_30d"] = out["selling_price"] * out["units_sold_30_days"]
    out["estimated_gross_profit_30d"] = (
        (out["selling_price"] - out["cost_price"]) * out["units_sold_30_days"]
    )
    out["inventory_value"] = out["stock"] * out["cost_price"]
    out["daily_sales_average"] = out["units_sold_30_days"] / 30.0
    out["estimated_days_of_stock"] = (
        out["stock"] / out["daily_sales_average"].replace(0, np.nan)
    )
    out["stock_status"] = np.where(
        out["stock"] <= out["reorder_level"], "Reorder", "Healthy"
    )
    out["suggested_reorder"] = (
        out["reorder_level"] - out["stock"]
    ).clip(lower=0).astype(int)
    return out


def load_sales(uploaded_file):
    """Read and validate dated sales history."""
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
        source = f"Uploaded: {uploaded_file.name}"
    elif SALES_FILE.exists():
        raw = pd.read_csv(SALES_FILE)
        source = "sales_history.csv"
    else:
        return None, "No sales history file found", 0

    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    missing = [c for c in SALES_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("Sales CSV is missing columns: " + ", ".join(missing))

    df = raw[SALES_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product"] = df["product"].astype("string").str.strip()
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")

    invalid = (
        df["date"].isna()
        | df["product"].isna()
        | df["product"].eq("").fillna(True)
        | df["units_sold"].isna()
        | (df["units_sold"] < 0)
    )
    removed = int(invalid.sum())
    df = df.loc[~invalid].copy()
    if df.empty:
        raise ValueError("No valid sales records remain.")
    return df.sort_values("date").reset_index(drop=True), source, removed


def make_features(daily):
    """Create leakage-safe lag and calendar features for daily sales."""
    frame = pd.DataFrame({"sales": daily.astype(float)})
    frame["lag_1"] = frame["sales"].shift(1)
    frame["lag_2"] = frame["sales"].shift(2)
    frame["lag_7"] = frame["sales"].shift(7)
    frame["mean_7"] = frame["sales"].shift(1).rolling(7).mean()
    frame["mean_14"] = frame["sales"].shift(1).rolling(14).mean()
    frame["day_of_week"] = frame.index.dayofweek
    frame["day_of_month"] = frame.index.day
    return frame


def forecast_product(sales_df, product, horizon=7):
    """
    Train a RandomForestRegressor for one product and recursively forecast.
    Uses a chronological holdout for a simple evaluation where possible.
    """
    p = sales_df[sales_df["product"].astype(str) == str(product)].copy()
    if p.empty:
        raise ValueError(f"No sales history found for {product}.")

    daily = p.groupby("date")["units_sold"].sum().sort_index()
    daily.index = pd.to_datetime(daily.index).normalize()
    daily = daily.groupby(level=0).sum()

    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    # Missing dates are treated as zero sales; confirm this assumption is valid
    # for the business before relying on forecasts.
    daily = daily.reindex(calendar, fill_value=0.0)

    if len(daily) < 35:
        raise ValueError(
            f"{product} has only {len(daily)} calendar days. "
            "At least 35 days are required for this demo model; 60–90+ days is preferable."
        )

    features = [
        "lag_1", "lag_2", "lag_7", "mean_7", "mean_14",
        "day_of_week", "day_of_month",
    ]
    frame = make_features(daily).dropna()
    if len(frame) < 20:
        raise ValueError("Not enough usable training examples after creating lag features.")

    # Time-ordered holdout evaluation; never randomly shuffle time-series rows.
    split = max(int(len(frame) * 0.8), len(frame) - 14)
    split = min(split, len(frame) - 1)
    train = frame.iloc[:split]
    test = frame.iloc[split:]

    model_eval = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    model_eval.fit(train[features], train["sales"])
    test_pred = np.maximum(model_eval.predict(test[features]), 0)

    mae = float(mean_absolute_error(test["sales"], test_pred))
    rmse = float(np.sqrt(mean_squared_error(test["sales"], test_pred)))

    # Refit using all available history before future prediction.
    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    model.fit(frame[features], frame["sales"])

    history_values = daily.astype(float).tolist()
    future_dates = pd.date_range(
        daily.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    predictions = []

    for date in future_dates:
        values = history_values
        row = pd.DataFrame([{
            "lag_1": values[-1],
            "lag_2": values[-2],
            "lag_7": values[-7],
            "mean_7": float(np.mean(values[-7:])),
            "mean_14": float(np.mean(values[-14:])),
            "day_of_week": date.dayofweek,
            "day_of_month": date.day,
        }])
        pred = max(0.0, float(model.predict(row[features])[0]))
        predictions.append(pred)
        history_values.append(pred)

    forecast = pd.DataFrame({
        "date": future_dates,
        "predicted_units": np.round(predictions, 1),
    })
    return forecast, {"mae": mae, "rmse": rmse, "training_days": len(daily)}


# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.title("📦 StockSense AI")
    st.caption("Smart Retail Intelligence")
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Inventory", "AI Advisor", "About Project"],
    )
    st.divider()
    st.caption("IBM SkillsBuild project prototype")
    st.caption("Primary SDG: 12")
    st.caption("Responsible Consumption")


# --------------------------- Main header ---------------------------
st.title("📦 StockSense AI")
st.write("Inventory analytics and demand forecasting for small businesses.")

with st.expander("Upload data / download templates", expanded=False):
    st.write("Inventory data describes current stock and recent 30-day sales.")
    inventory_upload = st.file_uploader(
        "Upload inventory CSV", type=["csv"], key="inventory_upload"
    )
    st.download_button(
        "Download inventory template",
        SAMPLE_INVENTORY.to_csv(index=False),
        file_name="stocksense_inventory_template.csv",
        mime="text/csv",
    )
    st.write("Sales history requires one row per product per date.")
    sales_upload = st.file_uploader(
        "Upload dated sales history CSV", type=["csv"], key="sales_upload"
    )
    st.download_button(
        "Download sales-history template",
        pd.DataFrame(columns=SALES_COLUMNS).to_csv(index=False),
        file_name="stocksense_sales_history_template.csv",
        mime="text/csv",
    )

try:
    inventory_raw, inventory_source, removed_inventory = load_inventory(inventory_upload)
    inventory = calculate_inventory(inventory_raw)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Inventory data issue: {exc}")
    st.stop()

if removed_inventory:
    st.warning(f"Excluded {removed_inventory} invalid inventory row(s).")

try:
    sales_history, sales_source, removed_sales = load_sales(sales_upload)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Sales-history data issue: {exc}")
    sales_history = None
    sales_source = "Invalid sales history"
    removed_sales = 0

if removed_sales:
    st.warning(f"Excluded {removed_sales} invalid sales record(s).")

st.caption(f"Inventory source: {inventory_source} · Sales source: {sales_source}")


# --------------------------- Dashboard ---------------------------
if page == "Dashboard":
    st.subheader("Business Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", f"{len(inventory):,}")
    c2.metric("Estimated Revenue (30d)", f"₹{inventory['estimated_revenue_30d'].sum():,.0f}")
    c3.metric("Estimated Gross Profit (30d)", f"₹{inventory['estimated_gross_profit_30d'].sum():,.0f}")
    c4.metric("Inventory Value at Cost", f"₹{inventory['inventory_value'].sum():,.0f}")

    low_stock = inventory[inventory["stock_status"] == "Reorder"]
    st.info(f"{len(low_stock)} product(s) are at or below their reorder level.")

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            inventory, x="product", y="stock", color="stock_status",
            title="Current Stock by Product",
            labels={"product": "Product", "stock": "Units in stock", "stock_status": "Status"},
            color_discrete_map={"Reorder": "#E76F51", "Healthy": "#2A9D8F"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(
            inventory, x="product", y="units_sold_30_days",
            title="Units Sold in the Last 30 Days",
            labels={"product": "Product", "units_sold_30_days": "Units sold"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Products Requiring Attention")
    if low_stock.empty:
        st.success("No products are at or below their reorder level.")
    else:
        st.dataframe(
            low_stock[["product", "stock", "reorder_level", "suggested_reorder"]],
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "Figures are estimates based on uploaded data. Gross profit here excludes "
        "operating expenses, taxes, returns and other costs."
    )


# --------------------------- Inventory ---------------------------
elif page == "Inventory":
    st.subheader("Inventory Explorer")
    query = st.text_input("Search products", placeholder="Type a product name")
    status = st.selectbox("Stock status", ["All", "Reorder", "Healthy"])
    filtered = inventory[inventory["product"].str.contains(query, case=False, regex=False, na=False)]
    if status != "All":
        filtered = filtered[filtered["stock_status"] == status]

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Download analyzed inventory",
        filtered.to_csv(index=False),
        file_name="stocksense_analyzed_inventory.csv",
        mime="text/csv",
    )


# --------------------------- AI Advisor ---------------------------
elif page == "AI Advisor":
    st.subheader("AI Demand Forecasting & Reorder Advisor")
    st.write(
        "The forecasting model learns patterns from dated sales history. "
        "Stock alerts and reorder thresholds are transparent business rules."
    )

    st.markdown("### Inventory recommendations")
    for _, row in inventory.iterrows():
        if row["stock_status"] == "Reorder":
            st.warning(
                f"**{row['product']}** — stock is {int(row['stock'])}, "
                f"at/below the threshold of {int(row['reorder_level'])}. "
                f"Preliminary quantity to reach threshold: {int(row['suggested_reorder'])}."
            )
        elif row["units_sold_30_days"] < 15:
            st.info(
                f"**{row['product']}** — only {int(row['units_sold_30_days'])} units "
                "sold in 30 days. Review demand before reordering."
            )

    st.divider()
    st.markdown("### Machine-learning demand forecast")

    if sales_history is None:
        st.info(
            "Upload a sales-history CSV above, or place sales_history.csv beside app.py. "
            "Use real dated sales data. The downloadable template shows the required columns."
        )
    else:
        available_products = sorted(sales_history["product"].astype(str).unique())
        chosen_product = st.selectbox(
            "Choose a product", available_products, key="forecast_product"
        )
        horizon = st.slider("Forecast horizon (days)", 1, 14, 7)

        if st.button("Train model and forecast", type="primary"):
            with st.spinner("Training and evaluating the model..."):
                try:
                    forecast, metrics = forecast_product(
                        sales_history, chosen_product, horizon
                    )
                    st.success("Forecast generated. Review the evaluation metrics below.")
                    a, b, c = st.columns(3)
                    a.metric("Forecast Total", f"{forecast['predicted_units'].sum():.1f} units")
                    b.metric("Holdout MAE", f"{metrics['mae']:.2f} units/day")
                    c.metric("Holdout RMSE", f"{metrics['rmse']:.2f} units/day")

                    st.dataframe(forecast, use_container_width=True, hide_index=True)
                    fig = px.line(
                        forecast, x="date", y="predicted_units", markers=True,
                        title=f"{chosen_product}: predicted daily demand",
                        labels={"date": "Date", "predicted_units": "Predicted units"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(
                        f"Model trained using {metrics['training_days']} calendar days. "
                        "MAE and RMSE are from a chronological holdout, not a guarantee "
                        "of future accuracy. Compare against a simple baseline before "
                        "claiming the model performs well."
                    )

                    matching = inventory[
                        inventory["product"].astype(str) == str(chosen_product)
                    ]
                    if not matching.empty:
                        current_stock = float(matching.iloc[0]["stock"])
                        forecast_total = float(forecast["predicted_units"].sum())
                        st.markdown("### Forecast-informed stock check")
                        st.write(f"Current stock: **{current_stock:.0f} units**")
                        st.write(f"Predicted demand over {horizon} days: **{forecast_total:.1f} units**")
                        if current_stock < forecast_total:
                            st.warning(
                                "Current stock is below predicted demand for this horizon. "
                                "Review supplier lead time, safety stock, incoming orders and "
                                "purchasing constraints before deciding what to order."
                            )
                        else:
                            st.success(
                                "Current stock covers the point forecast for this horizon. "
                                "This does not account for safety stock or lead time."
                            )
                except (ValueError, KeyError, TypeError) as exc:
                    st.error(str(exc))


# --------------------------- About ---------------------------
elif page == "About Project":
    st.subheader("About StockSense AI")
    st.markdown("""
**Problem:** Small retailers may struggle with stockouts, excess inventory and purchasing decisions.

**Solution:** A prototype that analyzes stock, sales and prices; gives transparent stock alerts;
and uses a machine-learning model to forecast demand when enough dated sales history is available.

**Primary SDG:** SDG 12 — Responsible Consumption and Production.

**AI/ML method:** Random Forest regression with lagged sales and calendar features.

**Evaluation:** A chronological holdout using MAE and RMSE. Forecast quality must be compared
with a simple baseline and tested on suitable real data.

**Limitations:** Sample data is fictional. Missing sales dates are treated as zero sales, which
is only valid if a missing record really means no sales. The prototype does not yet model
holidays, promotions, supplier lead times, incoming purchase orders or multiple stores.
""")

st.divider()
st.caption(
    "StockSense AI · Student prototype · Verify recommendations before purchasing. "
    "Forecasts are estimates, not guarantees."
)
