import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Pharma Demand Forecasting", layout="wide")

st.title("📦 Pharma Demand Forecasting & Inventory Optimization")

# ================= LOAD DATA =================
df = pd.read_csv("data/raw/weekly_demand.csv")
df['date'] = pd.to_datetime(df['date'])

sku_list = df['product_id'].unique()

# ==========================================================
# 🔎 SINGLE SKU ANALYSIS
# ==========================================================

st.header("🔎 Single SKU Analysis")

selected_sku = st.selectbox("Select SKU", sku_list)

product_df = df[df['product_id'] == selected_sku]
product_df = product_df.groupby('date')['units_sold'].sum().reset_index()
product_df = product_df.sort_values('date')

train = product_df[:-12]
test = product_df[-12:]

model = SARIMAX(
    train['units_sold'],
    order=(1,1,1),
    seasonal_order=(1,1,1,52)
)
model_fit = model.fit(disp=False)
forecast = model_fit.forecast(steps=12)

mae = np.mean(np.abs(test['units_sold'] - forecast))
mape = (np.abs(test['units_sold'] - forecast) / test['units_sold']).mean() * 100

# Inventory logic
lead_time = 2
z = 1.65
simulation_runs = 1000

avg_forecast = forecast.mean()
residuals = test['units_sold'] - forecast
std_dev = np.std(residuals)

safety_stock = z * std_dev * np.sqrt(lead_time)
reorder_point = (avg_forecast * lead_time) + safety_stock

current_stock = np.random.randint(200, 600)
unit_cost = np.random.randint(50, 200)

# Monte Carlo Risk
stockout_count = 0
for _ in range(simulation_runs):
    simulated_demand = np.random.normal(avg_forecast, std_dev, lead_time)
    if simulated_demand.sum() > current_stock:
        stockout_count += 1

stockout_risk = (stockout_count / simulation_runs) * 100
health_score = (current_stock - reorder_point) / reorder_point
capital_locked = current_stock * unit_cost

# KPI Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", round(mae, 2))
col2.metric("MAPE (%)", round(mape, 2))
col3.metric("Reorder Point", round(reorder_point, 2))
col4.metric("Stockout Risk (%)", round(stockout_risk, 2))

st.markdown("---")

# Forecast Chart
st.subheader("📈 Demand Forecast")

fig, ax = plt.subplots()
ax.plot(train['date'], train['units_sold'], label="Train")
ax.plot(test['date'], test['units_sold'], label="Test")
ax.plot(test['date'], forecast, label="Forecast")
ax.legend()
st.pyplot(fig)

st.markdown("---")

st.subheader("📦 Inventory Metrics")
st.write(f"Average Forecast Demand: {round(avg_forecast,2)}")
st.write(f"Safety Stock: {round(safety_stock,2)}")
st.write(f"Current Stock: {current_stock}")
st.write(f"Inventory Health Score: {round(health_score,2)}")
st.write(f"Capital Locked: ₹ {round(capital_locked,2)}")

# ==========================================================
# 📊 MULTI-SKU COMPARISON SECTION
# ==========================================================

st.markdown("----")
st.header("📊 Multi-SKU Comparison")

selected_skus = st.multiselect(
    "Select SKUs to Compare",
    sku_list,
    default=sku_list[:min(3, len(sku_list))]
)

if len(selected_skus) > 0:

    summary_data = []

    for sku in selected_skus:

        product_df = df[df['product_id'] == sku]
        product_df = product_df.groupby('date')['units_sold'].sum().reset_index()
        product_df = product_df.sort_values('date')

        if len(product_df) < 20:
            continue

        train = product_df[:-12]
        test = product_df[-12:]

        try:
            model = SARIMAX(
                train['units_sold'],
                order=(1,1,1),
                seasonal_order=(1,1,1,52)
            )
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=12)

            avg_forecast = forecast.mean()
            residuals = test['units_sold'] - forecast
            demand_std = np.std(residuals)

            safety_stock = z * demand_std * np.sqrt(lead_time)
            reorder_point = (avg_forecast * lead_time) + safety_stock

            current_stock = np.random.randint(200, 600)
            unit_cost = np.random.randint(50, 200)

            # Monte Carlo Risk
            stockout_count = 0
            for _ in range(simulation_runs):
                simulated_demand = np.random.normal(avg_forecast, demand_std, lead_time)
                if simulated_demand.sum() > current_stock:
                    stockout_count += 1

            stockout_risk = (stockout_count / simulation_runs) * 100
            health_score = (current_stock - reorder_point) / reorder_point
            capital_locked = current_stock * unit_cost

            summary_data.append([
                sku,
                round(reorder_point,2),
                current_stock,
                round(stockout_risk,2),
                round(health_score,2),
                round(capital_locked,2)
            ])

        except:
            continue

    if len(summary_data) > 0:

        summary_df = pd.DataFrame(summary_data, columns=[
            "SKU",
            "Reorder Point",
            "Current Stock",
            "Stockout Risk (%)",
            "Health Score",
            "Capital Locked"
        ])

        st.dataframe(summary_df, use_container_width=True)

        # Risk Chart
        st.subheader("⚠️ Stockout Risk Comparison")

        fig2, ax2 = plt.subplots()
        ax2.bar(summary_df["SKU"], summary_df["Stockout Risk (%)"])
        ax2.set_ylabel("Stockout Risk (%)")
        st.pyplot(fig2)

    else:
        st.warning("Selected SKUs do not have enough data.")

else:
    st.info("Select at least one SKU to compare.")