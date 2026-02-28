import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("data/raw/weekly_demand.csv")
df['date'] = pd.to_datetime(df['date'])

# Choose one product
product_id = df['product_id'].unique()[0]
product_df = df[df['product_id'] == product_id]

# Aggregate across warehouses
product_df = product_df.groupby('date')['units_sold'].sum().reset_index()

product_df = product_df.sort_values('date')

print(f"Forecasting for Product: {product_id}")
print("Total Records:", len(product_df))

# Train/Test Split (last 12 weeks for test)
train = product_df[:-12]
test = product_df[-12:]

# Fit SARIMA model
model = SARIMAX(
    train['units_sold'],
    order=(1,1,1),
    seasonal_order=(1,1,1,52)
)

model_fit = model.fit(disp=False)
forecast = model_fit.forecast(steps=12)
# Forecast
forecast = model_fit.forecast(steps=12)

# Evaluate
mae = mean_absolute_error(test['units_sold'], forecast)
print("MAE:", mae)

# Plot
plt.figure()
plt.plot(train['date'], train['units_sold'], label="Train")
plt.plot(test['date'], test['units_sold'], label="Test")
plt.plot(test['date'], forecast, label="Forecast")
plt.legend()
plt.title(f"ARIMA Forecast for {product_id}")
plt.tight_layout()
plt.show()

print("Average Test Demand:", test['units_sold'].mean())

mape = (abs(test['units_sold'] - forecast) / test['units_sold']).mean() * 100
print("MAPE (%):", round(mape, 2))


import numpy as np

# -------- INVENTORY OPTIMIZATION --------

lead_time = 2  # weeks
service_level_z = 1.65  # 95% service level

# Use forecast mean demand
avg_forecast_demand = forecast.mean()

# Use residual standard deviation as demand variability
residuals = test['units_sold'] - forecast
demand_std = np.std(residuals)

# Safety Stock
safety_stock = service_level_z * demand_std * np.sqrt(lead_time)

# Reorder Point
reorder_point = (avg_forecast_demand * lead_time) + safety_stock

print("\n========== INVENTORY METRICS ==========")
print("Avg Forecast Demand:", round(avg_forecast_demand, 2))
print("Demand Std Dev:", round(demand_std, 2))
print("Safety Stock:", round(safety_stock, 2))
print("Reorder Point:", round(reorder_point, 2))