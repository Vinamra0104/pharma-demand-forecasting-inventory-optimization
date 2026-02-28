import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/raw/weekly_demand.csv")

print("========== DATA OVERVIEW ==========")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nMissing Values:\n", df.isnull().sum())

# Convert date column
df['date'] = pd.to_datetime(df['date'])

print("\n========== DEMAND STATISTICS ==========")
print(df['units_sold'].describe())

# ---------- 1️⃣ Total Demand Over Time ----------
weekly_total = df.groupby('date')['units_sold'].sum()

plt.figure()
weekly_total.plot()
plt.title("Total Weekly Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Total Units Sold")
plt.tight_layout()
plt.show()

# ---------- 2️⃣ Demand by Warehouse ----------
plt.figure()
sns.boxplot(data=df, x='warehouse_id', y='units_sold')
plt.title("Demand Distribution by Warehouse")
plt.tight_layout()
plt.show()

# ---------- 3️⃣ Top 10 Products ----------
top_products = (
    df.groupby('product_id')['units_sold']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure()
top_products.plot(kind='bar')
plt.title("Top 10 Products by Total Units Sold")
plt.tight_layout()
plt.show()