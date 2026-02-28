import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# -----------------------------
# CONFIGURATION
# -----------------------------
np.random.seed(42)
random.seed(42)

WAREHOUSES = {
    "WH1": "Mumbai",
    "WH2": "Pune",
    "WH3": "Delhi",
    "WH4": "Bengaluru"
}

START_DATE = "2023-01-01"
WEEKS = 104

CATEGORY_CONFIG = {
    "Antibiotic": {"count": 30, "shelf_life": 365, "lead_time": (2, 3), "cost_range": (40, 120)},
    "Painkiller": {"count": 30, "shelf_life": 720, "lead_time": (1, 2), "cost_range": (5, 20)},
    "Vaccine": {"count": 20, "shelf_life": (180, 365), "lead_time": (3, 4), "cost_range": (200, 800)},
    "Chronic": {"count": 30, "shelf_life": 720, "lead_time": (2, 2), "cost_range": (30, 150)},
    "Seasonal": {"count": 20, "shelf_life": 365, "lead_time": (2, 3), "cost_range": (25, 100)},
    "Surgical": {"count": 30, "shelf_life": 720, "lead_time": (1, 2), "cost_range": (100, 400)}
}

DEMAND_TYPES = ["Stable", "Seasonal", "Trending", "Intermittent"]

# -----------------------------
# PRODUCT MASTER GENERATION
# -----------------------------
products = []
product_id_counter = 1

for category, config in CATEGORY_CONFIG.items():
    for _ in range(config["count"]):

        shelf_life = (
            random.randint(*config["shelf_life"])
            if isinstance(config["shelf_life"], tuple)
            else config["shelf_life"]
        )

        lead_time = random.randint(*config["lead_time"])
        unit_cost = round(random.uniform(*config["cost_range"]), 2)
        margin = random.uniform(0.15, 0.40)
        selling_price = round(unit_cost * (1 + margin), 2)

        critical_flag = 1 if category in ["Vaccine", "Chronic"] and random.random() > 0.3 else 0

        products.append([
            f"PHM{product_id_counter:04d}",
            f"{category}_Drug_{product_id_counter}",
            category,
            random.choice(DEMAND_TYPES),
            shelf_life,
            lead_time,
            unit_cost,
            selling_price,
            critical_flag
        ])

        product_id_counter += 1

products_df = pd.DataFrame(products, columns=[
    "product_id",
    "product_name",
    "category",
    "demand_type",
    "shelf_life_days",
    "lead_time_weeks",
    "unit_cost",
    "selling_price",
    "critical_flag"
])

# -----------------------------
# WAREHOUSE MASTER
# -----------------------------
warehouse_df = pd.DataFrame([
    [wid, city, random.randint(40000, 60000)]
    for wid, city in WAREHOUSES.items()
], columns=["warehouse_id", "city", "storage_capacity"])

# -----------------------------
# DEMAND GENERATION
# -----------------------------
start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
dates = [start_date + timedelta(weeks=i) for i in range(WEEKS)]

demand_records = []

for _, product in products_df.iterrows():

    base_demand = random.randint(20, 200)
    trend_factor = random.uniform(0.01, 0.05)

    for warehouse_id in warehouse_df["warehouse_id"]:

        for i, date in enumerate(dates):

            if product["demand_type"] == "Stable":
                demand = base_demand + np.random.normal(0, 5)

            elif product["demand_type"] == "Seasonal":
                seasonal_effect = 30 * np.sin(2 * np.pi * i / 52)
                demand = base_demand + seasonal_effect + np.random.normal(0, 5)

            elif product["demand_type"] == "Trending":
                demand = base_demand + (trend_factor * i * base_demand) + np.random.normal(0, 5)

            else:  # Intermittent
                demand = base_demand if random.random() > 0.6 else 0

            demand = max(0, int(demand))

            demand_records.append([
                date,
                product["product_id"],
                warehouse_id,
                demand
            ])

demand_df = pd.DataFrame(demand_records, columns=[
    "date",
    "product_id",
    "warehouse_id",
    "units_sold"
])

# -----------------------------
# SAVE FILES
# -----------------------------
products_df.to_csv("data/raw/products.csv", index=False)
warehouse_df.to_csv("data/raw/warehouses.csv", index=False)
demand_df.to_csv("data/raw/weekly_demand.csv", index=False)

print("Data generation complete.")