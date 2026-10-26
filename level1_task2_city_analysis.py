"""
Cognifyz Data Analysis Internship - Level 1, Task 2: City Analysis

Task:
  1. Identify the city with the highest number of restaurants.
  2. Calculate the average rating for restaurants in each city.
  3. Determine the city with the highest average rating.

Notes:
  - 'Aggregate rating' = 0 means "Not rated" (2148 restaurants), not a bad score,
    so those rows are excluded from every rating average.
  - A city with 1-2 restaurants can top the average list by luck. The script reports
    the raw winner AND the winner among cities with at least MIN_RESTAURANTS rated outlets.

Run:  python level1_task2_city_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

MIN_RESTAURANTS = 20  # minimum rated restaurants for a "reliable" city average

# ---------------------------------------------------------------- load data
df = pd.read_csv(next(BASE.glob("Dataset*.csv")), encoding="utf-8-sig")

# ------------------------------------------- 1. city with most restaurants
city_counts = df["City"].value_counts()
top_city = city_counts.index[0]
print(f"City with the most restaurants: {top_city} ({city_counts.iloc[0]} restaurants)")
print("\nTop 10 cities by restaurant count:")
print(city_counts.head(10))

# ------------------------------------------- 2. average rating per city
rated = df[df["Aggregate rating"] > 0]
city_stats = (
    rated.groupby("City")["Aggregate rating"]
    .agg(avg_rating="mean", rated_restaurants="count")
    .round(3)
    .sort_values("avg_rating", ascending=False)
)
print(f"\nAverage rating computed for {len(city_stats)} cities "
      f"(rated restaurants only: {len(rated)} of {len(df)})")

# ------------------------------------------- 3. city with highest average
best_raw = city_stats.iloc[0]
print(f"\nHighest average rating (any city): {city_stats.index[0]} "
      f"= {best_raw['avg_rating']:.2f} (from {int(best_raw['rated_restaurants'])} rated restaurants)")

reliable = city_stats[city_stats["rated_restaurants"] >= MIN_RESTAURANTS]
best_reliable = reliable.iloc[0]
print(f"Highest average rating (cities with >= {MIN_RESTAURANTS} rated restaurants): "
      f"{reliable.index[0]} = {best_reliable['avg_rating']:.2f} "
      f"(from {int(best_reliable['rated_restaurants'])} rated restaurants)")

print("\nTop 10 cities by average rating (>= %d rated restaurants):" % MIN_RESTAURANTS)
print(reliable.head(10))

# ------------------------------------------------------------------- charts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

top10_counts = city_counts.head(10)
axes[0].barh(top10_counts.index[::-1], top10_counts.values[::-1], color="#4C72B0")
axes[0].set_title("Top 10 cities by number of restaurants")
axes[0].set_xlabel("Restaurants")

top10_rating = reliable.head(10)
axes[1].barh(top10_rating.index[::-1], top10_rating["avg_rating"].values[::-1], color="#55A868")
axes[1].set_title(f"Top 10 cities by average rating\n(>= {MIN_RESTAURANTS} rated restaurants)")
axes[1].set_xlabel("Average rating")
axes[1].set_xlim(0, 5)

plt.tight_layout()
fig.savefig(OUT / "level1_task2_city_analysis.png", dpi=150)
city_stats.to_csv(OUT / "level1_task2_city_avg_rating.csv")
print(f"\nSaved chart and table to: {OUT}")
