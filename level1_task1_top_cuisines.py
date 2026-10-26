"""
Cognifyz Data Analysis Internship - Level 1, Task 1: Top Cuisines

Task:
  1. Determine the top three most common cuisines in the dataset.
  2. Calculate the percentage of restaurants that serve each of the top cuisines.

Notes:
  - The 'Cuisines' column holds comma-separated lists ("French, Japanese, Desserts"),
    so each list is split and exploded to one row per (restaurant, cuisine).
  - 9 restaurants have no cuisine listed; they are excluded from the percentage base.
  - A restaurant that serves several cuisines counts once for each, so the
    percentages add up to more than 100%.

Run:  python level1_task1_top_cuisines.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- load data
df = pd.read_csv(next(BASE.glob("Dataset*.csv")), encoding="utf-8-sig")

# ----------------------------------------------------------- clean cuisines
df = df.dropna(subset=["Cuisines"]).copy()
total_restaurants = len(df)

df["Cuisine List"] = df["Cuisines"].str.split(",").apply(lambda xs: [x.strip() for x in xs])
cuisine_rows = df[["Restaurant ID", "Cuisine List"]].explode("Cuisine List")
cuisine_rows = cuisine_rows.rename(columns={"Cuisine List": "Cuisine"})

# --------------------------------------------------------------- top cuisines
# nunique on Restaurant ID guards against a cuisine repeated inside one row
counts = cuisine_rows.groupby("Cuisine")["Restaurant ID"].nunique().sort_values(ascending=False)
top3 = counts.head(3)
top3_pct = (top3 / total_restaurants * 100).round(2)

result = pd.DataFrame({"Restaurants": top3, "Percentage (%)": top3_pct})

print(f"Restaurants with cuisine info: {total_restaurants}")
print("\nTop 3 most common cuisines:")
print(result)

# ------------------------------------------------------------------- chart
top10 = counts.head(10)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(top10.index[::-1], top10.values[::-1], color="#4C72B0")
for bar, value in zip(bars, top10.values[::-1]):
    ax.text(value + 30, bar.get_y() + bar.get_height() / 2,
            f"{value}  ({value / total_restaurants * 100:.1f}%)", va="center", fontsize=9)
ax.set_xlabel("Number of restaurants")
ax.set_title("Top 10 cuisines by number of restaurants")
ax.set_xlim(0, top10.max() * 1.2)
plt.tight_layout()
fig.savefig(OUT / "level1_task1_top_cuisines.png", dpi=150)
result.to_csv(OUT / "level1_task1_top3_cuisines.csv")
print(f"\nSaved chart and table to: {OUT}")
