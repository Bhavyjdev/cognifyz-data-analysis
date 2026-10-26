"""
Cognifyz Data Analysis Internship - Level 1, Task 3: Price Range Distribution

Task:
  1. Create a histogram or bar chart showing the distribution of price ranges.
  2. Calculate the percentage of restaurants in each price range category.

Notes:
  - 'Price range' is a category from 1 (cheapest) to 4 (most expensive), so a
    bar chart is the right choice (a histogram is meant for continuous values).

Run:  python level1_task3_price_range_distribution.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- load data
df = pd.read_csv(next(BASE.glob("Dataset*.csv")), encoding="utf-8-sig")

# ------------------------------------------------- count + percentage table
counts = df["Price range"].value_counts().sort_index()
percentages = (counts / len(df) * 100).round(2)

summary = pd.DataFrame({"Restaurants": counts, "Percentage (%)": percentages})
summary.index.name = "Price range"

print(f"Total restaurants: {len(df)}\n")
print(summary)

# ------------------------------------------------------------------- chart
labels = {1: "1\n(lowest)", 2: "2", 3: "3", 4: "4\n(highest)"}
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([labels.get(i, str(i)) for i in counts.index], counts.values,
              color=["#9ecae1", "#6baed6", "#3182bd", "#08519c"])
for bar, count, pct in zip(bars, counts.values, percentages.values):
    ax.text(bar.get_x() + bar.get_width() / 2, count + 40, f"{count}\n({pct:.1f}%)",
            ha="center", fontsize=10)
ax.set_xlabel("Price range category")
ax.set_ylabel("Number of restaurants")
ax.set_title("Distribution of restaurant price ranges")
ax.set_ylim(0, counts.max() * 1.18)
plt.tight_layout()
fig.savefig(OUT / "level1_task3_price_range_distribution.png", dpi=150)
summary.to_csv(OUT / "level1_task3_price_range_summary.csv")
print(f"\nSaved chart and table to: {OUT}")
