"""
Cognifyz Data Analysis Internship - Level 1, Task 4: Online Delivery

Task:
  1. Determine the percentage of restaurants that offer online delivery.
  2. Compare the average ratings of restaurants with and without online delivery.

Notes:
  - 'Aggregate rating' = 0 means "Not rated", so unrated restaurants are
    left out of the rating comparison (the count of each group is printed).
  - A Welch t-test is added to check whether the rating gap is statistically real.

Run:  python level1_task4_online_delivery.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- load data
df = pd.read_csv(next(BASE.glob("Dataset*.csv")), encoding="utf-8-sig")

# -------------------------------------------- 1. % offering online delivery
delivery_counts = df["Has Online delivery"].value_counts()
delivery_pct = (delivery_counts / len(df) * 100).round(2)
print("Online delivery availability (all restaurants):")
print(pd.DataFrame({"Restaurants": delivery_counts, "Percentage (%)": delivery_pct}))
print(f"\n-> {delivery_pct['Yes']}% of restaurants offer online delivery")

# ---------------------------------------- 2. average rating with vs without
rated = df[df["Aggregate rating"] > 0]
rating_stats = (
    rated.groupby("Has Online delivery")["Aggregate rating"]
    .agg(avg_rating="mean", median_rating="median", rated_restaurants="count")
    .round(3)
)
print(f"\nRating comparison (rated restaurants only: {len(rated)} of {len(df)}):")
print(rating_stats)

with_d = rated.loc[rated["Has Online delivery"] == "Yes", "Aggregate rating"]
without_d = rated.loc[rated["Has Online delivery"] == "No", "Aggregate rating"]
diff = with_d.mean() - without_d.mean()
t_stat, p_value = stats.ttest_ind(with_d, without_d, equal_var=False)
print(f"\nDifference in average rating (with - without): {diff:+.3f}")
print(f"Welch t-test: t = {t_stat:.2f}, p = {p_value:.3g} "
      f"-> {'statistically significant' if p_value < 0.05 else 'not significant'} at 5%")

# ------------------------------------------------------------------- charts
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].pie(delivery_counts[["Yes", "No"]], labels=["Online delivery", "No online delivery"],
            autopct="%1.1f%%", colors=["#55A868", "#C44E52"], startangle=90)
axes[0].set_title("Restaurants offering online delivery")

bars = axes[1].bar(["With online delivery", "Without online delivery"],
                   [rating_stats.loc["Yes", "avg_rating"], rating_stats.loc["No", "avg_rating"]],
                   color=["#55A868", "#C44E52"])
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{bar.get_height():.2f}", ha="center")
axes[1].set_ylim(0, 5)
axes[1].set_ylabel("Average rating (rated restaurants)")
axes[1].set_title("Average rating by delivery availability")

plt.tight_layout()
fig.savefig(OUT / "level1_task4_online_delivery.png", dpi=150)
rating_stats.to_csv(OUT / "level1_task4_rating_comparison.csv")
print(f"\nSaved chart and table to: {OUT}")
