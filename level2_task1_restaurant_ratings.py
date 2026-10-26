"""
Cognifyz Data Analysis Internship - Level 2, Task 1: Restaurant Ratings

Task:
  1. Analyze the distribution of aggregate ratings and determine the most common rating range.
  2. Calculate the average number of votes received by restaurants.

Notes:
  - 'Aggregate rating' = 0 means "Not rated" (2148 restaurants). The distribution is
    shown for rated restaurants only; the unrated count is reported separately.
  - Ratings are grouped into 0.5-wide ranges: [1.5-2.0), [2.0-2.5), ... [4.5-5.0].
  - Average votes is reported for all restaurants and for rated restaurants only.

Run:  python level2_task1_restaurant_ratings.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- load data
df = pd.read_csv(next(BASE.glob("Dataset*.csv")), encoding="utf-8-sig")

unrated = (df["Aggregate rating"] == 0).sum()
rated = df[df["Aggregate rating"] > 0].copy()
print(f"Total restaurants: {len(df)} | rated: {len(rated)} | not rated (0): {unrated}")

# ------------------------------------------------ 1. rating distribution
print("\nRating summary (rated restaurants):")
print(rated["Aggregate rating"].describe().round(3))

bins = np.arange(0.0, 5.5, 0.5)  # 0.0, 0.5, ... 5.0
labels = [f"{b:.1f}-{b + 0.5:.1f}" for b in bins[:-1]]
# right=False -> ranges are [low, high); include 5.0 in the top range
rated["Rating range"] = pd.cut(rated["Aggregate rating"], bins=bins, labels=labels,
                               right=False, include_lowest=True)
rated.loc[rated["Aggregate rating"] == 5.0, "Rating range"] = labels[-1]

range_counts = rated["Rating range"].value_counts().sort_index()
range_counts = range_counts[range_counts > 0]
range_pct = (range_counts / len(rated) * 100).round(2)

dist = pd.DataFrame({"Restaurants": range_counts, "Percentage (%)": range_pct})
print("\nDistribution by rating range:")
print(dist)

most_common = range_counts.idxmax()
print(f"\n-> Most common rating range: {most_common} "
      f"({range_counts.max()} restaurants, {range_pct.max()}% of rated restaurants)")

# ------------------------------------------------ 2. average votes
avg_votes_all = df["Votes"].mean()
avg_votes_rated = rated["Votes"].mean()
print(f"\nAverage votes per restaurant (all {len(df)}): {avg_votes_all:.2f}")
print(f"Average votes per restaurant (rated only):  {avg_votes_rated:.2f}")
print(f"Median votes (all): {df['Votes'].median():.0f}  "
      "(far below the mean -> a few very popular restaurants pull the average up)")

# ------------------------------------------------------------------- charts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(rated["Aggregate rating"], bins=np.arange(1.0, 5.25, 0.25), color="#4C72B0",
             edgecolor="white")
axes[0].set_title("Histogram of aggregate ratings (rated restaurants)")
axes[0].set_xlabel("Aggregate rating")
axes[0].set_ylabel("Restaurants")

bars = axes[1].bar(range_counts.index.astype(str), range_counts.values, color="#55A868")
bars[list(range_counts.index).index(most_common)].set_color("#C44E52")
axes[1].set_title(f"Restaurants per rating range (most common: {most_common})")
axes[1].set_xlabel("Rating range")
axes[1].set_ylabel("Restaurants")
plt.setp(axes[1].get_xticklabels(), rotation=30)

plt.tight_layout()
fig.savefig(OUT / "level2_task1_restaurant_ratings.png", dpi=150)
dist.to_csv(OUT / "level2_task1_rating_distribution.csv")
print(f"\nSaved chart and table to: {OUT}")
