"""
Cognifyz Data Analysis Internship - Level 2, Task 2: Cuisine Combination

Task:
  1. Identify the most common combinations of cuisines in the dataset.
  2. Determine if certain cuisine combinations tend to have higher ratings.

Approach:
  - Exact combinations: the full cuisine set of a restaurant (order ignored), e.g.
    "Chinese + North Indian". Only restaurants with 2+ cuisines count as combinations.
  - Pairs: every 2-cuisine pair inside a restaurant's list, so a restaurant serving
    A, B, C contributes the pairs A+B, A+C, B+C. This finds which cuisines are
    commonly paired even when the full lists differ.
  - Ratings use rated restaurants only (rating 0 = "Not rated").
  - Combinations with fewer than MIN_COUNT rated restaurants are ignored when ranking
    ratings, because tiny samples give misleading averages.

Run:  python level2_task2_cuisine_combination.py
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

MIN_COUNT = 30  # minimum rated restaurants for a combination to be ranked on rating

# ---------------------------------------------------------------- load data
df = pd.read_csv(next(BASE.glob("Dataset*.csv")), encoding="utf-8-sig")
df = df.dropna(subset=["Cuisines"]).copy()

# sorted tuple of unique cuisines -> order and duplicates don't matter
df["Cuisine Set"] = df["Cuisines"].apply(lambda s: tuple(sorted({c.strip() for c in s.split(",")})))
df["Num Cuisines"] = df["Cuisine Set"].apply(len)
multi = df[df["Num Cuisines"] >= 2].copy()
print(f"Restaurants with cuisine info: {len(df)} | with 2+ cuisines: {len(multi)}")

# ------------------------------------------- 1a. most common exact combinations
multi["Combination"] = multi["Cuisine Set"].apply(" + ".join)
combo_counts = multi["Combination"].value_counts()
print("\nTop 10 most common exact cuisine combinations:")
print(combo_counts.head(10))

# ------------------------------------------- 1b. most common cuisine pairs
pair_rows = []
for rid, cuisines, rating in zip(multi["Restaurant ID"], multi["Cuisine Set"], multi["Aggregate rating"]):
    for a, b in combinations(cuisines, 2):
        pair_rows.append((rid, f"{a} + {b}", rating))
pairs = pd.DataFrame(pair_rows, columns=["Restaurant ID", "Pair", "Aggregate rating"])
pair_counts = pairs["Pair"].value_counts()
print("\nTop 10 most common cuisine pairs:")
print(pair_counts.head(10))

# ------------------------------------------- 2. do some combinations rate higher?
overall_avg = df.loc[df["Aggregate rating"] > 0, "Aggregate rating"].mean()
print(f"\nOverall average rating (rated restaurants): {overall_avg:.3f}")


def rating_table(data, key):
    rated = data[data["Aggregate rating"] > 0]
    table = rated.groupby(key)["Aggregate rating"].agg(avg_rating="mean", rated_restaurants="count")
    table = table[table["rated_restaurants"] >= MIN_COUNT].copy()
    table["vs_overall"] = table["avg_rating"] - overall_avg
    return table.sort_values("avg_rating", ascending=False).round(3)


combo_ratings = rating_table(multi, "Combination")
pair_ratings = rating_table(pairs, "Pair")

print(f"\nExact combinations with >= {MIN_COUNT} rated restaurants - HIGHEST average rating:")
print(combo_ratings.head(10))
print(f"\nExact combinations with >= {MIN_COUNT} rated restaurants - LOWEST average rating:")
print(combo_ratings.tail(5))
print(f"\nCuisine pairs with >= {MIN_COUNT} rated restaurants - HIGHEST average rating:")
print(pair_ratings.head(10))

# single vs. multi cuisine overall
rated_all = df[df["Aggregate rating"] > 0]
by_num = rated_all.groupby(rated_all["Num Cuisines"].clip(upper=5))["Aggregate rating"].agg(["mean", "count"]).round(3)
by_num.index = [f"{i}" if i < 5 else "5+" for i in by_num.index]
by_num.index.name = "Cuisines served"
print("\nAverage rating by number of cuisines served:")
print(by_num)

# ------------------------------------------------------------------- charts
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

top = combo_counts.head(10)
axes[0].barh(top.index[::-1], top.values[::-1], color="#4C72B0")
axes[0].set_title("Top 10 most common cuisine combinations")
axes[0].set_xlabel("Restaurants")

best = combo_ratings.head(10)
axes[1].barh(best.index[::-1], best["avg_rating"].values[::-1], color="#55A868")
axes[1].axvline(overall_avg, color="#C44E52", linestyle="--", label=f"Overall avg {overall_avg:.2f}")
axes[1].set_xlim(2.5, best["avg_rating"].max() + 0.3)
axes[1].set_title(f"Highest-rated combinations (>= {MIN_COUNT} rated restaurants)")
axes[1].set_xlabel("Average rating")
axes[1].legend()

plt.tight_layout()
fig.savefig(OUT / "level2_task2_cuisine_combination.png", dpi=150)
combo_counts.head(25).to_csv(OUT / "level2_task2_top_combinations.csv", header=["Restaurants"])
combo_ratings.to_csv(OUT / "level2_task2_combination_ratings.csv")
print(f"\nSaved charts and tables to: {OUT}")
