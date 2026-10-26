"""
Cognifyz Data Analysis Internship - Level 3, Task 2: Votes Analysis

Task:
  1. Identify the restaurants with the highest and lowest number of votes.
  2. Analyze if there is a correlation between the number of votes and the rating.

Notes:
  - Many restaurants have 0 votes (usually the "Not rated" ones), so the "lowest" is a
    tie. The script reports how many tie at the minimum and lists a sample.
  - The correlation is computed on rated restaurants only (rating 0 = "Not rated" would
    distort it). Spearman is used alongside Pearson because votes are very skewed
    (median ~31, max ~10,900).

Run:  python level3_task2_votes_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

COLS = ["Restaurant Name", "City", "Votes", "Aggregate rating"]

# ---------------------------------------------------------------- load data
df = pd.read_csv(next(BASE.glob("Dataset*.csv")), encoding="utf-8-sig")

# ------------------------------------------- 1. highest / lowest votes
max_votes, min_votes = df["Votes"].max(), df["Votes"].min()
print("Restaurant(s) with the HIGHEST number of votes:")
print(df.loc[df["Votes"] == max_votes, COLS].to_string(index=False))

print("\nTop 10 restaurants by votes:")
print(df.nlargest(10, "Votes")[COLS].to_string(index=False))

n_min = (df["Votes"] == min_votes).sum()
print(f"\nLowest number of votes: {min_votes} -> shared by {n_min} restaurants "
      f"({n_min / len(df) * 100:.1f}% of the dataset). First 5:")
print(df.loc[df["Votes"] == min_votes, COLS].head(5).to_string(index=False))

zero_vote_rated = ((df["Votes"] == 0) & (df["Aggregate rating"] > 0)).sum()
print(f"\nZero-vote restaurants that still have a rating: {zero_vote_rated}")

rated = df[df["Aggregate rating"] > 0].copy()
min_rated_votes = rated["Votes"].min()
print(f"Lowest votes among RATED restaurants: {min_rated_votes} "
      f"({(rated['Votes'] == min_rated_votes).sum()} restaurants). Sample:")
print(rated.loc[rated["Votes"] == min_rated_votes, COLS].head(5).to_string(index=False))

# ------------------------------------------- 2. votes vs rating correlation
pearson_r, pearson_p = stats.pearsonr(rated["Votes"], rated["Aggregate rating"])
spearman_r, spearman_p = stats.spearmanr(rated["Votes"], rated["Aggregate rating"])
log_r, _ = stats.pearsonr(np.log1p(rated["Votes"]), rated["Aggregate rating"])

print(f"\nCorrelation between votes and rating (rated restaurants, n = {len(rated)}):")
print(f"  Pearson  r   = {pearson_r:.3f}  (p = {pearson_p:.3g})")
print(f"  Spearman rho = {spearman_r:.3f}  (p = {spearman_p:.3g})")
print(f"  Pearson on log(votes) = {log_r:.3f}")
strength = "weak" if abs(spearman_r) < 0.3 else "moderate" if abs(spearman_r) < 0.6 else "strong"
print(f"-> {strength} positive relationship: better-rated restaurants tend to get more votes, "
      "but votes do not determine the rating."
      if spearman_r > 0 else f"-> {strength} negative relationship")

print("\nAverage votes by rating band:")
bands = pd.cut(rated["Aggregate rating"], bins=[0, 2, 3, 3.5, 4, 4.5, 5],
               labels=["<=2", "2-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5"])
band_stats = rated.groupby(bands, observed=True)["Votes"].agg(["mean", "median", "count"]).round(1)
print(band_stats)

# ------------------------------------------------------------------- charts
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

axes[0].scatter(rated["Votes"], rated["Aggregate rating"], s=6, alpha=0.3, color="#4C72B0")
axes[0].set_xscale("symlog", linthresh=10)
axes[0].set_xlabel("Votes (symlog scale)")
axes[0].set_ylabel("Aggregate rating")
axes[0].set_title(f"Votes vs rating (Spearman rho = {spearman_r:.2f})")

axes[1].bar(band_stats.index.astype(str), band_stats["median"], color="#55A868")
axes[1].set_xlabel("Rating band")
axes[1].set_ylabel("Median votes")
axes[1].set_title("Median votes by rating band")

plt.tight_layout()
fig.savefig(OUT / "level3_task2_votes_analysis.png", dpi=150)
df.nlargest(10, "Votes")[COLS].to_csv(OUT / "level3_task2_top10_votes.csv", index=False)
print(f"\nSaved chart and table to: {OUT}")
