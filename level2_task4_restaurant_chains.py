"""
Cognifyz Data Analysis Internship - Level 2, Task 4: Restaurant Chains

Task:
  1. Identify if there are any restaurant chains present in the dataset.
  2. Analyze the ratings and popularity of different restaurant chains.

Approach:
  - The dataset has no 'chain' flag, so a chain is defined as the same Restaurant Name
    appearing at 2 or more outlets (rows). This is a name-based heuristic: two unrelated
    restaurants that share a name would be counted together.
  - Popularity = total votes and average votes per outlet.
  - Ratings use rated outlets only (rating 0 = "Not rated").
  - For the ranking charts only chains with at least MIN_OUTLETS outlets are used.

Run:  python level2_task4_restaurant_chains.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

MIN_OUTLETS = 10  # chains with fewer outlets are not ranked

# ---------------------------------------------------------------- load data
df = pd.read_csv(next(BASE.glob("Dataset*.csv")), encoding="utf-8-sig")
df["Restaurant Name"] = df["Restaurant Name"].str.strip()

# ------------------------------------------------------ 1. identify chains
outlet_counts = df["Restaurant Name"].value_counts()
chain_names = outlet_counts[outlet_counts >= 2].index
n_chains = len(chain_names)
chain_outlets = outlet_counts[outlet_counts >= 2].sum()

print(f"Unique restaurant names: {len(outlet_counts)}")
print(f"Names with 2+ outlets (chains): {n_chains}")
print(f"Outlets belonging to a chain: {chain_outlets} ({chain_outlets / len(df) * 100:.1f}% of all restaurants)")
print(f"Names with {MIN_OUTLETS}+ outlets: {(outlet_counts >= MIN_OUTLETS).sum()}")

print("\nTop 10 chains by number of outlets:")
print(outlet_counts.head(10))

# ------------------------------------------- 2. ratings + popularity per chain
chains = df[df["Restaurant Name"].isin(chain_names)]


def chain_summary(data):
    rated = data[data["Aggregate rating"] > 0]
    out = data.groupby("Restaurant Name").agg(
        outlets=("Restaurant ID", "count"),
        total_votes=("Votes", "sum"),
        avg_votes=("Votes", "mean"),
    )
    out["avg_rating"] = rated.groupby("Restaurant Name")["Aggregate rating"].mean()
    out["rated_outlets"] = rated.groupby("Restaurant Name")["Aggregate rating"].count()
    return out.round(2)


summary = chain_summary(chains)
big = summary[summary["outlets"] >= MIN_OUTLETS]

print(f"\nChains with >= {MIN_OUTLETS} outlets (sorted by outlets):")
print(big.sort_values("outlets", ascending=False).to_string())

print(f"\nHighest-rated chains (>= {MIN_OUTLETS} outlets, >= 5 rated outlets):")
print(big[big["rated_outlets"] >= 5].sort_values("avg_rating", ascending=False).head(10).to_string())

print(f"\nMost popular chains by total votes (>= {MIN_OUTLETS} outlets):")
print(big.sort_values("total_votes", ascending=False).head(10).to_string())

# chain vs non-chain
non_chain = df[~df["Restaurant Name"].isin(chain_names)]
avg_chain = chains.loc[chains["Aggregate rating"] > 0, "Aggregate rating"].mean()
avg_non = non_chain.loc[non_chain["Aggregate rating"] > 0, "Aggregate rating"].mean()
print(f"\nAverage rating - chain outlets: {avg_chain:.3f} | independent restaurants: {avg_non:.3f}")
print(f"Average votes  - chain outlets: {chains['Votes'].mean():.1f} | independent restaurants: {non_chain['Votes'].mean():.1f}")

# ------------------------------------------------------------------- chart
top = big.sort_values("outlets", ascending=False).head(10).iloc[::-1]
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)

axes[0].barh(top.index, top["outlets"], color="#4C72B0")
axes[0].set_title("Outlets")
axes[1].barh(top.index, top["avg_rating"], color="#55A868")
axes[1].set_xlim(0, 5)
axes[1].set_title("Average rating (rated outlets)")
axes[2].barh(top.index, top["avg_votes"], color="#DD8452")
axes[2].set_title("Average votes per outlet")
fig.suptitle("Top 10 restaurant chains by number of outlets")
plt.tight_layout()
fig.savefig(OUT / "level2_task4_restaurant_chains.png", dpi=150)
summary.sort_values("outlets", ascending=False).to_csv(OUT / "level2_task4_chain_summary.csv")
print(f"\nSaved chart and table to: {OUT}")
