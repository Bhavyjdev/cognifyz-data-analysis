"""
Cognifyz Data Analysis Internship - Level 3, Task 3: Price Range vs. Online Delivery and Table Booking

Task:
  1. Analyze if there is a relationship between the price range and the availability
     of online delivery and table booking.
  2. Determine if higher-priced restaurants are more likely to offer these services.

Approach:
  - For each price range (1-4): % of restaurants with online delivery, % with table booking.
  - Chi-square test of independence (is the relationship real, not chance?).
  - Cramer's V for the strength of the association and Spearman correlation for direction.

Run:  python level3_task3_price_vs_services.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- load data
df = pd.read_csv(next(BASE.glob("Dataset*.csv")), encoding="utf-8-sig")
print(f"Restaurants analysed: {len(df)}")

SERVICES = {"Has Online delivery": "Online delivery", "Has Table booking": "Table booking"}


def cramers_v(table):
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1)))


results = {}
for column, label in SERVICES.items():
    print("\n" + "=" * 60)
    print(f"Price range vs {label}")
    print("=" * 60)

    counts = pd.crosstab(df["Price range"], df[column])
    pct = (df.groupby("Price range")[column].apply(lambda s: (s == "Yes").mean() * 100)).round(2)

    table = counts.copy()
    table["% with service"] = pct
    print(table)

    chi2, p, dof, _ = stats.chi2_contingency(counts)
    v = cramers_v(counts)
    rho, rho_p = stats.spearmanr(df["Price range"], (df[column] == "Yes").astype(int))
    print(f"\nChi-square = {chi2:.1f}, dof = {dof}, p = {p:.3g} "
          f"-> {'significant' if p < 0.05 else 'not significant'} relationship")
    print(f"Cramer's V = {v:.3f} ({'weak' if v < 0.1 else 'small' if v < 0.3 else 'moderate' if v < 0.5 else 'strong'})")
    print(f"Spearman rho (price range vs service) = {rho:.3f}")
    results[label] = pct

    best = pct.idxmax()
    trend = "rises steadily" if pct.is_monotonic_increasing else (
        "falls steadily" if pct.is_monotonic_decreasing else "is not a simple up/down trend")
    print(f"-> Highest share is in price range {best} ({pct.max():.1f}%); "
          f"the share {trend} with price range.")

summary = pd.DataFrame(results)
summary.index.name = "Price range"
print("\n" + "=" * 60)
print("Summary: % of restaurants offering each service, by price range")
print("=" * 60)
print(summary)

# ------------------------------------------------------------------- chart
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(summary))
width = 0.38
b1 = ax.bar(x - width / 2, summary["Online delivery"], width, label="Online delivery", color="#4C72B0")
b2 = ax.bar(x + width / 2, summary["Table booking"], width, label="Table booking", color="#DD8452")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"Price range {i}" for i in summary.index])
ax.set_ylabel("% of restaurants offering the service")
ax.set_title("Online delivery and table booking by price range")
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "level3_task3_price_vs_services.png", dpi=150)
summary.to_csv(OUT / "level3_task3_price_vs_services.csv")
print(f"\nSaved chart and table to: {OUT}")
