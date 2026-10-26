"""
Cognifyz Data Analysis Internship - Level 2, Task 3: Geographic Analysis

Task:
  1. Plot the locations of restaurants on a map using longitude and latitude.
  2. Identify any patterns or clusters of restaurants in specific areas.

Approach:
  - 497 rows have Longitude = Latitude = 0 (missing coordinates) and are dropped.
  - Plot 1: world scatter of every restaurant (drawn with matplotlib, no internet needed).
  - Plot 2: density (hexbin) of the Delhi-NCR region, where most restaurants are.
  - Clusters: restaurants per city / locality, plus a DBSCAN density clustering
    (scikit-learn) on the coordinates to find dense groups automatically.
  - If the optional 'folium' package is installed, an interactive map (HTML) is also saved:
        pip install folium

Run:  python level2_task3_geographic_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- load data
df = pd.read_csv(next(BASE.glob("Dataset*.csv")), encoding="utf-8-sig")

missing = ((df["Longitude"] == 0) & (df["Latitude"] == 0)).sum()
geo = df[~((df["Longitude"] == 0) & (df["Latitude"] == 0))].copy()
print(f"Rows: {len(df)} | dropped (0,0 coordinates): {missing} | plotted: {len(geo)}")

# ------------------------------------------------------- plot 1: world view
fig, ax = plt.subplots(figsize=(13, 6))
sc = ax.scatter(geo["Longitude"], geo["Latitude"], c=geo["Aggregate rating"], cmap="RdYlGn",
                s=6, alpha=0.6)
plt.colorbar(sc, ax=ax, label="Aggregate rating")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Restaurant locations worldwide (colour = rating)")
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / "level2_task3_map_world.png", dpi=150)
plt.close(fig)

# ------------------------------------------------ plot 2: Delhi-NCR density
ncr = geo[geo["Longitude"].between(76.5, 78.0) & geo["Latitude"].between(27.9, 29.2)]
print(f"\nRestaurants in the Delhi-NCR box: {len(ncr)} ({len(ncr) / len(geo) * 100:.1f}% of plotted)")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].scatter(ncr["Longitude"], ncr["Latitude"], s=5, alpha=0.4, color="#4C72B0")
axes[0].set_title("Delhi-NCR restaurant locations")
hb = axes[1].hexbin(ncr["Longitude"], ncr["Latitude"], gridsize=45, cmap="YlOrRd", mincnt=1)
plt.colorbar(hb, ax=axes[1], label="Restaurants per cell")
axes[1].set_title("Delhi-NCR density (hexbin)")
for ax in axes:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
plt.tight_layout()
fig.savefig(OUT / "level2_task3_map_delhi_ncr_density.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------- clusters by city / area
print("\nTop 10 cities by number of restaurants:")
print(geo["City"].value_counts().head(10))

print("\nTop 10 localities (most restaurants):")
print(geo["Locality"].value_counts().head(10))

country_share = geo["Country Code"].value_counts(normalize=True).head(3) * 100
print("\nShare of plotted restaurants by Country Code (top 3):")
print(country_share.round(1))

# ------------------------------------------------------ DBSCAN density clusters
# eps ~ 0.01 degrees (~1 km). Dense areas become clusters, isolated points -> -1.
coords = geo[["Latitude", "Longitude"]].to_numpy()
labels = DBSCAN(eps=0.01, min_samples=15).fit_predict(coords)
geo["Cluster"] = labels
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"\nDBSCAN (eps=0.01 deg, min_samples=15): {n_clusters} dense clusters found, "
      f"{(labels == -1).sum()} restaurants not in any cluster")

clusters = (
    geo[geo["Cluster"] >= 0]
    .groupby("Cluster")
    .agg(restaurants=("Restaurant ID", "count"),
         main_city=("City", lambda s: s.mode().iat[0]),
         main_locality=("Locality", lambda s: s.mode().iat[0]),
         avg_rating=("Aggregate rating", lambda s: s[s > 0].mean()),
         lat=("Latitude", "mean"), lon=("Longitude", "mean"))
    .sort_values("restaurants", ascending=False)
    .round(3)
)
print("\nLargest 10 clusters:")
print(clusters.head(10).to_string())
clusters.to_csv(OUT / "level2_task3_dbscan_clusters.csv")

# ------------------------------------------------ optional interactive map
try:
    import folium
    from folium.plugins import HeatMap

    sample = geo.sample(min(len(geo), 3000), random_state=42)
    fmap = folium.Map(location=[geo["Latitude"].median(), geo["Longitude"].median()],
                      zoom_start=5, tiles="OpenStreetMap")
    HeatMap(sample[["Latitude", "Longitude"]].values.tolist(), radius=8).add_to(fmap)
    fmap.save(str(OUT / "level2_task3_interactive_map.html"))
    print("\nSaved interactive map (folium).")
except ImportError:
    print("\n(folium not installed - skipped the optional interactive map: pip install folium)")

print(f"\nSaved maps and tables to: {OUT}")
