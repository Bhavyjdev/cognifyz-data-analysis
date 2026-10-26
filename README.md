# Cognifyz Data Analysis Internship - Practice Solutions

Restaurant dataset (`Dataset.csv`, 9,551 restaurants). One standalone script per task - each has its own imports and can be run on its own.

| File | Task |
|---|---|
| `level1_task1_top_cuisines.py` | Top 3 cuisines + % of restaurants serving them |
| `level1_task2_city_analysis.py` | City with most restaurants, average rating per city |
| `level1_task3_price_range_distribution.py` | Price range bar chart + percentages |
| `level1_task4_online_delivery.py` | % with online delivery, rating comparison |
| `level2_task1_restaurant_ratings.py` | Rating distribution, average votes |
| `level2_task2_cuisine_combination.py` | Common cuisine combinations, rating by combination |
| `level2_task3_geographic_analysis.py` | Lat/long maps, density and DBSCAN clusters |
| `level2_task4_restaurant_chains.py` | Chains, their ratings and popularity |
| `level3_task1_restaurant_reviews.py` | Keywords / text length vs rating (see note) |
| `level3_task2_votes_analysis.py` | Highest/lowest votes, votes vs rating correlation |
| `level3_task3_price_vs_services.py` | Price range vs online delivery / table booking |

## Run
```
pip install pandas matplotlib scipy scikit-learn
python level1_task1_top_cuisines.py
```
Charts and CSV tables are saved to `outputs/`. Optional: `pip install folium` for an interactive map in Level 2 Task 3.

## Notes
- Rating `0` means "Not rated" (2,148 restaurants); these are excluded from rating averages.
- 497 rows have (0, 0) coordinates and are dropped from the maps.
- The dataset has **no review-text column**, so Level 3 Task 1 uses restaurant name + cuisines as a stand-in text (documented in the script).
- "Chain" in Level 2 Task 4 = same restaurant name at 2+ outlets (heuristic).
