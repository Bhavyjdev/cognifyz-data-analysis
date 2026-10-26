"""
Cognifyz Data Analysis Internship - Level 3, Task 1: Restaurant Reviews

Task:
  1. Analyze the text reviews to identify the most common positive and negative keywords.
  2. Calculate the average length of reviews and explore if there is a relationship
     between review length and rating.

IMPORTANT - dataset limitation:
  This dataset has NO review text column. The only text-like fields are
  'Rating text' (a one-word label: Excellent / Very Good / Good / Average / Poor / Not rated),
  'Restaurant Name', 'Cuisines', 'Address' and 'Locality'.
  So this script does the closest honest version of the task:

    - Keywords: words are pulled from the text we do have (Restaurant Name + Cuisines),
      then compared between POSITIVE restaurants (rating >= 4.0, i.e. "Very Good" / "Excellent")
      and NEGATIVE restaurants (rating < 2.5, i.e. "Poor").
      Keywords that appear much more in one group are listed as positive / negative keywords.
    - "Review length": there are no reviews, so Rating text and Restaurant Name length are
      measured instead, and the correlation with rating is tested. Treat this as a
      demonstration of the method, not a finding about real reviews.
    - If you later get a file with real review text, set REVIEW_COLUMN below and the
      same code works on it.

Run:  python level3_task1_restaurant_reviews.py
"""

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

REVIEW_COLUMN = None  # set to a real review-text column name if you have one
POSITIVE_MIN = 4.0
NEGATIVE_MAX = 2.5
MIN_WORD_COUNT = 15  # ignore words seen fewer times than this in the group

# Words that carry no meaning for this analysis
STOPWORDS = {
    "the", "and", "of", "a", "an", "in", "at", "by", "to", "for", "on", "restaurant", "cafe",
    "bar", "n", "s", "co", "amp", "pvt", "ltd", "no", "new", "delhi", "ncr",
}

# ---------------------------------------------------------------- load data
df = pd.read_csv(next(BASE.glob("Dataset*.csv")), encoding="utf-8-sig")
rated = df[df["Aggregate rating"] > 0].copy()

if REVIEW_COLUMN and REVIEW_COLUMN in df.columns:
    rated["text"] = rated[REVIEW_COLUMN].fillna("")
    print(f"Using real review text from column: {REVIEW_COLUMN}")
else:
    rated["text"] = (rated["Restaurant Name"].fillna("") + " " + rated["Cuisines"].fillna(""))
    print("No review-text column in this dataset -> using Restaurant Name + Cuisines as the text.\n")


def tokenize(text):
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


# ---------------------------------------------- 1. positive vs negative keywords
positive = rated[rated["Aggregate rating"] >= POSITIVE_MIN]
negative = rated[rated["Aggregate rating"] < NEGATIVE_MAX]
print(f"Positive group (rating >= {POSITIVE_MIN}): {len(positive)} restaurants")
print(f"Negative group (rating <  {NEGATIVE_MAX}): {len(negative)} restaurants")

pos_counts = Counter(w for t in positive["text"] for w in tokenize(t))
neg_counts = Counter(w for t in negative["text"] for w in tokenize(t))

print("\nMost common words - positive group:")
print(pos_counts.most_common(10))
print("\nMost common words - negative group:")
print(neg_counts.most_common(10))

# Which words lean positive / negative? Compare each word's share of its group.
pos_total, neg_total = sum(pos_counts.values()), sum(neg_counts.values())
rows = []
for word in set(pos_counts) | set(neg_counts):
    if pos_counts[word] + neg_counts[word] < MIN_WORD_COUNT:
        continue
    pos_share = (pos_counts[word] + 0.5) / pos_total
    neg_share = (neg_counts[word] + 0.5) / neg_total
    rows.append((word, pos_counts[word], neg_counts[word], pos_share / neg_share))
lean = pd.DataFrame(rows, columns=["word", "positive_count", "negative_count", "pos_to_neg_ratio"])
lean = lean.sort_values("pos_to_neg_ratio", ascending=False)

print("\nKeywords that lean POSITIVE (appear far more in high-rated restaurants):")
print(lean.head(10).round(2).to_string(index=False))
print("\nKeywords that lean NEGATIVE (appear far more in low-rated restaurants):")
print(lean.tail(10).iloc[::-1].round(2).to_string(index=False))

# 'Rating text' is the one real sentiment label in the data - show its split
print("\nRating text labels (the dataset's own sentiment wording):")
print(df["Rating text"].value_counts())

# ---------------------------------------------- 2. text length vs rating
rated["length_chars"] = rated["text"].str.len()
rated["length_words"] = rated["text"].str.split().str.len()
print(f"\nAverage text length: {rated['length_chars'].mean():.1f} characters, "
      f"{rated['length_words'].mean():.1f} words")

rho, p = stats.spearmanr(rated["length_chars"], rated["Aggregate rating"])
print(f"Spearman correlation (text length vs rating): rho = {rho:.3f}, p = {p:.3g}")
print("-> " + ("practically no relationship (|rho| < 0.1)" if abs(rho) < 0.1
               else "a weak-to-moderate relationship"))

by_label = rated.groupby("Rating text")["length_chars"].mean().round(1)
print("\nAverage text length by Rating text label:")
print(by_label)

# ------------------------------------------------------------------- charts
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

top_pos = lean.head(8).iloc[::-1]
axes[0].barh(top_pos["word"], top_pos["pos_to_neg_ratio"], color="#55A868")
axes[0].set_title("Positive-leaning keywords\n(ratio vs negative group)")

top_neg = lean.tail(8)
axes[1].barh(top_neg["word"], 1 / top_neg["pos_to_neg_ratio"], color="#C44E52")
axes[1].set_title("Negative-leaning keywords\n(ratio vs positive group)")

axes[2].scatter(rated["length_chars"], rated["Aggregate rating"], s=6, alpha=0.3, color="#4C72B0")
axes[2].set_xlabel("Text length (characters)")
axes[2].set_ylabel("Aggregate rating")
axes[2].set_title(f"Text length vs rating (rho = {rho:.2f})")

plt.tight_layout()
fig.savefig(OUT / "level3_task1_restaurant_reviews.png", dpi=150)
lean.round(3).to_csv(OUT / "level3_task1_keywords.csv", index=False)
print(f"\nSaved chart and table to: {OUT}")
