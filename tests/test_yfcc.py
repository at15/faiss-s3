# pip install polars
import polars as pl

# https://huggingface.co/datasets/ARKseal/YFCC14M_subset_webdataset
df = pl.read_parquet("yfcc14m.parquet")

print("Columns:", df.columns)
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
pl.Config.set_tbl_cols(-1)
print(df.head())

# Count unique tags (assuming tags are in 'usertags' column and comma-separated)
print("\n=== Tag Analysis ===")
tag_counts = (
    df.select("usertags")
    .filter(pl.col("usertags").is_not_null())
    .with_columns(pl.col("usertags").str.split(","))
    .explode("usertags")
    .with_columns(pl.col("usertags").str.strip_chars())
    .group_by("usertags")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)

# Total unique tags
print(f"Total unique tags: {len(tag_counts)}")

# Top 20 tags
print("\nTop 20 Tags:")
print(tag_counts.head(20))

# Distribution: how many tags appear X times
print("\n=== Distribution of Images per Tag ===")
distribution = (
    tag_counts
    .group_by("count")
    .agg(pl.len().alias("num_tags"))
    .sort("count", descending=True)
)
print(distribution.head(20))

# Histogram-style bins for better visualization
print("\n=== Distribution by Bins ===")
bins = [1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, float('inf')]
bin_labels = ["1", "2-4", "5-9", "10-49", "50-99", "100-499", "500-999", "1000-4999", "5000-9999", "10000+"]

for i in range(len(bins) - 1):
    count = tag_counts.filter(
        (pl.col("count") >= bins[i]) & (pl.col("count") < bins[i+1])
    ).shape[0]
    print(f"{bin_labels[i]:12} images: {count:8} tags")

# Statistics
print("\n=== Statistics ===")
print(f"Min images per tag: {tag_counts['count'].min()}")
print(f"Max images per tag: {tag_counts['count'].max()}")
print(f"Mean images per tag: {tag_counts['count'].mean():.2f}")
print(f"Median images per tag: {tag_counts['count'].median()}")

"""
=== Tag Analysis ===
Total unique tags: 3671471

Top 20 Tags:
shape: (20, 2)
┌────────────┬─────────┐
│ usertags   ┆ count   │
│ ---        ┆ ---     │
│ str        ┆ u32     │
╞════════════╪═════════╡
│            ┆ 1738922 │
│ california ┆ 401521  │
│ usa        ┆ 368971  │
│ nikon      ┆ 336456  │
│ london     ┆ 296132  │
│ …          ┆ …       │
│ 2010       ┆ 212302  │
│ water      ┆ 207105  │
│ 2009       ┆ 196881  │
│ 2013       ┆ 191430  │
│ sky        ┆ 187514  │
└────────────┴─────────┘

=== Distribution of Images per Tag ===
shape: (20, 2)
┌─────────┬──────────┐
│ count   ┆ num_tags │
│ ---     ┆ ---      │
│ u32     ┆ u32      │
╞═════════╪══════════╡
│ 1738922 ┆ 1        │
│ 401521  ┆ 1        │
│ 368971  ┆ 1        │
│ 336456  ┆ 1        │
│ 296132  ┆ 1        │
│ …       ┆ …        │
│ 212302  ┆ 1        │
│ 207105  ┆ 1        │
│ 196881  ┆ 1        │
│ 191430  ┆ 1        │
│ 187514  ┆ 1        │
└─────────┴──────────┘

=== Distribution by Bins ===
1            images:  1810535 tags
2-4          images:   902544 tags
5-9          images:   371184 tags
10-49        images:   415103 tags
50-99        images:    74401 tags
100-499      images:    73892 tags
500-999      images:    11131 tags
1000-4999    images:    10025 tags
5000-9999    images:     1315 tags
10000+       images:     1341 tags

=== Statistics ===
Min images per tag: 1
Max images per tag: 1738922
Mean images per tag: 32.89
Median images per tag: 2.0
"""