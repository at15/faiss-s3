import json
import csv
from typing import Any
import polars as pl

def normalize_description(desc: Any) -> str:
    if desc is None:
        return ""
    if isinstance(desc, list):
        return " ".join(str(item) for item in desc)
    return str(desc)

def normalize_categories(cats: Any) -> str:
    if cats is None:
        return ""
    if isinstance(cats, list):
        sorted_cats = sorted(str(item) for item in cats)
        return ", ".join(sorted_cats)
    return str(cats)

def read_meta() -> list[dict[str, Any]]:
    meta_file = "meta_Software.jsonl"
    softwares: list[dict[str, Any]] = []
    with open(meta_file, 'r') as fp:
        for line in fp:
            j = json.loads(line.strip())
            software: dict[str, Any] = {
                "main_category": j.get("main_category", ""),
                "title": j.get("title", ""),
                "average_rating": j.get("average_rating") or 0.0,
                "rating_number": j.get("rating_number") or 0,
                "price": j.get("price") or 0.0,
                "description": normalize_description(j.get("description")),
                "categories": normalize_categories(j.get("categories")),
            }
            softwares.append(software)
    return softwares

def write_csv(softwares: list[dict[str, Any]], output_file: str = "software_data.csv") -> None:
    if not softwares:
        print("No data to write")
        return

    fieldnames = ["main_category", "title", "average_rating", "rating_number", "price", "description", "categories"]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(softwares)

    print(f"Written {len(softwares)} records to {output_file}")

def analyze_data(csv_file: str = "software_data.csv") -> None:
    print("\n" + "="*60)
    print("DATA ANALYSIS WITH POLARS")
    print("="*60)

    df = pl.read_csv(csv_file)

    print(f"\nTotal records: {len(df)}")
    print(f"\nDataFrame shape: {df.shape}")

    print("\n" + "-"*60)
    print("UNIQUE MAIN CATEGORIES")
    print("-"*60)
    main_categories = df.group_by("main_category").agg(pl.len().alias("count")).sort("count", descending=True)
    print(main_categories)

    print("\n" + "-"*60)
    print("CATEGORIES ANALYSIS")
    print("-"*60)
    null_categories = df.filter(pl.col("categories").is_null()).height
    empty_categories = df.filter(pl.col("categories") == "").height
    non_empty_categories = df.filter(pl.col("categories").is_not_null() & (pl.col("categories") != "")).height
    print(f"Records with categories: {non_empty_categories}")
    print(f"Records without categories (null): {null_categories}")
    print(f"Records without categories (empty string): {empty_categories}")

    # Get top categories (split by comma and count)
    print("\nTop 20 individual categories:")
    categories_df = df.filter(
        pl.col("categories").is_not_null() & (pl.col("categories") != "")
    ).select(
        pl.col("categories").str.split(", ").alias("category_list")
    ).explode("category_list").group_by("category_list").agg(
        pl.len().alias("count")
    ).sort("count", descending=True).head(20)
    print(categories_df)

    print("\n" + "-"*60)
    print("AVERAGE RATING DISTRIBUTION")
    print("-"*60)
    rating_stats = df.select([
        pl.col("average_rating").mean().alias("mean"),
        pl.col("average_rating").median().alias("median"),
        pl.col("average_rating").std().alias("std"),
        pl.col("average_rating").min().alias("min"),
        pl.col("average_rating").max().alias("max"),
    ])
    print(rating_stats)

    print("\nRating distribution (binned):")
    rating_bins = df.with_columns(
        pl.col("average_rating").cut([1, 2, 3, 4], labels=["0-1", "1-2", "2-3", "3-4", "4-5"]).alias("rating_bin")
    ).group_by("rating_bin").agg(pl.len().alias("count")).sort("rating_bin")
    print(rating_bins)

    print("\n" + "-"*60)
    print("RATING NUMBER DISTRIBUTION")
    print("-"*60)
    rating_number_stats = df.select([
        pl.col("rating_number").mean().alias("mean"),
        pl.col("rating_number").median().alias("median"),
        pl.col("rating_number").std().alias("std"),
        pl.col("rating_number").min().alias("min"),
        pl.col("rating_number").max().alias("max"),
    ])
    print(rating_number_stats)

    print("\nRating number distribution:")
    print(f"  Zero ratings: {df.filter(pl.col('rating_number') == 0).height}")
    print(f"  1-10 ratings: {df.filter((pl.col('rating_number') > 0) & (pl.col('rating_number') <= 10)).height}")
    print(f"  11-100 ratings: {df.filter((pl.col('rating_number') > 10) & (pl.col('rating_number') <= 100)).height}")
    print(f"  101-1000 ratings: {df.filter((pl.col('rating_number') > 100) & (pl.col('rating_number') <= 1000)).height}")
    print(f"  1000+ ratings: {df.filter(pl.col('rating_number') > 1000).height}")

    print("\n" + "-"*60)
    print("PRICE DISTRIBUTION")
    print("-"*60)
    price_stats = df.select([
        pl.col("price").mean().alias("mean"),
        pl.col("price").median().alias("median"),
        pl.col("price").std().alias("std"),
        pl.col("price").min().alias("min"),
        pl.col("price").max().alias("max"),
    ])
    print(price_stats)

    print(f"\nFree apps (price = 0): {df.filter(pl.col('price') == 0).height}")
    print(f"Paid apps (price > 0): {df.filter(pl.col('price') > 0).height}")

if __name__ == "__main__":
    softwares = read_meta()

    print(f"Total records: {len(softwares)}")
    print("\nFirst 3 records:")
    for software in softwares[:3]:
        print(software)

    write_csv(softwares)

    analyze_data()