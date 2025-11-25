use anyhow::Result;
use ordered_float::OrderedFloat;
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Instant;

use tantivy::collector::TopDocs;
use tantivy::query::Occur;
use tantivy::query::{BooleanQuery, QueryParser, RangeQuery};
use tantivy::schema::*;
use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument, doc};

#[derive(Debug, Deserialize)]
struct SoftwareRecord {
    main_category: String,
    title: String,
    average_rating: f64,
    rating_number: u64,
    price: f64,
    description: String,
    categories: Option<String>,
}

fn load_software_data(csv_path: &str) -> Result<Vec<SoftwareRecord>> {
    let mut reader = csv::Reader::from_path(csv_path)?;
    let mut records = Vec::new();

    for result in reader.deserialize() {
        let record: SoftwareRecord = result?;
        records.push(record);
    }

    println!("Loaded {} software records from {}", records.len(), csv_path);
    Ok(records)
}

#[test]
fn test_tantivy_fts() -> tantivy::Result<()> {
    println!("=== LOADING SOFTWARE DATA FROM CSV ===");

    // Load CSV data
    let software_data = load_software_data("software_data.csv")
        .expect("Failed to load software data");

    // 1. Define the schema with text and numeric fields matching our CSV
    let mut schema_builder = Schema::builder();

    // Text fields - stored and indexed for full-text search
    let main_category =
        schema_builder.add_text_field("main_category", TEXT | STORED);
    let title = schema_builder.add_text_field("title", TEXT | STORED);
    let description =
        schema_builder.add_text_field("description", TEXT | STORED);
    let categories = schema_builder.add_text_field("categories", TEXT | STORED);

    // Numeric fields
    let price = schema_builder.add_f64_field("price", INDEXED | STORED | FAST);
    let average_rating =
        schema_builder.add_f64_field("average_rating", INDEXED | STORED | FAST);
    let rating_number =
        schema_builder.add_u64_field("rating_number", INDEXED | STORED | FAST);

    let schema = schema_builder.build();

    // 2. Create the index (in-memory for this example)
    let index = Index::create_in_ram(schema.clone());

    // For persistent storage, use:
    // let index = Index::create_in_dir("./software_index", schema.clone())?;

    // 3. Get an IndexWriter with 50MB buffer
    let mut index_writer: IndexWriter = index.writer(50_000_000)?;

    // 4. Index all software documents from CSV
    println!("Indexing {} documents...", software_data.len());
    let start_time = Instant::now();

    for record in &software_data {
        index_writer.add_document(doc!(
            main_category => record.main_category.as_str(),
            title => record.title.as_str(),
            description => record.description.as_str(),
            categories => record.categories.as_deref().unwrap_or(""),
            price => record.price,
            average_rating => record.average_rating,
            rating_number => record.rating_number,
        ))?;
    }

    // 5. Commit the changes
    index_writer.commit()?;

    let elapsed = start_time.elapsed();
    println!("Indexed {} documents in {:?}", software_data.len(), elapsed);

    // 6. Get a reader - ReloadPolicy determines when index changes are visible
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;

    let searcher = reader.searcher();

    println!("\n=== TEXT SEARCH: 'game' ===");
    // 7. Text search using QueryParser - search in title, description, and categories
    let query_parser =
        QueryParser::for_index(&index, vec![title, description, categories]);
    let query = query_parser.parse_query("game")?;

    let top_docs = searcher.search(&query, &TopDocs::with_limit(5))?;
    println!("Found {} results:", top_docs.len());

    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let title_val = retrieved_doc
            .get_first(title)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let rating_val = retrieved_doc
            .get_first(average_rating)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let categories_val = retrieved_doc
            .get_first(categories)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        println!(
            "  [Score: {:.2}] {} (Rating: {:.1}, Categories: {})",
            score,
            title_val,
            rating_val,
            if categories_val.is_empty() { "None" } else { categories_val }
        );
    }

    println!("\n=== FREE SOFTWARE: price = 0 ===");
    // 8. Numeric exact match query: free software
    let free_query = RangeQuery::new(
        std::ops::Bound::Included(Term::from_field_f64(price, 0.0)),
        std::ops::Bound::Included(Term::from_field_f64(price, 0.0)),
    );

    let top_docs = searcher.search(&free_query, &TopDocs::with_limit(5))?;
    println!("Found {} free apps (showing 5):", top_docs.len());
    for (_score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let title_val = retrieved_doc
            .get_first(title)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let rating_val = retrieved_doc
            .get_first(average_rating)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        println!("  {} (Rating: {:.1})", title_val, rating_val);
    }

    println!(
        "\n=== CATEGORY SEARCH: Education books with 'English' in title ==="
    );
    // 9. Search with category filter and text query

    // Text search in title
    let title_parser = QueryParser::for_index(&index, vec![title]);
    let title_query = title_parser.parse_query("English")?;

    // Category filter - search for "Education" token in categories field
    let category_parser = QueryParser::for_index(&index, vec![categories]);
    let category_query = category_parser.parse_query("Education")?;

    // Combine with boolean query
    let bool_query = BooleanQuery::from(vec![
        (Occur::Must, title_query), // Scoring query on title
        (Occur::Must, category_query), // Must have Education in categories
    ]);

    let top_docs = searcher.search(&bool_query, &TopDocs::with_limit(5))?;
    println!("Found {} results:", top_docs.len());
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let title_val = retrieved_doc
            .get_first(title)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let rating_val = retrieved_doc
            .get_first(average_rating)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let categories_val = retrieved_doc
            .get_first(categories)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        println!(
            "  [Score: {:.2}] {} (Rating: {:.1})",
            score, title_val, rating_val
        );
        println!(
            "    Categories: {}",
            if categories_val.is_empty() { "None" } else { categories_val }
        );
    }

    println!("\n=== HIGH RATED GAMES: rating >= 4.5 + 'game' ===");
    // 10. Combined text + numeric query
    use tantivy::query::BooleanQuery;
    use tantivy::query::Occur;

    let text_query = query_parser.parse_query("game")?;
    let rating_query = RangeQuery::new(
        std::ops::Bound::Included(Term::from_field_f64(average_rating, 4.5f64)),
        std::ops::Bound::Unbounded,
    );

    let bool_query = BooleanQuery::from(vec![
        (Occur::Must, text_query),
        (Occur::Must, Box::new(rating_query)),
    ]);

    let top_docs = searcher.search(&bool_query, &TopDocs::with_limit(10))?;
    println!("Found {} highly rated games:", top_docs.len());
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let title_val = retrieved_doc
            .get_first(title)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let rating_val = retrieved_doc
            .get_first(average_rating)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let price_val = retrieved_doc
            .get_first(price)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        println!(
            "  [Score: {:.2}] {} (Rating: {:.1}, Price: ${:.2})",
            score, title_val, rating_val, price_val
        );
    }

    println!("\n=== STATISTICS ===");
    // 11. Show some statistics using fast fields
    println!("Total documents indexed: {}", searcher.num_docs());

    Ok(())
}

struct Product {
    price: f64,
    category: String,
    cluster: u64,
    cluster_local_id: u64, // TODO: we might need the vector id (across clusters) as well
}

// Used as key for grouping
// OrderedFloat allows f64 to be used in HashMap keys
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProductAttributes {
    price: OrderedFloat<f64>,
    category: String,
}

fn build_attributes_to_clusters_map(
    products: &[Product],
) -> HashMap<ProductAttributes, HashMap<u64, Vec<u64>>> {
    let mut attributes_to_clusters: HashMap<
        ProductAttributes,
        HashMap<u64, Vec<u64>>,
    > = HashMap::new();

    for product in products {
        let attributes = ProductAttributes {
            price: OrderedFloat(product.price),
            category: product.category.clone(),
        };

        attributes_to_clusters
            .entry(attributes)
            .or_insert_with(HashMap::new)
            .entry(product.cluster)
            .or_insert_with(Vec::new)
            .push(product.cluster_local_id);
    }

    attributes_to_clusters
}

fn build_cluster_level_index(
    attributes_to_clusters: &HashMap<ProductAttributes, HashMap<u64, Vec<u64>>>,
) -> tantivy::Result<Index> {
    let mut schema_builder = Schema::builder();
    let price_field = schema_builder.add_f64_field("price", INDEXED | STORED);
    let category_field =
        schema_builder.add_text_field("category", STRING | STORED);
    let cluster_field =
        schema_builder.add_u64_field("cluster", INDEXED | STORED);
    let vector_count_field =
        schema_builder.add_u64_field("vector_count", STORED);

    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut index_writer: IndexWriter = index.writer(50_000_000)?;

    for (attrs, clusters) in attributes_to_clusters {
        for (cluster_id, local_ids) in clusters {
            index_writer.add_document(doc!(
                price_field => attrs.price.0,
                category_field => attrs.category.as_str(),
                cluster_field => *cluster_id,
                vector_count_field => local_ids.len() as u64,
            ))?;
        }
    }

    index_writer.commit()?;
    Ok(index)
}

fn query_cluster_level_index(
    index: &Index,
    price: f64,
    category: &str,
) -> tantivy::Result<Vec<u64>> {
    let schema = index.schema();
    let price_field = schema.get_field("price").unwrap();
    let category_field = schema.get_field("category").unwrap();
    let cluster_field = schema.get_field("cluster").unwrap();

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();

    let price_query: Box<dyn tantivy::query::Query> =
        Box::new(RangeQuery::new(
            std::ops::Bound::Included(Term::from_field_f64(price_field, price)),
            std::ops::Bound::Included(Term::from_field_f64(price_field, price)),
        ));

    let category_parser = QueryParser::for_index(index, vec![category_field]);
    let category_query = category_parser.parse_query(category)?;

    let bool_query = BooleanQuery::from(vec![
        (Occur::Must, price_query),
        (Occur::Must, category_query),
    ]);

    let top_docs = searcher.search(&bool_query, &TopDocs::with_limit(10))?;

    let mut found_clusters = Vec::new();
    for (_score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let cluster = retrieved_doc
            .get_first(cluster_field)
            .and_then(|v| v.as_u64())
            .unwrap();
        found_clusters.push(cluster);
    }

    Ok(found_clusters)
}

fn query_per_cluster_index(
    cluster_index: &Index,
    price: f64,
    category: &str,
) -> tantivy::Result<Vec<u64>> {
    let schema = cluster_index.schema();
    let price_field = schema.get_field("price").unwrap();
    let category_field = schema.get_field("category").unwrap();
    let cluster_local_id_field = schema.get_field("cluster_local_id").unwrap();

    let reader = cluster_index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();

    let price_query: Box<dyn tantivy::query::Query> =
        Box::new(RangeQuery::new(
            std::ops::Bound::Included(Term::from_field_f64(price_field, price)),
            std::ops::Bound::Included(Term::from_field_f64(price_field, price)),
        ));

    let category_parser =
        QueryParser::for_index(cluster_index, vec![category_field]);
    let category_query = category_parser.parse_query(category)?;

    let bool_query = BooleanQuery::from(vec![
        (Occur::Must, price_query),
        (Occur::Must, category_query),
    ]);

    let top_docs = searcher.search(&bool_query, &TopDocs::with_limit(10))?;

    let mut found_local_ids = Vec::new();
    for (_score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let local_id = retrieved_doc
            .get_first(cluster_local_id_field)
            .and_then(|v| v.as_u64())
            .unwrap();
        found_local_ids.push(local_id);
    }

    found_local_ids.sort();
    Ok(found_local_ids)
}

fn build_per_cluster_index(
    products: &Vec<Product>,
    n_clusters: u64,
) -> tantivy::Result<Vec<Index>> {
    let mut schema_builder = Schema::builder();
    let price_field = schema_builder.add_f64_field("price", INDEXED | STORED);
    let category_field =
        schema_builder.add_text_field("category", STRING | STORED);
    let cluster_local_id_field =
        schema_builder.add_u64_field("cluster_local_id", INDEXED | STORED);
    let schema = schema_builder.build();

    let mut indexes: Vec<Index> = Vec::new();
    let mut index_writers: Vec<IndexWriter> = Vec::new();

    for _cluster in 0..n_clusters {
        let index = Index::create_in_ram(schema.clone());
        let writer = index.writer(50_000_000)?;
        indexes.push(index);
        index_writers.push(writer);
    }

    for product in products {
        let cluster_idx = product.cluster as usize;
        index_writers[cluster_idx].add_document(doc!(
            price_field => product.price,
            category_field => product.category.as_str(),
            cluster_local_id_field => product.cluster_local_id,
        ))?;
    }

    for writer in index_writers.iter_mut() {
        writer.commit()?;
    }

    Ok(indexes)
}

#[test]
fn test_tantivy_attributes_to_cluster() -> tantivy::Result<()> {
    println!("=== TWO-LEVEL ATTRIBUTES INDEX EXAMPLE ===\n");

    let n_clusters = 2;
    let products = vec![
        Product {
            price: 1.0,
            category: "book".to_string(),
            cluster: 0,
            cluster_local_id: 0,
        },
        Product {
            price: 1.0,
            category: "book".to_string(),
            cluster: 0,
            cluster_local_id: 1,
        },
        Product {
            price: 2.0,
            category: "software".to_string(),
            cluster: 1,
            cluster_local_id: 0,
        },
    ];

    println!("Input products:");
    for p in &products {
        println!(
            "  price={}, category={}, cluster={}, local_id={}",
            p.price, p.category, p.cluster, p.cluster_local_id
        );
    }

    let attributes_to_clusters = build_attributes_to_clusters_map(&products);

    println!("\n=== NESTED DATA STRUCTURE ===");
    println!("Unique attribute sets: {}", attributes_to_clusters.len());
    for (attrs, clusters) in &attributes_to_clusters {
        println!(
            "\nAttributes: price={}, category={}",
            attrs.price, attrs.category
        );
        for (cluster_id, local_ids) in clusters {
            println!("  cluster={} -> local_ids={:?}", cluster_id, local_ids);
        }
    }

    println!("\n=== CLUSTER-LEVEL INDEX ===");
    let cluster_index = build_cluster_level_index(&attributes_to_clusters)?;
    println!("Cluster-level index created");

    println!("\n=== QUERY 1: price=1.0 AND category=book ===");
    let found_clusters =
        query_cluster_level_index(&cluster_index, 1.0, "book")?;
    println!(
        "Found {} matching cluster(s): {:?}",
        found_clusters.len(),
        found_clusters
    );
    assert_eq!(found_clusters, vec![0], "Should find cluster 0");

    println!("\n=== QUERY 2: price=2.0 AND category=software ===");
    let found_clusters =
        query_cluster_level_index(&cluster_index, 2.0, "software")?;
    println!(
        "Found {} matching cluster(s): {:?}",
        found_clusters.len(),
        found_clusters
    );
    assert_eq!(found_clusters, vec![1], "Should find cluster 1");

    println!(
        "\n=== QUERY 3: Non-existent attributes (price=3.0, category=electronics) ==="
    );
    let found_clusters =
        query_cluster_level_index(&cluster_index, 3.0, "electronics")?;
    println!("Found {} matching cluster(s)", found_clusters.len());
    assert_eq!(
        found_clusters.len(),
        0,
        "Should find no clusters for non-existent attributes"
    );

    println!("\n=== SUCCESS ===");
    println!("Two-level attributes index working correctly!");
    println!("- Unique attribute sets map to specific clusters");
    println!("- Queries efficiently identify which clusters to search");
    println!("- No false positives at cluster level");

    println!("\n=== PER-CLUSTER INDEXES ===");
    let per_cluster_indexes = build_per_cluster_index(&products, n_clusters)?;
    println!("Created {} per-cluster indexes", per_cluster_indexes.len());

    for (cluster_id, cluster_index) in per_cluster_indexes.iter().enumerate() {
        let reader = cluster_index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        let searcher = reader.searcher();
        println!(
            "  Cluster {} index: {} documents",
            cluster_id,
            searcher.num_docs()
        );
    }

    println!("\n=== QUERY CLUSTER 0: price=1.0 AND category=book ===");
    let found_local_ids =
        query_per_cluster_index(&per_cluster_indexes[0], 1.0, "book")?;
    println!(
        "Found {} matching vectors: {:?}",
        found_local_ids.len(),
        found_local_ids
    );
    assert_eq!(
        found_local_ids,
        vec![0, 1],
        "Should find local_ids [0, 1] in cluster 0"
    );

    println!("\n=== QUERY CLUSTER 1: price=2.0 AND category=software ===");
    let found_local_ids =
        query_per_cluster_index(&per_cluster_indexes[1], 2.0, "software")?;
    println!(
        "Found {} matching vectors: {:?}",
        found_local_ids.len(),
        found_local_ids
    );
    assert_eq!(
        found_local_ids,
        vec![0],
        "Should find local_id [0] in cluster 1"
    );

    println!("\n=== COMPLETE TWO-LEVEL INDEX SYSTEM ===");
    println!("Cluster-level index: Identifies which clusters to search");
    println!("Per-cluster indexes: Find exact local_ids within each cluster");
    println!("This enables efficient attribute filtering on vector search!");

    Ok(())
}
