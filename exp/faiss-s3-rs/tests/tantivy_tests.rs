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

#[test]
fn test_tantivy_attributes_to_cluster() -> tantivy::Result<()> {
    println!("=== TWO-LEVEL ATTRIBUTES INDEX EXAMPLE ===\n");

    // Build two level index like turbopuffer does for filter by attributes
    // and map back to the vector index's cluster and local id
    //
    // Design: Unique attribute sets map to clusters, which map to local IDs
    // This enables efficient filtering: query attributes -> clusters to search -> local IDs

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

    // Step 1: Build nested data structure
    // HashMap<ProductAttributes, HashMap<cluster_id, Vec<local_ids>>>
    let mut attributes_to_clusters: HashMap<
        ProductAttributes,
        HashMap<u64, Vec<u64>>,
    > = HashMap::new();

    for product in &products {
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

    // Step 2: Create Tantivy index for the attribute sets
    println!("\n=== CREATING TANTIVY INDEX ===");

    let mut schema_builder = Schema::builder();
    let price_field = schema_builder.add_f64_field("price", INDEXED | STORED);
    let category_field =
        schema_builder.add_text_field("category", STRING | STORED);
    let cluster_field =
        schema_builder.add_u64_field("cluster", INDEXED | STORED);
    // Only store the vector count instead of the local ids to reduce the index size
    let vector_count_field =
        schema_builder.add_u64_field("vector_count", STORED);

    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut index_writer: IndexWriter = index.writer(50_000_000)?;

    // Index each unique attribute set with its cluster mappings
    for (attrs, clusters) in &attributes_to_clusters {
        for (cluster_id, local_ids) in clusters {
            // NOTE: We don't store local id to reduce the index size
            // let local_ids_json = format!("{:?}", local_ids); // Simple format: [0, 1]

            index_writer.add_document(doc!(
                price_field => attrs.price.0,
                category_field => attrs.category.as_str(),
                cluster_field => *cluster_id,
                vector_count_field => local_ids.len() as u64,
            ))?;

            println!(
                "Indexed: price={}, category={}, cluster={}, vector_count={}",
                attrs.price,
                attrs.category,
                cluster_id,
                local_ids.len()
            );
        }
    }
    // TODO: We still tantivy index for each cluster, though this one we does not need to group by attributes set anymore

    index_writer.commit()?;

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();

    println!("\nTotal documents indexed: {}", searcher.num_docs());

    // Step 3: Query demonstration - find clusters by attributes
    println!("\n=== QUERY 1: price=1.0 AND category=book ===");

    let price_query: Box<dyn tantivy::query::Query> =
        Box::new(RangeQuery::new(
            std::ops::Bound::Included(Term::from_field_f64(price_field, 1.0)),
            std::ops::Bound::Included(Term::from_field_f64(price_field, 1.0)),
        ));

    let category_parser = QueryParser::for_index(&index, vec![category_field]);
    let category_query = category_parser.parse_query("book")?;

    let bool_query = BooleanQuery::from(vec![
        (Occur::Must, price_query),
        (Occur::Must, category_query),
    ]);

    let top_docs = searcher.search(&bool_query, &TopDocs::with_limit(10))?;
    println!("Found {} matching cluster(s):", top_docs.len());

    let mut found_clusters = Vec::new();
    for (_score, doc_address) in &top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(*doc_address)?;
        let cluster = retrieved_doc
            .get_first(cluster_field)
            .and_then(|v| v.as_u64())
            .unwrap();
        let vector_count = retrieved_doc
            .get_first(vector_count_field)
            .and_then(|v| v.as_u64())
            .unwrap();

        println!("  cluster={}, vector_count={}", cluster, vector_count);
        found_clusters.push(cluster);
    }

    assert_eq!(found_clusters, vec![0], "Should find cluster 0");

    println!("\n=== QUERY 2: price=2.0 AND category=software ===");

    let price_query: Box<dyn tantivy::query::Query> =
        Box::new(RangeQuery::new(
            std::ops::Bound::Included(Term::from_field_f64(price_field, 2.0)),
            std::ops::Bound::Included(Term::from_field_f64(price_field, 2.0)),
        ));

    let category_query = category_parser.parse_query("software")?;

    let bool_query = BooleanQuery::from(vec![
        (Occur::Must, price_query),
        (Occur::Must, category_query),
    ]);

    let top_docs = searcher.search(&bool_query, &TopDocs::with_limit(10))?;
    println!("Found {} matching cluster(s):", top_docs.len());

    let mut found_clusters = Vec::new();
    for (_score, doc_address) in &top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(*doc_address)?;
        let cluster = retrieved_doc
            .get_first(cluster_field)
            .and_then(|v| v.as_u64())
            .unwrap();
        let vector_count = retrieved_doc
            .get_first(vector_count_field)
            .and_then(|v| v.as_u64())
            .unwrap();

        println!("  cluster={}, vector_count={}", cluster, vector_count);
        found_clusters.push(cluster);
    }

    assert_eq!(found_clusters, vec![1], "Should find cluster 1");

    println!(
        "\n=== QUERY 3: Non-existent attributes (price=3.0, category=electronics) ==="
    );

    let price_query: Box<dyn tantivy::query::Query> =
        Box::new(RangeQuery::new(
            std::ops::Bound::Included(Term::from_field_f64(price_field, 3.0)),
            std::ops::Bound::Included(Term::from_field_f64(price_field, 3.0)),
        ));

    let category_query = category_parser.parse_query("electronics")?;

    let bool_query = BooleanQuery::from(vec![
        (Occur::Must, price_query),
        (Occur::Must, category_query),
    ]);

    let top_docs = searcher.search(&bool_query, &TopDocs::with_limit(10))?;
    println!("Found {} matching cluster(s)", top_docs.len());

    assert_eq!(
        top_docs.len(),
        0,
        "Should find no clusters for non-existent attributes"
    );

    println!("\n=== SUCCESS ===");
    println!("Two-level attributes index working correctly!");
    println!("- Unique attribute sets map to specific clusters");
    println!("- Queries efficiently identify which clusters to search");
    println!("- No false positives at cluster level");

    Ok(())
}
