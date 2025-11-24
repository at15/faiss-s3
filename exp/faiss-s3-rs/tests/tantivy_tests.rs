use anyhow::Result;
use serde::Deserialize;
use std::time::Instant;
use std::collections::HashMap;

use tantivy::collector::TopDocs;
use tantivy::query::{QueryParser, RangeQuery};
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
struct ProductAttributes {
    price: f64,
    category: String,
}

fn test_tantivy_attributes_to_cluster() -> tantivy::Result<()> {
    // Build two level index like turpobuffer does for filter by attributes
    // and map back to the vector index's cluster and local id

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

    // FIXME: impl
    let mut attributes_to_cluster: HashMap<ProductAttributes, Vec<u64>> = HashMap::new();
    for product in products {
        let attributes = ProductAttributes {
            price: product.price,
            category: product.category,
        };
        attributes_to_cluster.entry(attributes).or_insert_with(Vec::new).push(product.cluster);
    }

    Ok(())
}
