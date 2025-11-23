use anyhow::Result;
// use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest};
use object_store::aws::AmazonS3Builder;
use object_store::{ObjectStore, path::Path};
use serde::Deserialize;
use std::sync::Arc;
use std::time::Instant;
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

fn test_tantivy() -> tantivy::Result<()> {
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

fn test_embedding_random() -> Result<()> {
    use rand::Rng;

    let dim = 128;
    let n_vectors = 100_000;
    let n_clusters = 100;

    println!("Generating {} random {}-dimensional vectors...", n_vectors, dim);
    let start_time = Instant::now();

    let mut rng = rand::thread_rng();
    let total_elements = dim * n_vectors;
    let mut embeddings = vec![0.0f32; total_elements];
    rng.fill(&mut embeddings[..]);

    let end_time = Instant::now();
    let duration = end_time.duration_since(start_time);
    println!("Finished generating random vectors, time taken: {:?}", duration);

    println!("Creating Faiss IVF index with {} clusters...", n_clusters);
    faiss_s3_rs::create_example_ivf_index_with_data(
        "test-from-rust.ivf",
        dim,
        n_vectors,
        embeddings,
        n_clusters,
    );

    Ok(())
}

// Too slow and let's just use random vectors for now
// async fn _test_embedding() -> Result<()> {
//     // NOTE: Automatically download model from huggingface
//     // hf auth login
//     // sentence-transformers/quora-distilbert-multilingual ... not supported
//     // google/embeddinggemma-300m  ... too slow
//     let model = EmbeddingModelBuilder::new("sentence-transformers/quora-distilbert-multilingual")
//         .with_logging()
//         .with_throughput_logging()
//         .build()
//         .await?;

//     let dim = 768;
//     let n_vectors = 5000; // TODO: 100k is too much for gemma moel on my mac
//     let n_clusters = 50;
//     let mut prompts: Vec<String> = vec![];
//     for i in 0..n_vectors {
//         prompts.push(format!("This is a test prompt {}", i));
//     }

//     println!("Generating embeddings...");
//     let start_time = Instant::now();
//     let embeddings = model
//         .generate_embeddings(
//             EmbeddingRequest::builder().add_prompts(prompts),
//         )
//         .await?;
//     let end_time = Instant::now();
//     let duration = end_time.duration_since(start_time);
//     println!("Finished generating embeddings, time taken: {:?}", duration);

//     // Flatten the embeddings
//     let embeddings = embeddings.into_iter().flatten().collect::<Vec<f32>>();

//     faiss_s3_rs::create_example_ivf_index_with_data("test-from-rust.ivf", dim, n_vectors, embeddings, n_clusters);

//     Ok(())
// }

fn test_ivf_local_file() {
    // faiss_s3_rs::create_example_ivf_index("example.ivf");
    // let offset = faiss_s3_rs::get_cluster_data_offset("example.ivf");
    // 52139, matches the python output from tests/test_meta.py
    // println!("Cluster data offset: {:?}", offset);
    faiss_s3_rs::search_example_ivf_index("example.ivf");
}

async fn test_ivf_s3() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing IVF on S3...");

    // Build S3 client with configuration from environment variables
    // This matches the configuration used in your C++ implementation
    let s3 = AmazonS3Builder::new()
        .with_bucket_name("test-bucket")
        .with_region("us-east-1")
        .with_endpoint(
            std::env::var("S3_ENDPOINT_URL")
                .unwrap_or_else(|_| "http://localhost:9000".to_string()),
        )
        .with_access_key_id(
            std::env::var("AWS_ACCESS_KEY_ID")
                .unwrap_or_else(|_| "test".to_string()),
        )
        .with_secret_access_key(
            std::env::var("AWS_SECRET_ACCESS_KEY")
                .unwrap_or_else(|_| "test".to_string()),
        )
        .with_allow_http(true)
        .build()?;

    let s3: Arc<dyn ObjectStore> = Arc::new(s3);

    // Test reading the existing example.ivf file
    let path = Path::from("example.ivf");
    let cluster_data_offset = 52139; // TODO: hard coded for now

    // Fetch index metadata from S3 (bytes 0 to cluster_data_offset)
    // This downloads the index structure without the actual cluster data
    println!(
        "Fetching index metadata from S3 (0..{} bytes)",
        cluster_data_offset
    );
    let get_result = s3.get_range(&path, 0..cluster_data_offset).await?;
    let index_without_cluster_data = get_result.to_vec();
    println!(
        "Downloaded {} bytes of index metadata",
        index_without_cluster_data.len()
    );

    // Create Faiss index from the downloaded metadata
    // This will create an index with S3ReadNothingInvertedLists placeholder
    let index =
        faiss_s3_rs::create_faiss_ivf_index_s3(index_without_cluster_data)?;
    println!("Successfully created Faiss IVF index from S3");

    let cluster_sizes = index.ClusterSizes();
    for (i, size) in cluster_sizes.iter().enumerate() {
        println!("Cluster {}: size = {}", i, size);
    }

    // TODO: search with actual inverted lists

    Ok(())
}

async fn _test_object_store() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing object_store with S3...");

    // Build S3 client with configuration from environment variables
    // This matches the configuration used in your C++ implementation
    let s3 = AmazonS3Builder::new()
        .with_bucket_name("test-bucket")
        .with_region("us-east-1")
        .with_endpoint(
            std::env::var("S3_ENDPOINT_URL")
                .unwrap_or_else(|_| "http://localhost:9000".to_string()),
        )
        .with_access_key_id(
            std::env::var("AWS_ACCESS_KEY_ID")
                .unwrap_or_else(|_| "test".to_string()),
        )
        .with_secret_access_key(
            std::env::var("AWS_SECRET_ACCESS_KEY")
                .unwrap_or_else(|_| "test".to_string()),
        )
        .with_allow_http(true)
        .build()?;

    let s3: Arc<dyn ObjectStore> = Arc::new(s3);

    // Test reading the existing test.txt file
    let path = Path::from("quora-20251101-202217/meta.json");

    println!("Reading s3://test-bucket/{}", path);

    match s3.get(&path).await {
        Ok(result) => {
            // Print metadata first before consuming result
            println!("Metadata:");
            println!("  Size: {} bytes", result.meta.size);
            // TODO: It is always 1970 when using gos3mock ... i.e. 0 unix epoch
            println!("  Last Modified: {:?}", result.meta.last_modified);
            if let Some(e_tag) = &result.meta.e_tag {
                println!("  ETag: {}", e_tag);
            }

            let bytes = result.bytes().await?;
            println!("Successfully read {} bytes from S3", bytes.len());

            // Try to convert to string if it's text content
            match String::from_utf8(bytes.to_vec()) {
                Ok(text) => {
                    println!("Content: {}", text);
                }
                Err(_) => {
                    println!("Content is binary data");
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to read from S3: {}", e);
            return Err(Box::new(e));
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    println!("Hello, world!");

    test_ivf_local_file();

    // if let Err(e) = test_object_store().await {
    //     eprintln!("Error in test_object_store: {}", e);
    // }

    // if let Err(e) = test_ivf_s3().await {
    //     eprintln!("Error in test_ivf_s3: {}", e);
    // }

    // if let Err(e) = test_embedding_random() {
    //     eprintln!("Error in test_embedding_random: {}", e);
    // }

    // if let Err(e) = test_embedding().await {
    //     eprintln!("Error in test_embedding: {}", e);
    // }

    // if let Err(e) = test_tantivy() {
    //     eprintln!("Error in test_tantivy: {}", e);
    // }
}
