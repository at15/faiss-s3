use anyhow::Result;
// use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest};
use object_store::aws::AmazonS3Builder;
use object_store::{ObjectStore, path::Path};
use std::sync::Arc;
use std::time::Instant;
use tantivy::collector::TopDocs;
use tantivy::query::{QueryParser, RangeQuery};
use tantivy::schema::*;
use tantivy::{doc, Index, IndexWriter, ReloadPolicy, TantivyDocument};

fn test_tantivy() -> tantivy::Result<()> {
    // 1. Define the schema with text and numeric fields
    let mut schema_builder = Schema::builder();
    
    // Text field - stored and indexed for full-text search
    let title = schema_builder.add_text_field("title", TEXT | STORED);
    let body = schema_builder.add_text_field("body", TEXT);
    
    // Numeric fields - i64, u64, f64 are supported
    let price = schema_builder.add_u64_field("price", INDEXED | STORED | FAST);
    let rating = schema_builder.add_f64_field("rating", INDEXED | STORED | FAST);
    let year = schema_builder.add_i64_field("year", INDEXED | STORED | FAST);
    
    let schema = schema_builder.build();
    
    // 2. Create the index (in-memory for this example)
    let index = Index::create_in_ram(schema.clone());
    
    // For persistent storage, use:
    // let index = Index::create_in_dir("/path/to/index", schema.clone())?;
    
    // 3. Get an IndexWriter with 50MB buffer
    let mut index_writer: IndexWriter = index.writer(50_000_000)?;
    
    // 4. Index some documents
    index_writer.add_document(doc!(
        title => "Rust Programming",
        body => "Learn Rust with practical examples",
        price => 29u64,
        rating => 4.5f64,
        year => 2023i64,
    ))?;
    
    index_writer.add_document(doc!(
        title => "Advanced Tantivy",
        body => "Deep dive into full-text search with Tantivy",
        price => 45u64,
        rating => 4.8f64,
        year => 2024i64,
    ))?;
    
    index_writer.add_document(doc!(
        title => "Python Basics",
        body => "Introduction to Python programming",
        price => 19u64,
        rating => 4.2f64,
        year => 2022i64,
    ))?;
    
    index_writer.add_document(doc!(
        title => "Rust Search Engines",
        body => "Building search engines with Rust and Tantivy",
        price => 55u64,
        rating => 4.9f64,
        year => 2024i64,
    ))?;
    
    // 5. Commit the changes
    index_writer.commit()?;
    
    // 6. Get a reader - ReloadPolicy determines when index changes are visible
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    
    let searcher = reader.searcher();
    
    println!("=== TEXT SEARCH ===");
    // 7. Text search using QueryParser
    let query_parser = QueryParser::for_index(&index, vec![title, body]);
    let query = query_parser.parse_query("Rust")?;
    
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
    
    for (_score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        println!("Found: {:?}", retrieved_doc);
    }
    
    println!("\n=== NUMERIC RANGE SEARCH ===");
    // 8. Numeric range query: price between 20 and 50
    let range_query = RangeQuery::new(
        std::ops::Bound::Included(Term::from_field_u64(price, 20u64)),
        std::ops::Bound::Included(Term::from_field_u64(price, 50u64)),
    );

    let top_docs = searcher.search(&range_query, &TopDocs::with_limit(10))?;
    println!("Items priced between 20-50:");
    for (_score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        println!("  {:?}", retrieved_doc);
    }
    
    println!("\n=== COMBINED QUERY ===");
    // 9. Combined text + numeric query
    // Find "Rust" documents with rating >= 4.5
    use tantivy::query::BooleanQuery;
    use tantivy::query::Occur;
    
    let text_query = query_parser.parse_query("Rust")?;
    let rating_query = RangeQuery::new(
        std::ops::Bound::Included(Term::from_field_f64(rating, 4.5f64)),
        std::ops::Bound::Unbounded,
    );
    
    let bool_query = BooleanQuery::from(vec![
        (Occur::Must, text_query),
        (Occur::Must, Box::new(rating_query)),
    ]);
    
    let top_docs = searcher.search(&bool_query, &TopDocs::with_limit(10))?;
    println!("Rust books with rating >= 4.5:");
    for (_score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        println!("  {:?}", retrieved_doc);
    }
    
    println!("\n=== FAST FIELD ACCESS ===");
    // 10. Fast fields for efficient numeric access/sorting
    // Fast fields are column-oriented storage optimized for accessing
    // all values of a field across documents
    if searcher.segment_readers().len() > 0 {
        let segment_reader = &searcher.segment_readers()[0];
        let price_reader = segment_reader.fast_fields().u64("price")?;
        println!("Fast field access to prices:");
        for doc_id in 0..segment_reader.max_doc() {
            let price_val = price_reader.first(doc_id).unwrap_or(0);
            println!("  Doc {}: price = {}", doc_id, price_val);
        }
    }
    
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

fn _test_ivf_local_file() {
    faiss_s3_rs::create_example_ivf_index("example.ivf");
    let offset = faiss_s3_rs::get_cluster_data_offset("example.ivf");
    // 52139, matches the python output from tests/test_meta.py
    println!("Cluster data offset: {:?}", offset);
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

    // test_ivf_local_file();

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

    if let Err(e) = test_tantivy() {
        eprintln!("Error in test_tantivy: {}", e);
    }
}
