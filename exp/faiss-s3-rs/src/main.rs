use object_store::aws::AmazonS3Builder;
use object_store::{ObjectStore, path::Path};
use std::sync::Arc;

fn test_ivf() {
    faiss_s3_rs::create_example_ivf_index("example.ivf");
    let offset = faiss_s3_rs::get_cluster_data_offset("example.ivf");
    // 52139, matches the python output from tests/test_meta.py
    println!("Cluster data offset: {:?}", offset);
}

async fn test_object_store() -> Result<(), Box<dyn std::error::Error>> {
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

    test_ivf();

    if let Err(e) = test_object_store().await {
        eprintln!("Error in test_object_store: {}", e);
    }
}
