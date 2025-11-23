use anyhow::Result;
use std::time::Instant;

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

#[test]
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
