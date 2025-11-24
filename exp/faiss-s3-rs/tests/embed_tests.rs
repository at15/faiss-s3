use anyhow::Result;
use fastembed::{
    EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions,
    InitOptions, TextEmbedding,
};
use std::path::{Path, PathBuf};
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

#[test]
fn test_fastembed_text() -> Result<()> {
    // Must be mut because embed takes mutable reference
    // TODO: Why embed takes mutable reference?
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(true),
    )?;

    let documents = vec![
        "passage: Hello, World!",
        "query: Hello, World!",
        "passage: This is an example passage.",
        // You can leave out the prefix but it's recommended
        "fastembed-rs is licensed under Apache  2.0",
    ];

    // Generate embeddings with the default batch size, 256
    let embeddings = model.embed(documents, None)?;

    println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 4
    println!("Embedding dimension: {}", embeddings[0].len());
    Ok(())
}

#[test]
fn test_fastembed_image() -> Result<()> {
    // With custom options
    let mut model = ImageEmbedding::try_new(
        ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32)
            .with_show_download_progress(true),
    )?;
    let base_path = Path::new("dataset/caltech-101/101_ObjectCategories");
    let categories = vec!["airplanes", "camera", "panda", "umbrella"];

    let mut all_images = Vec::new();

    for category in &categories {
        let category_path = base_path.join(category);
        let category_images: Vec<PathBuf> = std::fs::read_dir(category_path)?
            .take(10)
            .map(|entry| entry.unwrap().path())
            .collect();
        all_images.extend(category_images);
    }

    let images = all_images.iter().collect::<Vec<&PathBuf>>();

    // Generate embeddings with the default batch size, 256
    let embeddings = model.embed(images, None)?;

    println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 2
    println!("Embedding dimension: {}", embeddings[0].len()); // -> Embedding dimension: 512
    Ok(())
}
