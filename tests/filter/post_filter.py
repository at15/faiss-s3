# Show why post filter after vector search is not effective
# The similar vector may not match the attributes filter and we got nothing after filtering

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# 384 dimensions for all-MiniLM-L6-v2
embedding_size = 384
model_name = "all-MiniLM-L6-v2"

data = [
    {
        "id": 1,
        "desc": "Incredible forehand winners montage",
        "player": "federer",
        "tournament": "wimbledon",
        "year": 2017,
    },
    {
        "id": 2,
        "desc": "Best forehand shots compilation",
        "player": "nadal",
        "tournament": "french open",
        "year": 2018,
    },
    {
        "id": 3,
        "desc": "Powerful forehand rally",
        "player": "djokovic",
        "tournament": "australian open",
        "year": 2019,
    },
    {
        "id": 4,
        "desc": "Amazing cross-court forehands",
        "player": "federer",
        "tournament": "us open",
        "year": 2015,
    },
    {
        "id": 5,
        "desc": "Forehand winners highlight reel",
        "player": "thiem",
        "tournament": "french open",
        "year": 2019,
    },
    {
        "id": 6,
        "desc": "Best forehand angles ever",
        "player": "federer",
        "tournament": "wimbledon",
        "year": 2012,
    },
    {
        "id": 7,
        "desc": "Incredible forehand down-the-line",
        "player": "nadal",
        "tournament": "french open",
        "year": 2020,
    },
    {
        "id": 8,
        "desc": "Forehand winner compilation",
        "player": "alcaraz",
        "tournament": "us open",
        "year": 2022,
    },
    {
        "id": 9,
        "desc": "Amazing forehand winners",
        "player": "federer",
        "tournament": "wimbledon",
        "year": 2009,
    },
    {
        "id": 10,
        "desc": "Powerful baseline forehands",
        "player": "sinner",
        "tournament": "australian open",
        "year": 2024,
    },
]


def create_embeddings(model):
    """Create embeddings for all video descriptions using SentenceTransformer

    Args:
        model: Pre-loaded SentenceTransformer model
    """
    # Extract descriptions from data
    descriptions = [d["desc"] for d in data]

    print(f"Encoding {len(descriptions)} descriptions...")
    embeddings = model.encode(
        descriptions, show_progress_bar=True, convert_to_numpy=True
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def create_index(embeddings):
    """Create Faiss IVF index with the embeddings"""
    # For small dataset, use fewer clusters
    n_clusters = 4

    print(f"Creating IVF index with {n_clusters} clusters...")
    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFFlat(
        quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT
    )

    # Number of clusters to explore at search time
    index.nprobe = 2

    # Normalize embeddings so that dot product equals cosine similarity
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]

    print("Training index...")
    index.train(normalized_embeddings)

    print("Adding embeddings to index...")
    index.add(normalized_embeddings)

    print(f"Index created with {index.ntotal} vectors")
    return index


def search_with_post_filter(model, index, query, top_k=10, filter_fn=None):
    """
    Search the index and apply post-filtering to results

    Args:
        model: Pre-loaded SentenceTransformer model
        index: Faiss index
        query: Query text
        top_k: Number of results to return after filtering
        filter_fn: Optional function that takes a data dict and returns True to keep it
    """
    # Encode query
    query_embedding = model.encode(query, convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = np.expand_dims(query_embedding, axis=0)

    # NOTE: Search - get more results than needed to account for filtering
    search_k = top_k * 3 if filter_fn else top_k
    distances, indices = index.search(query_embedding, search_k)

    # Build all vector search results first
    all_results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx == -1:  # Faiss returns -1 for missing results
            continue

        video = data[idx]
        all_results.append(
            {
                "id": video["id"],
                "desc": video["desc"],
                "player": video["player"],
                "tournament": video["tournament"],
                "year": video["year"],
                "score": float(score),
            }
        )

    # Print raw vector search results
    print("\n  Vector search results (before attribute filter):")
    if len(all_results) == 0:
        print("  No results from vector search")
    else:
        for i, result in enumerate(all_results[:search_k], 1):
            print(f"  {i}. [{result['score']:.3f}] {result['desc']}")
            print(
                f"     Player: {result['player']}, Tournament: {result['tournament']}, Year: {result['year']}"
            )

    # Apply filter if provided
    if filter_fn:
        filtered_results = []
        for result in all_results:
            if filter_fn(result):
                filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break

        print("\n  Results after applying attribute filter:")
        if len(filtered_results) == 0:
            print("  ❌ No results match the filter")
        else:
            for i, result in enumerate(filtered_results, 1):
                print(f"  {i}. [{result['score']:.3f}] {result['desc']}")
                print(
                    f"     Player: {result['player']}, Tournament: {result['tournament']}, Year: {result['year']}"
                )


def main():
    print("=" * 80)
    print("Post-Filter Example: Tennis Video Search")
    print("=" * 80)

    # Load model once
    print(f"\nLoading model: {model_name}...")
    model = SentenceTransformer(model_name)

    # Create embeddings and index
    embeddings = create_embeddings(model)
    index = create_index(embeddings)

    # Example 1: Search without filter
    print("\n" + "=" * 80)
    print("Search 1: 'amazing forehand winners' (no filter)")
    print("=" * 80)
    search_with_post_filter(model, index, "amazing forehand winners", top_k=1)

    # Example 2: Filter by player, expect no results because nadal is not in top 1 based on vector search
    print("\n" + "=" * 80)
    print("Search 2: 'amazing forehand winners' (filter: player == 'nadal')")
    print("=" * 80)
    search_with_post_filter(
        model,
        index,
        "amazing forehand winners",
        top_k=1,
        filter_fn=lambda v: v["player"] == "nadal",
    )
    print(
        "\n💡 Notice: The top vector search result is Federer, but we filtered for Nadal."
    )
    print("   This demonstrates why post-filtering is ineffective - relevant results")
    print(
        "   matching the filter may exist but aren't in the top-k vector search results."
    )


if __name__ == "__main__":
    main()
