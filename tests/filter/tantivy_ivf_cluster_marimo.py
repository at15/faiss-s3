import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import faiss
    return faiss, np


@app.cell
def _():
    from sentence_transformers import SentenceTransformer # NOTE: need to install ipython when using marimo
    return (SentenceTransformer,)


@app.cell
def _():
    # 384 dimensions for all-MiniLM-L6-v2
    embedding_size = 384
    model_name = "all-MiniLM-L6-v2"
    return embedding_size, model_name


@app.cell
def _():

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
    return (data,)


@app.cell
def _(data):
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

    return (create_embeddings,)


@app.cell
def _(embedding_size, faiss, np):
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
    return (create_index,)


@app.cell
def _(SentenceTransformer, model_name):
    model = SentenceTransformer(model_name)
    return (model,)


@app.cell
def _(create_embeddings, model):
    embeddings = create_embeddings(model)
    return (embeddings,)


@app.cell
def _(create_index, embeddings):
    index = create_index(embeddings)
    return (index,)


@app.cell
def _(faiss, index):
    if not isinstance(index, faiss.IndexIVFFlat):
        raise ValueError("must be IVFFlat")
    else:
        print("Index is IVFFlat")
    return


@app.cell
def _(index):
    # number of clusters
    nlist = index.nlist
    print(f"Total {nlist} clusters")
    return (nlist,)


@app.cell
def _(faiss, nlist, np):
    # get the vector ids of each cluster
    def get_cluster_ids(index):
        invlists = index.invlists
        cluster_ids = {} # map from cluster id to list[vector id]
        for list_no in range(nlist):
            sz = invlists.list_size(list_no)
            # print(f"Cluster {list_no} has size {sz}")
            if sz == 0:
                continue # skip empty clusters
            ids_ptr = invlists.get_ids(list_no)
            # print(ids_ptr) # <Swig Object of type 'long long *' at 0x1346d7000>
            try:
                ids_view = faiss.rev_swig_ptr(ids_ptr, sz)
                # Copy the ids
                cluster_ids[list_no] = np.array(ids_view, copy=True)
            finally:
                invlists.release_ids(list_no, ids_ptr)
        return cluster_ids
    return (get_cluster_ids,)


@app.cell
def _(get_cluster_ids, index):
    print(get_cluster_ids(index))
    return


@app.cell
def _(index, nlist, np):
    # get vector that is cloest to each cluster's centroid
    # the centroid is kmeans, NOT any vector from the dataset.
    # We can search using the centroid to get a closest dataset vector
    def get_cluster_centroids(ivf):
        cluster_centroids = {}
        d = ivf.d
        q = ivf.quantizer
        # Need a matrix to search, cannot search single vector
        buf = np.empty((1, d), dtype='float32')
        for lid in range(nlist):
            # reconstruct centroid vector for list `lid`
            q.reconstruct(lid, buf[0])
            # ask the IVF index: who is nearest to this centroid?
            D, I = index.search(buf, 1)
            rep_id = int(I[0, 0]) if I[0, 0] != -1 else -1  # -1 if list empty or no result
            # print(rep_id)
            cluster_centroids[lid] = rep_id
        return cluster_centroids
    
    return (get_cluster_centroids,)


@app.cell
def _(get_cluster_centroids, index):
    print(index.d)
    print(get_cluster_centroids(index))
    return


@app.cell
def _(data, get_cluster_centroids, get_cluster_ids):
    # Print the centroid representive and what is in each cluster
    def print_clusters(index):
        cluster_ids = get_cluster_ids(index)
        cluster_centroids = get_cluster_centroids(index)
        nlist = index.nlist
        for cluster_id in range(nlist):
            center = cluster_centroids[cluster_id]
            ids = cluster_ids[cluster_id]
            print(f"==== Cluster {cluster_id} ====")
            print(f"Center {center} {data[center]["desc"]}, {data[center]["player"]}")
            for id in ids:
                print(f"{id} {data[id]["desc"]}, {data[id]["player"]}")

    return (print_clusters,)


@app.cell
def _(index, print_clusters):
    print_clusters(index)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
