import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create Tantivy index that maps back to clusters

    Instead of post filter after searching faiss index.
    We build attributes index that maps to cluster id.
    This make sure the vectors in the clusters we search all meet the attribute requirement.
    """)
    return


@app.cell
def _():
    import tantivy
    return (tantivy,)


@app.cell
def _(data, get_cluster_ids, tantivy):
    def create_tantivy_index(vector_index):
        sb = tantivy.SchemaBuilder()
        # TODO: Not sure if we need stored=True for all fields ...
        sb.add_text_field("desc", stored=True)
        sb.add_text_field("player", stored=True)
        sb.add_integer_field("year", stored=True)
        sb.add_integer_field("cluster_id", stored=True)
        sb.add_integer_field("vector_id", stored=True)
        schema = sb.build()

        fts_index = tantivy.Index(schema)
        writer = fts_index.writer(num_threads=1)
    
        cluster_ids = get_cluster_ids(vector_index)
        nlist = vector_index.nlist
        for cluster_id in range(nlist):
            ids = cluster_ids[cluster_id]
            for id in ids:
                d = data[id]
                # FIXME: this logic is actually not right, it is indexing every document
                # We should index every unique attributes pair's cluster id instead
                # Which is much smaller
                writer.add_document(tantivy.Document.from_dict({
                    "vector_id": id,
                    "cluster_id": cluster_id,
                    "desc": d["desc"],
                    "player": d["player"],
                    "year": d["year"]
                }))

        writer.commit()
        fts_index.reload()
        return fts_index
    return (create_tantivy_index,)


@app.cell
def _(create_tantivy_index, index):
    fts_index = create_tantivy_index(index)
    return (fts_index,)


@app.function
def search_fts(index):
    search = index.searcher()
    query = index.parse_query("nadal", ["player"])
    results = search.search(query)
    if len(results.hits) == 0:
        print("not result")
        return
    for score, doc_address in results.hits:
        doc = search.doc(doc_address)
        print(doc)


@app.cell
def _(fts_index):
    search_fts(fts_index)
    return


@app.cell
def _():
    # Since we have the cluster_id to 3 ... we can just search that specific cluster
    # TODO: We have assign but we need centroid_dis
    # TODO: Why do we need centroid_dis? For IVFFlat, we can get the actual distance, or it is for IVFPQ?
    # distances, labels are for storing the returned distances and ids
    # index.search
    """
    def search(
        n, x, k, distances, labels, params=None)

    def search_preassigned(
        n, x, k, assign, centroid_dis, distances, labels,
      store_pairs, params=None, stats=None)
    """
    # index.search_preassigned
    return


@app.cell
def _():
    import inspect
    return (inspect,)


@app.cell
def _(index, inspect):
    sig = inspect.signature(index.quantizer.search)
    print(sig)
    return


@app.cell
def _(model, np):
    def embed_query(query: str):
        query_embedding = model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        # faiss requires a query matrix, not a single vector
        query_embedding = np.expand_dims(query_embedding, axis=0)
        return query_embedding
    return (embed_query,)


@app.cell
def _(data, embed_query, index, np):
    def search_cluster_no_id_selector():
        coarse_idx = np.array([[3]])
        # FIXME: I don't think whould use 0, we should calculate using actual query and centroid
        coarse_dis = np.zeros_like(coarse_idx, dtype='float32')
        q = embed_query("forehand")
        index.nprobe = 1 # change from 2 to 1 to deal with assert Iq.shape == (n, self.nprobe)
        D, I = index.search_preassigned(q, 2, coarse_idx, coarse_dis)
        print(D, I)
        # Ironically, in that cluster, the first match is sinner instead of nadal
        # TODO: Just going to the right cluster is not enough, we need to pass in id selector to get the top k that matches filter. Actually in this case, it is easier to only calculate the distance on a subset of ids ... which is what the id selector does intenally though
        for i_q in I:
            for i in i_q:
                print(data[i])
    search_cluster_no_id_selector()
    return


@app.cell
def _(data, embed_query, faiss, index, np):
    def search_cluster_id_selector():
        coarse_idx = np.array([[3]])
        coarse_dis = np.zeros_like(coarse_idx, dtype='float32')
        q = embed_query("forehand")
        index.nprobe = 1
        # 1, 6 are the only nadal items
        sel = faiss.IDSelectorBatch(np.array([1, 6], dtype='int64'))
        params = faiss.SearchParameters()
        params.sel = sel
        # FIXME: params is not supported
        # https://github.com/facebookresearch/faiss/blob/1deba7b90f21d952c86affe79721a06ec5800907/faiss/python/class_wrappers.py#L729-L732
        D, I = index.search_preassigned(q, 2, coarse_idx, coarse_dis, params=params)
        for i_q in I:
            for i in i_q:
                print(data[i])
    search_cluster_id_selector()
        
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Conclusion

    After using `search_preassigned` to limit cluster, we still need to filter within the cluster, simply picking top k base on vector similarity for all vectors inside the cluster would still lead issue of empty after post filter.

    Within a cluster, we still need the ID selector because having some
    vectors in the clsuter matching the attributes filter does NOT mean
    all the vectors within the cluster match the filter.

    For top k within a cluster, we can either

    - Have attributes index for each cluster
    - Save vector id in the fts index and use id selector with ids from fts query

    Python wrapper does not support params in search_preassigned, so we cannot use id selector from python, search does support params though ... https://github.com/facebookresearch/faiss/blob/1deba7b90f21d952c86affe79721a06ec5800907/faiss/python/class_wrappers.py#L729-L732

    ```python
    def replacement_search_preassigned(self, x, k, Iq, Dq, *, params=None, D=None, I=None)
         Iq = np.ascontiguousarray(Iq, dtype='int64')
            assert params is None, "params not supported"
            assert Iq.shape == (n, self.nprobe)
    ```

    We still need to have proper coarse_dis though, seems only IVFPQ need it https://github.com/facebookresearch/faiss/blob/eff0898a13ae4d0d7cfce61092299fea0041479a/contrib/ivf_tools.py#L50-L57

    ```python
    # the coarse distances are used in IVFPQ with L2 distance and
    # by_residual=True otherwise we provide dummy coarse_dis
    if coarse_dis is None:
        coarse_dis = np.zeros((n, index_ivf.nprobe), dtype=dis_type)
    else:
        assert coarse_dis.shape == (n, index_ivf.nprobe)

    return index_ivf.search_preassigned(xq, k, list_nos, coarse_dis)
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
