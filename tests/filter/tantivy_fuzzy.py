import tantivy


def create_index():
    """Create and populate a Tantivy index with sample documents"""
    # Define schema
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("title", stored=True)
    schema_builder.add_text_field("body", stored=True)
    schema = schema_builder.build()

    # Create an in-memory index
    index = tantivy.Index(schema)

    # Get an IndexWriter
    writer = index.writer(num_threads=1, heap_size=128_000_000)

    # Add documents
    documents = [
        {"title": "The Old Man and the Sea", "body": "A story about an old fisherman."},
        {"title": "Of Mice and Men", "body": "A novel about friendship and dreams."},
        {"title": "The Sea Wolf", "body": "Adventure on the sea."},
        {
            "title": "Lord of the Rings",
            "body": "A fantasy novel about a journey to a magical land.",
        },
    ]
    for doc in documents:
        writer.add_document(tantivy.Document(**doc))

    # Commit the index
    writer.commit()

    # Reload the index and create a searcher
    index.reload()

    return index


def test_tantivy():
    """Test both regular and fuzzy search to demonstrate typo tolerance"""
    index = create_index()
    searcher = index.searcher()

    # Test 1: Regular search (no typo tolerance)
    print("=== Regular Search ===")
    print("Query: 'loard' (typo)")
    query = index.parse_query("loard", ["title", "body"])
    results = searcher.search(query, limit=3)

    if len(results.hits) > 0:
        for score, doc_address in results.hits:
            doc = searcher.doc(doc_address)
            print(f"  score={score:.2f}, title={doc['title']}")
    else:
        print("  No results found")

    # Test 2: Fuzzy search (typo tolerance enabled)
    print("\n=== Fuzzy Search ===")
    print("Query: 'loard' (typo) with fuzzy matching")

    # fuzzy_fields: (prefix, distance, transpose_cost_one)
    # distance=2 means allow up to 2 character edits (Levenshtein distance)
    query = index.parse_query(
        "loard",
        ["title", "body"],
        fuzzy_fields={"title": (False, 2, True), "body": (False, 2, True)},
    )
    results = searcher.search(query, limit=3)

    if len(results.hits) > 0:
        print(f"  Found {len(results.hits)} results:")
        for score, doc_address in results.hits:
            doc = searcher.doc(doc_address)
            print(f"  score={score:.2f}, title={doc['title']}")
    else:
        print("  No results found")


if __name__ == "__main__":
    test_tantivy()
