# faiss-s3

On-demand Faiss vector search from S3. Lazy-loads clusters without fetching entire index.
Original implementation: https://github.com/at15/faiss/tree/at15/faiss-on-s3/demos/s3_ivf

## TODO

- [ ] e2e test script in python

## Usage

- [ ] Add how to build server

```bash
python3.13 -m venv .venv
source .venv/bin/activate
# Brings in boto3 and faiss-cpu
pip install -e ./client
# Generate embeddings
pip install sentence-transformers pandas
```

## License

MIT