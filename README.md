# faiss-s3

On-demand Faiss vector search from S3. Lazy-loads clusters without fetching entire index.

## TODO

- [ ] Copy code from https://github.com/at15/faiss/tree/at15/faiss-on-s3/demos/s3_ivf
  - [ ] Python client
  - [ ] Test script to generate embeddings and query from disk/memory
- [ ] Build on Ubuntu 24.04, I can try with docker, need to use it for codex envrionment as well
- [ ] Fix memory usage, lock etc. https://claude.ai/code/session_011CUak16Gb3r9A4ke9jxbUp

## Usage

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