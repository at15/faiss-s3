# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.17.0",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import json
    return (json,)


@app.cell
def _(json):
    review_file = "Software.jsonl"
    with open(review_file, 'r') as fp:
        i = 0
        for line in fp:
            print(json.loads(line.strip()))
            i += 1
            if i > 10:
                break
    return (review_file,)


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(pl, review_file):
    df = pl.read_ndjson(review_file) # well ... does not work
    return


@app.cell
def _(json):
    def read_meta():
        meta_file = "meta_Software.jsonl"
        with open(meta_file, 'r') as fp:
            i = 0
            for line in fp:
                print(json.loads(line.strip()))
                i += 1
                if i > 10:
                    break
    return (read_meta,)


@app.cell
def _(read_meta):
    read_meta()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
