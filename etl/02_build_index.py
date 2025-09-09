# etl/02_build_index.py
# this script reads data/chunks.jsonl, builds embeddings, and writes a faiss index and metadata

import os  # for folders
import json  # for reading json lines
import math  # for simple batching math
import numpy as np  # for arrays and math
import pandas as pd  # for tables and saving parquet
import faiss  # for vector index
from sentence_transformers import SentenceTransformer  # for text embeddings


def load_chunks(jsonl_path):
    # this function loads all records from a jsonl file
    texts = []  # list for chunk text
    metas = []  # list for chunk metadata
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue  # skip empty lines
                rec = json.loads(line)  # parse one json line
                texts.append(rec.get("text", ""))  # store chunk text
                metas.append(rec.get("meta", {}))  # store chunk meta
    except FileNotFoundError:
        print("file not found:", jsonl_path)  # print missing file
        return [], []
    except Exception as e:
        print("failed to read jsonl:", e)  # print other errors
        return [], []
    return texts, metas  # return the two lists


def make_output_folders(index_dir):
    # this function ensures the index folder exists
    os.makedirs(index_dir, exist_ok=True)  # create folder if missing


def normalize_rows(X):
    # this function normalizes each vector to unit length
    # safe for inner product search to act like cosine
    norms = np.linalg.norm(X, axis=1, keepdims=True)  # compute vector norms
    norms[norms == 0] = 1.0  # avoid division by zero
    return X / norms  # return normalized vectors


def embed_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=512):
    # this function encodes a list of texts into vectors using sentence-transformers
    model = SentenceTransformer(model_name)  # load the embedding model
    all_vecs = []  # list to collect batches of vectors
    n = len(texts)  # how many texts we have
    if n == 0:
        return np.zeros((0, 384), dtype="float32")  # return empty array if no texts

    steps = math.ceil(n / batch_size)  # how many batches we need
    for i in range(steps):
        a = i * batch_size  # batch start index
        b = min((i + 1) * batch_size, n)  # batch end index
        batch = texts[a:b]  # slice the batch
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)  # encode batch
        all_vecs.append(vecs.astype("float32"))  # ensure float32 for faiss
        # simple progress print
        if (i + 1) % 10 == 0 or (i + 1) == steps:
            print("embedded batches:", i + 1, "/", steps)  # print progress
    X = np.vstack(all_vecs)  # stack all batches into one array
    X = normalize_rows(X).astype("float32")  # normalize vectors for inner product
    return X  # return the matrix of embeddings


def build_faiss_index(X):
    # this function builds an inner product faiss index for the vectors
    if X.shape[0] == 0:
        return None  # return none if there are no vectors
    d = X.shape[1]  # vector dimension
    index = faiss.IndexFlatIP(d)  # simple flat index with inner product
    index.add(X)  # add all vectors
    return index  # return the index


def save_index_and_meta(index, metas, index_dir):
    # this function saves the faiss index and the metadata sidecar
    index_path = os.path.join(index_dir, "orders.faiss")  # path for faiss file
    meta_path = os.path.join(index_dir, "orders_meta.parquet")  # path for meta parquet

    if index is None:
        print("no index to save")  # print if index missing
        return

    faiss.write_index(index, index_path)  # write the faiss index to disk
    meta_df = pd.DataFrame(metas)  # turn metadata list into dataframe
    meta_df.to_parquet(meta_path, index=False)  # save metadata to parquet
    print("wrote index:", index_path)  # print index path
    print("wrote meta:", meta_path)  # print meta path


def main(
    chunks_path="data/chunks.jsonl",  # input jsonl with chunks
    index_dir="index",  # output folder for faiss and meta
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # embedding model name
    batch_size=512  # batch size for encoding
):
    # this function runs the full pipeline for building the index

    make_output_folders(index_dir)  # ensure output folder exists

    texts, metas = load_chunks(chunks_path)  # load all chunk texts and metas
    if len(texts) == 0:
        print("no chunks found, aborting")  # print if nothing to index
        return

    X = embed_texts(texts, model_name=model_name, batch_size=batch_size)  # compute embeddings
    print("embeddings shape:", X.shape)  # print final shape

    index = build_faiss_index(X)  # build the faiss index
    save_index_and_meta(index, metas, index_dir)  # save index and meta files

    print("done building index")  # final message


if __name__ == "__main__":
    # this is the main entry point
    main()  # call the main function with default arguments
