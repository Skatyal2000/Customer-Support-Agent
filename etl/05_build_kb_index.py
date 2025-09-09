# etl/05_build_kb_index.py
# this script reads kb/kb_chunks.jsonl, builds embeddings, and writes a faiss index and meta

import os  # for folders
import json  # for reading json lines
import math  # for batch math
import numpy as np  # for arrays
import pandas as pd  # for tables
import faiss  # for vector index
from sentence_transformers import SentenceTransformer  # for embeddings


def load_kb(jsonl_path):
    # this function loads all kb records and keeps text in meta
    texts = []  # list of chunk text
    metas = []  # list of metadata (including text copy)
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue  # skip blanks
                rec = json.loads(line)  # parse json
                txt = rec.get("text", "")  # chunk text
                meta = rec.get("meta", {})  # metadata dict
                meta = dict(meta)  # shallow copy
                meta["text"] = txt  # store text in meta for kb answers
                texts.append(txt)  # add to text list
                metas.append(meta)  # add to meta list
    except FileNotFoundError:
        print("file not found:", jsonl_path)  # print if missing
        return [], []
    except Exception as e:
        print("failed to read jsonl:", e)  # print error
        return [], []
    return texts, metas  # return lists


def normalize_rows(X):
    # this function normalizes rows to unit length
    n = np.linalg.norm(X, axis=1, keepdims=True)  # compute norms
    n[n == 0] = 1.0  # avoid div by zero
    return X / n  # return normalized matrix


def embed_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=512):
    # this function encodes a list of texts using sentence-transformers
    model = SentenceTransformer(model_name)  # load model
    all_vecs = []  # list of batch arrays
    n = len(texts)  # number of texts
    if n == 0:
        return np.zeros((0, 384), dtype="float32")  # empty matrix
    steps = math.ceil(n / batch_size)  # number of batches
    for i in range(steps):
        a = i * batch_size  # start index
        b = min((i + 1) * batch_size, n)  # end index
        vecs = model.encode(texts[a:b], convert_to_numpy=True, show_progress_bar=False)  # encode
        all_vecs.append(vecs.astype("float32"))  # convert dtype
        if (i + 1) % 10 == 0 or (i + 1) == steps:
            print("embedded kb batches:", i + 1, "/", steps)  # progress
    X = np.vstack(all_vecs)  # stack arrays
    X = normalize_rows(X).astype("float32")  # normalize
    return X  # return matrix


def build_index(X):
    # this function builds a faiss inner product index
    if X.shape[0] == 0:
        return None  # no vectors
    d = X.shape[1]  # vector dim
    index = faiss.IndexFlatIP(d)  # create index
    index.add(X)  # add vectors
    return index  # return index


def save_all(index, metas, out_dir="kb_index"):
    # this function saves faiss and meta parquet
    os.makedirs(out_dir, exist_ok=True)  # make folder
    faiss_path = os.path.join(out_dir, "kb.faiss")  # index file
    meta_path = os.path.join(out_dir, "kb_meta.parquet")  # meta file
    if index is None:
        print("no index to save")  # info print
        return
    faiss.write_index(index, faiss_path)  # save index
    pd.DataFrame(metas).to_parquet(meta_path, index=False)  # save meta
    print("wrote:", faiss_path)  # print path
    print("wrote:", meta_path)  # print path


def main(kb_chunks_path="kb/kb_chunks.jsonl",
         out_dir="kb_index",
         model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # this function runs the kb indexing pipeline
    texts, metas = load_kb(kb_chunks_path)  # load chunks
    if not texts:
        print("no kb chunks found, aborting")  # guard
        return
    X = embed_texts(texts, model_name=model_name, batch_size=512)  # compute embeddings
    idx = build_index(X)  # build faiss
    save_all(idx, metas, out_dir=out_dir)  # save files
    print("done building kb index")  # final message


if __name__ == "__main__":
    # this is the main entry point
    main()  # call main
