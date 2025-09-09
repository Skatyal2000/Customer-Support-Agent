# etl/04_build_kb_chunks.py
# this script reads files from kb/ and writes kb/kb_chunks.jsonl for embeddings

import os  # for paths and folder stuff
import json  # for writing json lines
import re  # for cleaning titles
from pathlib import Path  # for file paths
from typing import List  # for type hints
from langchain.text_splitter import RecursiveCharacterTextSplitter  # for splitting text

# optional pdf support
try:
    import PyPDF2  # for reading pdf pages
    HAS_PDF = True  # flag when pdf lib is present
except Exception:
    HAS_PDF = False  # flag false if import failed


def read_text_file(p: Path) -> str:
    # this function reads a .txt or .md file as utf-8
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(p: Path) -> List[str]:
    # this function reads a pdf and returns a list of page texts
    pages = []  # list to store page texts
    if not HAS_PDF:
        return pages  # return empty if PyPDF2 is missing
    try:
        with p.open("rb") as f:
            reader = PyPDF2.PdfReader(f)  # load pdf
            for i in range(len(reader.pages)):
                pages.append(reader.pages[i].extract_text() or "")  # extract page text
    except Exception:
        pass  # ignore pdf errors and keep empty
    return pages


def clean_title(name: str) -> str:
    # this function builds a simple title from a filename
    t = re.sub(r"[_\-]+", " ", name)  # replace underscores and dashes
    t = re.sub(r"\.[^.]+$", "", t)  # remove extension
    return t.strip().title()  # title case


def build_kb_chunks(src_dir="kb", out_path="kb/kb_chunks.jsonl",
                    chunk_size=800, chunk_overlap=120):
    # this function walks the kb folder, splits text, and writes jsonl

    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # make kb folder if missing
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # max chars per chunk
        chunk_overlap=chunk_overlap  # overlap between chunks
    )  # create text splitter

    p = Path(src_dir)  # path to kb folder
    files = sorted([x for x in p.iterdir() if x.is_file()])  # list files
    if not files:
        print("no files found in", src_dir)  # print if empty
        return

    with open(out_path, "w", encoding="utf-8") as fout:
        chunk_id = 0  # counter for chunk ids
        for fpath in files:
            fname = fpath.name  # filename string
            title = clean_title(fname)  # friendly title
            suffix = fpath.suffix.lower()  # file extension

            if suffix in {".txt", ".md"}:
                text = read_text_file(fpath)  # read text
                parts = splitter.split_text(text)  # split into parts
                for j, part in enumerate(parts):
                    meta = {
                        "chunk_id": f"kb::{chunk_id}",  # global chunk id
                        "type": "kb",  # marks as kb chunk
                        "source": fname,  # original file name
                        "title": title,  # human title
                        "page": None  # no page for text files
                    }  # metadata dict
                    rec = {"text": part, "meta": meta}  # build record
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")  # write json line
                    chunk_id += 1  # increment id

            elif suffix == ".pdf":
                pages = read_pdf_file(fpath)  # list of page texts
                if not pages:
                    print("skip pdf (no text or PyPDF2 missing):", fname)  # info print
                    continue
                for pi, page_text in enumerate(pages):
                    parts = splitter.split_text(page_text or "")  # split page text
                    for j, part in enumerate(parts):
                        meta = {
                            "chunk_id": f"kb::{chunk_id}",  # global chunk id
                            "type": "kb",  # marks as kb chunk
                            "source": fname,  # original pdf file
                            "title": title,  # human title
                            "page": pi + 1  # 1-based page number
                        }  # metadata dict
                        rec = {"text": part, "meta": meta}  # build record
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")  # write json line
                        chunk_id += 1  # increment id
            else:
                print("skip unsupported file:", fname)  # info print

    print("done. wrote:", out_path)  # success print


if __name__ == "__main__":
    # this is the main entry point
    build_kb_chunks()  # call function with defaults
