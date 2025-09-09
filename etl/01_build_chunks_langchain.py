# etl/01_build_chunks_langchain.py
# this script reads a unified csv and creates data/chunks.jsonl with order and review chunks

import os  # for making folders
import json  # for writing json lines
import pandas as pd  # for reading csv and handling tables
from langchain.text_splitter import RecursiveCharacterTextSplitter  # for splitting long text


def safe_int(x):
    # this function converts a value to int or returns None if it cannot
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def safe_float(x):
    # this function converts a value to float or returns None if it cannot
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def safe_str(x):
    # this function converts a value to string or returns empty string if it is nan
    if pd.isna(x):
        return ""
    return str(x)


def make_order_summary_text(row):
    # this function builds a small summary string for an order row
    order_id = safe_str(row.get("order_id"))  # stores order id as string
    first = safe_str(row.get("first_name")).strip()  # stores first name
    last = safe_str(row.get("last_name")).strip()  # stores last name
    email = safe_str(row.get("customer_email")).lower()  # stores email in lowercase
    status = safe_str(row.get("order_status"))  # stores order status text
    purchase = safe_str(row.get("purchase_date"))  # stores purchase date text
    num_items = safe_int(row.get("num_items"))  # stores number of items as int
    total_payment = safe_float(row.get("total_payment"))  # stores total payment as float
    pay_type = safe_str(row.get("payment_type"))  # stores payment method text
    installments = safe_int(row.get("installments"))  # stores installment count as int
    delivery_days = safe_int(row.get("delivery_time_days"))  # stores delivery time days as int
    review_score = safe_int(row.get("review_score"))  # stores review score as int

    # builds the readable sentence with simple fields
    text = (
        f"Order {order_id} for {first} {last} ({email}). "
        f"Status: {status}. Purchase: {purchase}. "
        f"Items: {num_items if num_items is not None else 'N/A'}. "
        f"Paid: {total_payment if total_payment is not None else 'N/A'} via {pay_type} "
        f"in {installments if installments is not None else 1} installments. "
        f"Delivery time: {delivery_days if delivery_days is not None else 'N/A'} days. "
        f"Review score: {review_score if review_score is not None else 'N/A'}."
    )

    return text  # returns the summary string


def split_review_text(review_text, splitter):
    # this function splits a long review string into smaller parts
    txt = (review_text or "").strip()  # stores cleaned review text
    if txt == "":
        return []  # returns empty list if no review
    parts = splitter.split_text(txt)  # uses langchain to split the text
    return parts  # returns a list of chunk strings


def build_chunks(
    unified_csv_path="data/olist_cleaned.csv",  # path to input csv
    chunks_out_path="data/chunks.jsonl",  # path to output jsonl
    review_chunk_size=500,  # size of review sub-chunks in characters
    review_chunk_overlap=80  # overlap of review sub-chunks in characters
):
    # this function reads the csv and writes the chunks jsonl

    os.makedirs(os.path.dirname(chunks_out_path), exist_ok=True)  # makes output folder

    try:
        df = pd.read_csv(unified_csv_path)  # reads the csv into a dataframe
    except FileNotFoundError:
        print("file not found:", unified_csv_path)  # prints message if file is missing
        return
    except Exception as e:
        print("failed to read csv:", e)  # prints other errors
        return

    expected_cols = [
        "order_id",
        "customer_unique_id",
        "first_name",
        "last_name",
        "customer_email",
        "order_status",
        "purchase_date",
        "delivery_time_days",
        "num_items",
        "total_payment",
        "payment_type",
        "installments",
        "review_score",
        "review_comment_message",
    ]  # list of required columns for building chunks

    missing = [c for c in expected_cols if c not in df.columns]  # finds any missing columns
    if missing:
        print("missing columns in csv:", missing)  # prints which columns are missing
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=review_chunk_size,  # sets the chunk size for reviews
        chunk_overlap=review_chunk_overlap  # sets the overlap for reviews
    )  # creates the langchain splitter

    total_rows = len(df)  # stores how many rows are in the csv

    with open(chunks_out_path, "w", encoding="utf-8") as fout:
        # loops through each row to create chunks
        for i, row in df.iterrows():
            order_id = safe_str(row.get("order_id"))  # stores order id string
            cust_uid = safe_str(row.get("customer_unique_id"))  # stores customer unique id string

            # builds order summary chunk text
            order_text = make_order_summary_text(row)  # stores the order summary
            order_meta = {
                "chunk_id": f"order::{order_id}",  # unique id for this chunk
                "type": "order",  # marks chunk type as order
                "order_id": order_id,  # copies order id
                "customer_unique_id": cust_uid,  # copies customer unique id
                "first_name": safe_str(row.get("first_name")),  # copies first name
                "last_name": safe_str(row.get("last_name")),  # copies last name
                "customer_email": safe_str(row.get("customer_email")).lower(),  # copies email
                "order_status": safe_str(row.get("order_status")),  # copies status
                "purchase_date": safe_str(row.get("purchase_date")),  # copies purchase date
                "delivery_time_days": safe_int(row.get("delivery_time_days")),  # copies delivery time
                "num_items": safe_int(row.get("num_items")),  # copies number of items
                "total_payment": safe_float(row.get("total_payment")),  # copies total payment
                "payment_type": safe_str(row.get("payment_type")),  # copies payment type
                "installments": safe_int(row.get("installments")),  # copies installments
                "review_score": safe_int(row.get("review_score")),  # copies review score
            }  # builds metadata for order chunk

            # writes the order summary chunk
            order_record = {"text": order_text, "meta": order_meta}  # builds the record
            fout.write(json.dumps(order_record, ensure_ascii=False) + "\n")  # writes json line

            # builds review sub-chunks using langchain splitter
            review_text = safe_str(row.get("review_comment_message"))  # stores the review text
            review_parts = split_review_text(review_text, splitter)  # splits review text into parts

            # writes each review part as its own chunk
            for j, part in enumerate(review_parts):
                review_meta = {
                    "chunk_id": f"review::{order_id}::{j}",  # unique id for review sub-chunk
                    "type": "review",  # marks chunk type as review
                    "order_id": order_id,  # copies order id for mapping
                    "customer_unique_id": cust_uid,  # copies customer id for mapping
                    "purchase_date": safe_str(row.get("purchase_date")),  # copies purchase date
                    "review_score": safe_int(row.get("review_score")),  # copies review score
                }  # builds metadata for review sub-chunk

                review_record = {"text": part, "meta": review_meta}  # builds the record
                fout.write(json.dumps(review_record, ensure_ascii=False) + "\n")  # writes json line

            # prints simple progress every few thousand rows
            if (i + 1) % 5000 == 0 or (i + 1) == total_rows:
                print("processed rows:", i + 1, "/", total_rows)  # prints progress

    print("done. wrote:", chunks_out_path)  # prints final success message


if __name__ == "__main__":
    # this is the main entry point for running the script
    build_chunks()  # calls the function with default arguments
