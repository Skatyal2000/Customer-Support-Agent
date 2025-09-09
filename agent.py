# agent/tools.py
# this script has helper tools for retrieval and simple actions

import os  # for file paths
import numpy as np  # for arrays
import pandas as pd  # for loading parquet
import faiss  # for vector index
from sentence_transformers import SentenceTransformer  # for embeddings
from rapidfuzz import process, fuzz  # for fuzzy matching
import smtplib  # for email
from email.mime.text import MIMEText  # email body
import json  # for writing json lines
from datetime import timedelta 
import requests


# load faiss index and metadata when the file is imported
INDEX_PATH = "index/orders.faiss"  # path to faiss index file
META_PATH = "index/orders_meta.parquet"  # path to meta parquet file
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # embedding model
KB_INDEX_PATH = "kb_index/kb.faiss"  # path to kb faiss file
KB_META_PATH = "kb_index/kb_meta.parquet"  # path to kb meta file
ORDERS_CSV = os.getenv("ORDERS_CSV_PATH", "data/olist_cleaned.csv")  # csv path
_ORDERS_DF = None  # cache for the dataframe

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")  # slack incoming webhook url
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "")  # destination email for escalations
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

FAISS = None  # variable to hold the faiss index
META = None  # variable to hold metadata dataframe
MODEL = None  # variable to hold embedding model
KB_FAISS = None  # variable to hold kb faiss index
KB_META = None  # variable to hold kb meta dataframe

def load_resources():
    # this function loads faiss, meta, and model into memory
    global FAISS, META, MODEL  # mark as global
    if FAISS is None and os.path.exists(INDEX_PATH):
        FAISS = faiss.read_index(INDEX_PATH)  # load faiss index
    if META is None and os.path.exists(META_PATH):
        META = pd.read_parquet(META_PATH)  # load metadata as dataframe
    if MODEL is None:
        MODEL = SentenceTransformer(MODEL_NAME)  # load embedding model


def rag_search(query, k=6, filter_type=None):
    # this function does similarity search over the faiss index
    load_resources()  # make sure things are loaded
    if FAISS is None or META is None or MODEL is None:
        return []  # return empty if not ready

    vec = MODEL.encode([query], normalize_embeddings=True)  # encode the query
    vec = np.array(vec, dtype="float32")  # make sure it is float32
    D, I = FAISS.search(vec, k)  # search faiss for top k
    hits = META.iloc[I[0]].to_dict(orient="records")  # get rows from meta
    if filter_type:
        hits = [h for h in hits if h.get("type") == filter_type]  # filter by type
    return hits  # return list of dicts

def _write_jsonl(path, rec):
    # this function appends one json line to a file
    os.makedirs(os.path.dirname(path), exist_ok=True)  # make folder if missing
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")  # write one line

def _today_anchor():
    # this function returns "today" for eligibility calculations
    val = os.getenv("SIMULATED_TODAY", "").strip()  # read env override
    if val:
        try:
            return pd.to_datetime(val)  # parse date like 2018-10-17 or 2018-10-17 17:30:18
        except Exception:
            pass  # fall through if parse fails

    # optional: allow using the dataset's latest purchase_date as "today"
    use_data_max = os.getenv("USE_DATASET_MAX_AS_TODAY", "false").lower() == "true"
    if use_data_max and (META is not None) and ("purchase_date" in META.columns):
        try:
            return pd.to_datetime(META["purchase_date"]).max()
        except Exception:
            pass  # fall through to real now

    return pd.Timestamp.now(tz=None)  # default to real clock


def check_return_eligibility(facts, window_days=30):
    # this function checks if the order is eligible for return
    if not facts:
        return {"eligible": False, "reason": "no order facts"}

    status = str(facts.get("order_status", "")).lower()  # get order status

    try:
        purchase = pd.to_datetime(facts.get("purchase_date"))  # parse purchase date
    except Exception:
        purchase = None

    try:
        d_days = int(facts.get("delivery_time_days")) if facts.get("delivery_time_days") is not None else None
    except Exception:
        d_days = None

    delivered = None
    if purchase is not None and d_days is not None:
        delivered = purchase + timedelta(days=d_days)  # purchase + shipping days

    if status in {"created", "approved", "processing", "shipped"} and delivered is None:
        return {"eligible": False, "reason": "not delivered yet, cancel instead", "suggest": "cancel"}

    if delivered is not None:
        # *** use the simulated "today" here instead of real now ***
        today = _today_anchor().normalize()  # anchor date (simulated or real)
        try:
            days_since = (today - delivered.normalize()).days
        except Exception:
            days_since = (today - delivered).days  # fallback if normalize fails

        if days_since <= window_days:
            return {
                "eligible": True,
                "reason": f"within {window_days}d window",
                "delivered_date": str(delivered.date()),
                "today_anchor": str(today.date())
            }
        else:
            return {
                "eligible": False,
                "reason": f"outside {window_days}d window (delivered {delivered.date()})",
                "today_anchor": str(today.date())
            }

    return {"eligible": False, "reason": "missing delivery date info"}


def start_return(order_id, reason, rma_prefix="RMA"):
    # this function creates a mock RMA and logs it
    rma_id = f"{rma_prefix}-{order_id[-8:]}"  # build a simple rma id
    rec = {"type": "return", "order_id": order_id, "rma_id": rma_id, "status": "initiated", "reason": reason}  # record dict
    _write_jsonl("logs/actions.jsonl", rec)  # log to actions file
    return {"rma_id": rma_id, "status": "initiated", "reason": reason}  # return result


def cancel_order(order_id, reason):
    # this function mocks a cancel request
    rec = {"type": "cancel", "order_id": order_id, "status": "requested", "reason": reason}  # record dict
    _write_jsonl("logs/actions.jsonl", rec)  # log to actions file
    return {"cancel_id": f"CXL-{order_id[-8:]}", "status": "requested", "reason": reason}  # return result

def fuzzy_find_order_id(query):
    # this function tries to find an order id or email with fuzzy matching
    load_resources()  # make sure meta is loaded
    if META is None:
        return None  # return none if no meta
    choices = META[["order_id", "customer_email"]].astype(str).agg(" - ".join, axis=1).tolist()  # build choices list
    best = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)  # run fuzzy match
    if best and best[1] > 85:  # if match score is good
        idx = best[2]  # get the row index
        return META.iloc[idx].order_id  # return the order id
    return None  # return none if no match


def get_order_facts(order_id):
    # this function returns one row of metadata for an order id
    load_resources()  # make sure meta is loaded
    if META is None:
        return None  # return none if no meta
    row = META.loc[META.order_id == order_id]  # find row with this order id
    if row.empty:
        return None  # return none if not found
    r = row.iloc[0].to_dict()  # convert row to dict
    # add some simple policy flags
    r["is_delayed"] = (r.get("delivery_time_days") is not None) and (r["delivery_time_days"] > 10)  # mark if late
    r["low_review"] = (r.get("review_score") is not None) and (r["review_score"] <= 2)  # mark if low review
    return r  # return the dict


def initiate_refund(order_id, reason):
    # this function mocks a refund initiation
    return {
        "refund_id": f"RF-{order_id[-6:]}",  # make a fake refund id
        "status": "initiated",  # mark as initiated
        "reason": reason,  # copy reason
    }

def load_kb_resources():
    # this function loads kb faiss and kb meta
    global KB_FAISS, KB_META  # mark as global
    if KB_FAISS is None and os.path.exists(KB_INDEX_PATH):
        KB_FAISS = faiss.read_index(KB_INDEX_PATH)  # load kb index
    if KB_META is None and os.path.exists(KB_META_PATH):
        KB_META = pd.read_parquet(KB_META_PATH)  # load kb meta

def kb_search(query, k=6):
    # this function does similarity search over kb index and returns records with text
    load_resources()  # ensure embedding model is ready
    load_kb_resources()  # ensure kb index is ready
    if KB_FAISS is None or KB_META is None or MODEL is None:
        return []  # return empty if kb not ready
    vec = MODEL.encode([query], normalize_embeddings=True)  # encode query
    vec = np.array(vec, dtype="float32")  # ensure dtype
    D, I = KB_FAISS.search(vec, k)  # search kb
    hits = KB_META.iloc[I[0]].to_dict(orient="records")  # pick rows
    return hits  # return list of dicts (includes 'text', 'title', 'source', 'page')

def _load_orders_df():
    # this function loads the orders csv once into memory
    global _ORDERS_DF 
    if _ORDERS_DF is None:
        _ORDERS_DF = pd.read_csv(ORDERS_CSV)
        if "purchase_date" in _ORDERS_DF.columns:
            _ORDERS_DF["purchase_date"] = pd.to_datetime(_ORDERS_DF["purchase_date"], errors="coerce")
    return _ORDERS_DF  # return the dataframe

def get_orders_by_email(email: str, limit: int = 10):
    # this function returns up to 'limit' most recent orders for an email
    df = _load_orders_df()  
    if "customer_email" not in df.columns:
        return []
    
    mask = df["customer_email"].astype(str).str.lower() == str(email).lower()
    # sort by purchase_date desc if exists, else leave order
    if "purchase_date" in df.columns:
        rows = df.loc[mask].sort_values("purchase_date", ascending=False).head(limit)
    else:
        rows = df.loc[mask].head(limit)
    return rows.to_dict(orient="records")

def first_order_facts(orders: list) -> dict | None:
    return orders[0] if orders else None

def create_ticket(order_id, issue):
    # this function mocks a support ticket creation
    return {
        "ticket_id": f"TKT-{order_id[-6:]}",  # make a fake ticket id
        "status": "open",  # mark status as open
        "issue": issue,  # copy issue
    }

def notify_human_slack(payload: dict) -> bool:
    # this function posts a payload to slack webhook
    if not SLACK_WEBHOOK_URL:
        return False
    try:
        text = "*New Support Handoff*\n```" + json.dumps(payload, indent=2) + "```"
        r = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False
    
def notify_human_email(payload: dict) -> bool:
    # this function emails the payload to SUPPORT_EMAIL
    if not (SUPPORT_EMAIL and SMTP_HOST and SMTP_USER and SMTP_PASS):
        return False
    try:
        body = json.dumps(payload, indent=2)
        msg = MIMEText(body)
        msg["Subject"] = "New Support Handoff"
        msg["From"] = SMTP_USER
        msg["To"] = SUPPORT_EMAIL
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [SUPPORT_EMAIL], msg.as_string())
        return True
    except Exception:
        return False
    
def escalate_to_human(order_id: str, issue: str, extra: dict = None) -> dict:
    # this function logs a handoff and notifies slack/email if configured
    rec = {
        "handoff": True,
        "status": "awaiting_human",
        "order_id": order_id or None,
        "issue": issue,
        "extra": extra or {},
    }  # build record

    os.makedirs("logs", exist_ok=True)  # make logs dir
    with open("logs/handoffs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")  # write line

    ok_slack = notify_human_slack(rec)  # try slack
    ok_email = notify_human_email(rec)  # try email

    rec["notified_slack"] = ok_slack  # record flags
    rec["notified_email"] = ok_email

    return rec  # return record so compose can show it
