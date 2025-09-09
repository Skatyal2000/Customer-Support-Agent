# metrics.py
# this file measures latency and a simple accuracy check and writes logs

import os  # for paths
import json  # for json logs
import time  # for timing
from datetime import datetime  # for timestamps


def now_ms():
    # this function returns current time in milliseconds
    return int(time.perf_counter() * 1000)


def grounding_score(answer_text, facts):
    # this function checks if important facts are present in the answer
    ans = (answer_text or "").lower()  # stores answer in lowercase
    f = facts or {}  # stores facts dict or empty

    # prepare expected strings from facts for simple containment checks
    expect = {
        "order_id": str(f.get("order_id", "")).lower(),
        "order_status": str(f.get("order_status", "")).lower(),
        "total_payment": str(f.get("total_payment", "")).lower(),
        "payment_type": str(f.get("payment_type", "")).lower(),
        "customer_email": str(f.get("customer_email", "")).lower(),
    }  # keys to check

    # do simple contains checks
    checks = {k: (v != "" and v in ans) for k, v in expect.items()}  # dict of booleans

    # compute a simple score (fraction of true)
    if len(checks) == 0:
        score = 0.0  # default score if nothing to check
    else:
        score = sum(1 for v in checks.values() if v) / float(len(checks))  # fraction

    return {"score": round(score, 3), "checks": checks}  # return score and details


def write_log(record, path="logs/metrics.jsonl"):
    # this function appends one json record to a log file
    os.makedirs(os.path.dirname(path), exist_ok=True)  # ensure folder exists
    record["ts"] = datetime.utcnow().isoformat() + "Z"  # add timestamp
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")  # write one line
