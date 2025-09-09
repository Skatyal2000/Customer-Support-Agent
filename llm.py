# llm.py
# this file wraps a local SLM or a hosted LLM behind one simple function

import os  # for reading env vars
import requests 
from dotenv import load_dotenv
import time 
import json

load_dotenv()


def build_prompt(user_input, facts, hits):
    # this function builds a simple prompt string using facts and hits
    facts_lines = []  # list to store fact lines
    if facts:
        facts_lines.append(f"order_id: {facts.get('order_id','')}")
        facts_lines.append(f"customer: {facts.get('first_name','')} {facts.get('last_name','')} <{facts.get('customer_email','')}>")
        facts_lines.append(f"status: {facts.get('order_status','')}")
        facts_lines.append(f"items: {facts.get('num_items','')}")
        facts_lines.append(f"paid: {facts.get('total_payment','')} via {facts.get('payment_type','')}")
        facts_lines.append(f"delivery_time_days: {facts.get('delivery_time_days','')}")
        facts_lines.append(f"review_score: {facts.get('review_score','')}")
    facts_text = "\n".join(facts_lines)  # join facts lines

    hit_lines = []  # list to store top chunk hints
    for h in (hits or [])[:3]:
        if "text" in h and h["text"]:
            snippet = (h["text"].strip().replace("\n", " "))[:400]  # trim snippet
            src = h.get("source", h.get("type", ""))
            tag = f"[{src}]"
            hit_lines.append(f"{tag} {snippet}")
        else:
            t = f"[{h.get('type','')}] order_id={h.get('order_id','')}"
            hit_lines.append(t)
    hits_text = "\n".join(hit_lines)
    instructions = (
        "You are a customer support assistant.\n"
        "Use the facts to answer the user. Do not invent data.\n"
        "If key info is missing, ask briefly for the order id or email.\n"
        "Be concise and action-oriented.\n"
    )  # simple instructions string

    prompt = (
        instructions + "\n"
        "=== USER QUESTION ===\n" + (user_input or "") + "\n\n"
        "=== FACTS ===\n" + facts_text + "\n\n"
        "=== TOP CHUNKS ===\n" + hits_text + "\n\n"
        "=== ANSWER ===\n"
    )  # final prompt text

    return prompt  # return the prompt string



def generate_answer(user_input, facts=None, hits=None):
    # this function calls ollama /api/generate and returns the text answer
    prompt = build_prompt(user_input, facts, hits)  # build the prompt string

    # read model and url from env, with defaults
    model = os.getenv("LOCAL_LLM_NAME", "llama3:instruct")  # ollama model name
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")  # ollama endpoint

    # read generation options from env
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "200"))  # limit tokens
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))  # sampling temp
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))  # context tokens
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "10m")  # keep model loaded
    timeout_sec = int(os.getenv("OLLAMA_TIMEOUT_SEC", "60"))  # http timeout

    payload = {
        "model": model,  # which model to use
        "prompt": prompt,  # the prompt text
        "stream": False,  # disable streaming for simplicity
        "keep_alive": keep_alive,  # keep model in memory
        "options": {  # fine-tune speed/quality
            "num_predict": num_predict,
            "temperature": temperature,
            "num_ctx": num_ctx
        }
    }  # request payload

    try:
        resp = requests.post(url, json=payload, timeout=timeout_sec)  # post to ollama api
        if resp.status_code == 404:
            return (
                "Ollama returned 404. Possible causes:\n"
                f"- Wrong URL: {url}\n"
                f"- Model not found: {model}\n"
                "Try: `curl http://localhost:11434/api/tags` to see installed models and set LOCAL_LLM_NAME accordingly."
            )
        resp.raise_for_status()  # raise error for bad responses
        data = resp.json()  # parse json
        return (data.get("response") or "").strip()  # return model text
    except requests.exceptions.ReadTimeout:
        return "The local model took too long to respond. Try a smaller model or lower OLLAMA_NUM_PREDICT."
    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama. Is the server running and the URL correct?"
    except Exception as e:
        return f"Sorry, I could not reach the local model: {e}"
    

def generate_answer_timed(user_input, facts=None, hits=None):
    # this function measures how long the model takes to answer
    t0 = time.perf_counter()  # start timer
    text = generate_answer(user_input, facts=facts, hits=hits)  # call main function
    t1 = time.perf_counter()  # end timer
    ms = int((t1 - t0) * 1000)  # compute milliseconds
    return text, ms  # return both text and latency


def _extract_json(text):
    # this function tries to extract the first json object from a string
    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    # naive scan for a {...} block
    i = t.find("{"); j = t.rfind("}")
    if i != -1 and j != -1 and j > i:
        return t[i:j+1]
    return "{}"  # fallback to empty json


def nlu_classify(user_text, memory=None):
    # this function asks the model to output only json with intent and slots
    mem = memory or {}  # memory dict or empty
    current_order = mem.get("current_order_id", "")
    current_email = mem.get("current_email", "")

    # build an instruction that forces json output
    schema_hint = (
        "Return ONLY valid JSON with this schema and nothing else:\n"
        "{\n"
        '  "intent": "track|payment|refund|return|cancel|kb|analytics|general|handoff_yes|handoff_no",\n'
        '  "slots": {"order_id": string|null, "email": string|null, "reason": string|null},\n'
        '  "yes_no": "yes|no|null"\n'
        "}\n"
        "Rules:\n"
        "- Map follow-up confirmations like 'yes please'/'no thanks' to yes_no.\n"
        "- If the user references the same order as before, you may leave slots null; we already have memory.\n"
        "- Detect KB/policy questions (return policy, shipping, payments) as 'kb' intent if no specific order.\n"
        "- Use 'handoff_yes' or 'handoff_no' ONLY when the user is clearly answering a prior escalation question.\n"
    )

    # include a tiny context so model knows about current memory
    context = f"(context) current_order_id={current_order or 'none'}, current_email={current_email or 'none'}"

    prompt = (
        "You are a classifier and slot extractor for a support agent.\n"
        + schema_hint + "\n"
        + context + "\n"
        "User:\n" + (user_text or "")
    )

    # read ollama settings (same as generate_answer)
    model = os.getenv("LOCAL_LLM_NAME", "llama3:instruct")
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "200"))
    temperature = 0.0  # for JSON it is better to be deterministic
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
    timeout_sec = int(os.getenv("OLLAMA_TIMEOUT_SEC", "60"))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"num_predict": num_predict, "temperature": temperature, "num_ctx": num_ctx}
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout_sec)
        r.raise_for_status()
        txt = (r.json().get("response") or "").strip()
        obj = json.loads(_extract_json(txt))
        # minimal sanity defaults
        intent = obj.get("intent") or "general"
        slots = obj.get("slots") or {}
        yes_no = obj.get("yes_no")
        return {"intent": intent, "slots": slots, "yes_no": yes_no}
    except Exception:
        # fallback to your old keyword rules if parsing fails
        t = (user_text or "").lower()
        if any(w in t for w in ["refund", "return", "cancel", "exchange"]):
            return {"intent": "refund", "slots": {}, "yes_no": None}
        if ("order" in t and any(w in t for w in ["where", "track", "status"])) or "track" in t:
            return {"intent": "track", "slots": {}, "yes_no": None}
        if any(w in t for w in ["payment", "pay", "installment"]):
            return {"intent": "payment", "slots": {}, "yes_no": None}
        return {"intent": "general", "slots": {}, "yes_no": None}
