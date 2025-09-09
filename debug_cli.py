# debug_cli.py
# this script lets you chat with the agent in the terminal and see internals

import os  # for env vars and paths
from dotenv import load_dotenv  # to load .env
load_dotenv()  # load env vars at start

import json  # to pretty print dicts
import traceback  # to show errors
import graph  # your langgraph app
import agent  # tools (optional to warm caches)


def pprint(obj, label=None):
    # this function prints a dict as pretty json
    if label:
        print(f"\n{label}:")
    try:
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        print(obj)


def summarize_facts(f):
    # this function builds one line summary for order facts
    if not f:
        return "no facts"
    return (
        f"order_id={f.get('order_id','')}, "
        f"status={f.get('order_status','')}, "
        f"paid={f.get('total_payment','')} via {f.get('payment_type','')}, "
        f"email={f.get('customer_email','')}"
    )


def main():
    # this function runs a simple input loop and keeps memory between turns
    print("Debug CLI for Customer Support AI (type 'exit' to quit)")
    print("Tip: commands: /mem to show memory, /reset to clear memory, /raw to toggle raw result\n")

    memory = {}  # dict to store conversation memory
    show_raw = False  # flag to toggle full result print

    # optional: warm up resources to avoid first-call latency
    try:
        agent.load_resources()  # loads FAISS + embed model
        print("resources loaded.")
    except Exception:
        pass

    while True:
        try:
            user_q = input("\n> ").strip()  # read user input
        except (EOFError, KeyboardInterrupt):
            break  # quit on ctrl+d / ctrl+c

        if user_q.lower() in {"exit", "quit", ":q"}:
            break  # exit the loop

        # simple debug commands
        if user_q == "/mem":
            pprint(memory, "memory")
            continue
        if user_q == "/reset":
            memory = {}
            print("memory cleared.")
            continue
        if user_q == "/raw":
            show_raw = not show_raw
            print("raw result:", show_raw)
            continue

        # build start state with last memory and new input
        state_in = {"input": user_q, "memory": dict(memory)}  # copy memory into state

        try:
            result = graph.APP.invoke(state_in)  # run the agent graph
        except Exception as e:
            print("ERROR running graph:")
            traceback.print_exc()
            continue

        # update memory if returned
        memory = result.get("memory", memory)

        # print assistant answer
        answer = result.get("output", "(no output)")
        print("\nASSISTANT:\n" + answer)

        # small debug summary (intent, facts, actions, timings)
        intent = result.get("intent", state_in.get("intent", None))  # may or may not be present
        facts = result.get("order_facts")
        actions = result.get("actions", [])
        timings = result.get("timings", {})
        kb_hits = result.get("kb_hits", [])

        print("\n--- debug summary ---")
        if intent:
            print("intent:", intent)
        print("facts:", summarize_facts(facts))
        if actions:
            pprint(actions, "actions")
        if timings:
            pprint(timings, "timings")
        if kb_hits:
            print(f"kb_hits: {len(kb_hits)} (showing first title/source)")
            try:
                first = kb_hits[0]
                print("  ->", first.get("title"), "|", first.get("source"))
            except Exception:
                pass

        if show_raw:
            pprint(result, "raw result")

    print("\nbye.")


if __name__ == "__main__":
    # this is the main entry point
    main()
