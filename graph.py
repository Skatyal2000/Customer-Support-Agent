# graph.py
# this script builds a simple langgraph workflow for customer support with memory

from langgraph.graph import StateGraph, END  # for making the graph
from typing import TypedDict, List, Dict, Any  # for state schema types
import re  # for regex simple checks
import agent  # import the helper tools file (agent.py)
import llm  # for the model call (compose step)

AUTO_ESCALATE_ON = True       # turn auto-handoff on/off
MAX_REPEAT_INTENT = 2         # how many repeated intents before handoff
MAX_NO_FACTS = 2              # how many turns with no order facts before handoff

class GraphState(TypedDict, total=False):
    # this class defines the keys we keep in the graph state
    input: str  # the user's latest message
    intent: str  # the detected intent name
    hits: List[Dict[str, Any]]  # top rag results
    order_facts: Dict[str, Any]  # the chosen order facts
    actions: List[Dict[str, Any]]  # any actions taken
    output: str  # final answer text
    timings: Dict[str, Any]  # optional timings
    accuracy: float  # optional grounding score
    kb_hits: List[Dict[str, Any]]
    accuracy_checks: Dict[str, bool]  # optional which fields found
    orders: List[Dict[str, Any]]  # optional multiple orders for email
    memory: Dict[str, Any]  # this stores context across turns (like current_order_id, current_email)


def classify(state: GraphState) -> Dict[str, Any]:
    # this function asks the LLM to decide the intent and extract slots
    q = (state.get("input") or "")  # user text
    mem = dict(state.get("memory", {}))  # copy memory to update
    nlu = llm.nlu_classify(q, memory=mem)  # call llm for intent + slots

    intent = nlu.get("intent", "general")  # read intent
    yes_no = nlu.get("yes_no")  # read yes/no
    slots = (nlu.get("slots") or {})  # read slots dict

    # map explicit yes/no for pending handoff to special intents
    if mem.get("pending_handoff") and yes_no in {"yes", "no"}:
        intent = "handoff_yes" if yes_no == "yes" else "handoff_no"

    # store extracted slots into memory for downstream steps
    if slots.get("order_id"):
        mem["current_order_id"] = slots["order_id"]
    if slots.get("email"):
        mem["current_email"] = slots["email"]
    if slots.get("reason"):
        mem["last_reason"] = slots["reason"]

    return {"intent": intent, "memory": mem}




def retrieve(state): #RAG Search
    q = state.get("input", "")  # query
    order_hits = agent.rag_search(q, k=6)  # order/review search
    kb_hits = agent.kb_search(q, k=6)  # kb search
    return {"hits": order_hits, "kb_hits": kb_hits}  # store both



def resolve_facts(state: GraphState) -> Dict[str, Any]:
    # this function tries to resolve an order using query, memory, or rag
    q = state.get("input", "")  # user text
    mem = dict(state.get("memory", {}))  # copy memory to update
    orders = []  # list for email path (multi orders)
    facts = None  # chosen order facts

    # 1) try to parse email from the query (exact email lookup)
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", q, flags=re.I)
    if email_match:
        email = email_match.group(0)
        mem["current_email"] = email
        try:
            orders = agent.get_orders_by_email(email, limit=10)  # exact from csv
        except Exception:
            orders = []
        if orders:
            facts = agent.first_order_facts(orders) if hasattr(agent, "first_order_facts") else dict(orders[0])
            if facts and "order_id" in facts:
                mem["current_order_id"] = facts["order_id"]

    # 2) if still no facts, try fuzzy order id from text (typo-tolerant id)
    if not facts:
        oid_txt = agent.fuzzy_find_order_id(q)
        if oid_txt:
            facts = agent.get_order_facts(oid_txt)
            if facts and "order_id" in facts:
                mem["current_order_id"] = facts["order_id"]

    # 3) if still no facts, try memory from previous turns (same order)
    if not facts and "current_order_id" in mem:
        facts = agent.get_order_facts(mem["current_order_id"])

    # 4) if still no facts AND we have rag hits, use first hit's order_id
    if not facts and state.get("hits"):
        first_hit = state["hits"][0]
        oid = first_hit.get("order_id")
        if oid:
            facts = agent.get_order_facts(oid)
            if facts and "order_id" in facts:
                mem["current_order_id"] = facts["order_id"]

    return {"order_facts": facts, "orders": orders, "memory": mem}  # return results



def policy_and_actions(state: GraphState) -> Dict[str, Any]:
    # this function decides business actions using facts + intent
    facts = state.get("order_facts")
    intent = (state.get("intent") or "").lower()
    actions = list(state.get("actions", []))  # copy so we append safely

    if not facts:
        return {"actions": actions}  # nothing to do without an order

    oid = facts.get("order_id")
    status = str(facts.get("order_status", "")).lower()

    # auto tickets (examples)
    if facts.get("low_review"):
        actions.append(agent.create_ticket(oid, "low review complaint"))
    if facts.get("is_delayed"):
        actions.append(agent.create_ticket(oid, "late delivery"))

    # cancel (pre-delivery)
    if intent == "cancel":
        if status in {"created", "approved", "processing", "shipped"}:
            actions.append(agent.cancel_order(oid, "user requested cancel"))
        else:
            actions.append(agent.escalate_to_human(oid, "cancel after delivery not supported", {"status": status}))
        return {"actions": actions}

    # refund/return (post-delivery)
    if intent in {"refund", "return"}:
        elig = agent.check_return_eligibility(facts, window_days=30)
        if elig.get("eligible"):
            actions.append(agent.start_return(oid, f"user requested {intent}"))
        else:
            actions.append(agent.escalate_to_human(oid, f"{intent} not eligible", {"eligibility": elig}))
        return {"actions": actions}

    return {"actions": actions}  # default: no extra actions

def supervise_and_escalate(state: GraphState) -> Dict[str, Any]:
    # this function detects loops and triggers automatic handoff
    mem = dict(state.get("memory", {}))  # copy memory
    actions = list(state.get("actions", []))  # copy actions

    intent = (state.get("intent") or "").lower()
    has_facts = bool(state.get("order_facts"))

    sc = dict(mem.get("stuck_counts", {}))  # counters
    sc.setdefault("repeat_intent", 0)
    sc.setdefault("no_facts", 0)

    last_intent = mem.get("last_intent", "")
    took_action = len(actions) > 0

    # track repeats
    if intent and last_intent and intent == last_intent and not took_action:
        sc["repeat_intent"] += 1
    else:
        sc["repeat_intent"] = 0

    # track no-facts on order-required intents
    if intent in {"track", "payment", "refund", "return", "cancel"} and not has_facts:
        sc["no_facts"] += 1
    else:
        sc["no_facts"] = 0

    should_auto = AUTO_ESCALATE_ON and (sc["repeat_intent"] >= MAX_REPEAT_INTENT or sc["no_facts"] >= MAX_NO_FACTS)
    if should_auto:
        oid = state.get("order_facts", {}).get("order_id") if state.get("order_facts") else None
        issue = "conversation stuck (repeat/no-facts)"
        extra = {
            "intent": intent,
            "repeat_intent": sc["repeat_intent"],
            "no_facts": sc["no_facts"],
            "current_email": mem.get("current_email"),
            "current_order_id": mem.get("current_order_id"),
        }
        hand = agent.escalate_to_human(oid or "", issue, extra)  # notify/log
        actions.append(hand)
        mem["auto_handoff"] = True
        mem.pop("pending_handoff", None)
        sc["repeat_intent"] = 0
        sc["no_facts"] = 0

    mem["stuck_counts"] = sc
    if intent:
        mem["last_intent"] = intent

    return {"actions": actions, "memory": mem}


def compose(state: GraphState) -> Dict[str, Any]:
    # this function creates the final reply text
    user_q = state.get("input", "")  # user question text
    facts = state.get("order_facts")  # chosen order facts
    hits = state.get("hits", [])  # order/review hits
    kb_hits = state.get("kb_hits", [])  # kb hits
    timings_prev = state.get("timings", {})  # timings so far
    mem = state.get("memory", {})  # memory dict

    # 0) show clear message if auto-handoff happened
    if mem.get("auto_handoff"):
        acts = state.get("actions", [])
        hand = next((a for a in acts if isinstance(a, dict) and a.get("handoff")), None)
        case_id = hand.get("order_id", "") if hand else ""
        notified = []
        if hand and hand.get("notified_slack"): notified.append("Slack")
        if hand and hand.get("notified_email"): notified.append("email")
        note = " and ".join(notified) if notified else "log"
        lines = ["I'm handing this conversation to a human specialist so you get faster help."]
        if case_id: lines.append(f"Reference ID: {case_id}")
        lines.append(f"(notification sent via {note})")
        return {"output": "\n".join(lines), "timings": timings_prev}

    # 1) if multiple orders were found (email path), list a few
    orders = state.get("orders", [])
    if orders:
        lines = []
        first = orders[0] if len(orders) > 0 else {}
        lines.append(
            f"Found {len(orders)} orders for "
            f"{first.get('first_name','')} {first.get('last_name','')} "
            f"<{first.get('customer_email','')}>"
        )
        lines.append("")
        lines.append("Most recent orders:")
        for r in orders[:5]:
            pid = r.get("purchase_date", "")
            if hasattr(pid, "strftime"):
                pid = pid.strftime("%Y-%m-%d")
            lines.append(
                f"- {r.get('order_id','')} | {r.get('order_status','')} | {pid} | "
                f"{r.get('total_payment','')} via {r.get('payment_type','')} | "
                f"items={r.get('num_items','')} | review={r.get('review_score','')}"
            )
        return {"output": "\n".join(lines), "timings": timings_prev}

    # 2) kb-only answer if no order facts but we have kb hits
    if not facts and kb_hits:
        snippets = []
        for h in kb_hits[:3]:
            txt = (h.get("text") or "").strip().replace("\n", " ")
            if len(txt) > 400: txt = txt[:400] + "..."
            src = h.get("source", "kb")
            page = h.get("page", None)
            tag = f"{src}" + (f" p.{page}" if page else "")
            snippets.append(f"[{tag}] {txt}")
        prompt = (
            f"{user_q}\n\n"
            "Use the following knowledge snippets to answer. Do not invent facts.\n"
            + "\n".join(snippets)
            + "\n\nAnswer succinctly and cite files inline like [filename p.X] when relevant."
        )
        answer_text, gen_ms = llm.generate_answer_timed(prompt, facts={"order_id": "KB-ONLY"}, hits=[])
        timings = dict(timings_prev); timings["generation_ms"] = gen_ms
        if "retrieve_ms" in timings:
            try: timings["total_ms"] = int(timings.get("retrieve_ms", 0)) + int(gen_ms)
            except Exception: pass
        return {"output": answer_text, "timings": timings}

    # 3) ask for identifiers if still no facts
    if not facts:
        return {"output": "I could not find your order. Please share your order id or the email used for purchase."}

    # 4) grounded single-order answer using facts (+ optional hits)
    answer_text, gen_ms = llm.generate_answer_timed(user_q, facts=facts, hits=hits)
    timings = dict(timings_prev); timings["generation_ms"] = gen_ms
    if "retrieve_ms" in timings:
        try: timings["total_ms"] = int(timings.get("retrieve_ms", 0)) + int(gen_ms)
        except Exception: pass

    # 5) append any actions we took
    acts = state.get("actions", [])
    if acts:
        lines = [answer_text, "", "Actions taken:"]
        for a in acts: lines.append(str(a))
        return {"output": "\n".join(lines), "timings": timings}

    # 6) otherwise just return the answer
    return {"output": answer_text, "timings": timings}




def route_after_resolve_1(state: GraphState) -> str:
    # this function chooses where to go after first resolve
    return "policy_and_actions" if state.get("order_facts") else "retrieve"

def route_after_resolve_2(state: GraphState) -> str:
    # this function chooses where to go after second resolve
    return "policy_and_actions" if state.get("order_facts") else "supervise_and_escalate"


# now build the graph with a state schema
graph = StateGraph(GraphState)                         
graph.add_node("classify", classify)                   
graph.add_node("retrieve", retrieve)                   
graph.add_node("resolve_facts", resolve_facts)        
graph.add_node("resolve_facts_2", resolve_facts)       
graph.add_node("policy_and_actions", policy_and_actions)  
graph.add_node("supervise_and_escalate", supervise_and_escalate)  
graph.add_node("compose", compose)



graph.set_entry_point("classify")                      # start at classify
graph.add_edge("classify", "resolve_facts")            # go to resolve first

# after first resolve → either policy (have facts) or retrieve (no facts)
graph.add_conditional_edges("resolve_facts", route_after_resolve_1)

# if we went to retrieve, then we run a second resolve
graph.add_edge("retrieve", "resolve_facts_2")

# after second resolve → either policy (have facts) or supervisor (still none)
graph.add_conditional_edges("resolve_facts_2", route_after_resolve_2)

# after policy we always run supervisor, then compose
graph.add_edge("policy_and_actions", "supervise_and_escalate")
graph.add_edge("supervise_and_escalate", "compose")

# compose is last
graph.add_edge("compose", END)

APP = graph.compile()  # compile the graph into an app
