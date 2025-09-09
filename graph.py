# graph.py
# this script builds a simple langgraph workflow for customer support with memory

from langgraph.graph import StateGraph, END  # for making the graph
from typing import TypedDict, List, Dict, Any  # for state schema types
import re  # for regex simple checks
import agent  # import the helper tools file (agent.py)
import llm  # for the model call (compose step)


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




def retrieve(state):
    q = state.get("input", "")  # query
    order_hits = agent.rag_search(q, k=6)  # order/review search
    kb_hits = agent.kb_search(q, k=6)  # kb search
    return {"hits": order_hits, "kb_hits": kb_hits}  # store both



def resolve_facts(state: GraphState) -> Dict[str, Any]:
    # this function tries to resolve an order using query or memory
    q = state.get("input", "")  # get query text
    mem = dict(state.get("memory", {}))  # copy memory dict so we can update it
    orders = []  # list to store multiple orders for email lookups
    facts = None  # this will store one chosen order facts

    # try to parse email from the query
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", q, flags=re.I)  # find an email pattern
    if email_match:
        email = email_match.group(0)  # extract email text
        mem["current_email"] = email  # remember last seen email
        try:
            # this helper exists if you added it earlier; if not, it will just not run
            orders = agent.get_orders_by_email(email, limit=10)  # get orders for this email
        except Exception:
            orders = []  # if helper not present or fails, keep empty

        if orders:
            # if we found orders, pick the first one as current context
            facts = agent.first_order_facts(orders) if hasattr(agent, "first_order_facts") else dict(orders[0])
            if "order_id" in facts:
                mem["current_order_id"] = facts["order_id"]  # remember current order id

    # if we still have no facts, try to detect an order id from the query
    if not facts:
        order_id_from_text = agent.fuzzy_find_order_id(q)  # find order id by fuzzy
        if order_id_from_text:
            facts = agent.get_order_facts(order_id_from_text)  # get facts for that id
            if facts and "order_id" in facts:
                mem["current_order_id"] = facts["order_id"]  # remember it

    # if still nothing, try to use the last known order id from memory
    if not facts and "current_order_id" in mem:
        facts = agent.get_order_facts(mem["current_order_id"])  # reuse previous order id

    # as a last fallback, use the first RAG hit's order_id (if present)
    if not facts and state.get("hits"):
        first_hit = state["hits"][0]  # take first hit
        oid = first_hit.get("order_id")  # extract order id
        if oid:
            facts = agent.get_order_facts(oid)  # get facts
            if facts and "order_id" in facts:
                mem["current_order_id"] = facts["order_id"]  # remember it

    # return facts, any found orders list, and the updated memory
    return {"order_facts": facts, "orders": orders, "memory": mem}


def policy_and_actions(state: GraphState) -> Dict[str, Any]:
    # this function decides if any actions are needed like escalate, return, refund, cancel
    facts = state.get("order_facts")  # get order facts
    intent = (state.get("intent") or "").lower()  # get intent
    actions = []  # list for actions

    if not facts:
        return {"actions": actions}  # if no facts, nothing to do

    oid = facts.get("order_id")  # order id string
    status = str(facts.get("order_status", "")).lower()  # status text

    # always open tickets for SLA or low review (as before)
    if facts.get("low_review"):
        actions.append(agent.create_ticket(oid, "low review complaint"))  # escalate ticket
    if facts.get("is_delayed"):
        actions.append(agent.create_ticket(oid, "late delivery"))  # escalate ticket

    # handle cancel requests (pre-delivery)
    if intent in {"cancel"}:
        if status in {"created", "approved", "processing", "shipped"}:
            actions.append(agent.cancel_order(oid, "user requested cancel"))  # request cancel
        else:
            # if already delivered, escalate to human
            actions.append(agent.escalate_to_human(oid, "cancel after delivery not supported", {"status": status}))
        return {"actions": actions}  # return after handling cancel

    # handle refund/return requests (post-delivery)
    if intent in {"refund", "return"}:
        elig = agent.check_return_eligibility(facts, window_days=30)  # check window
        if elig.get("eligible"):
            actions.append(agent.start_return(oid, f"user requested {intent}"))  # start return
        else:
            # not eligible → handoff to human for discretion
            actions.append(agent.escalate_to_human(oid, f"{intent} not eligible", {"eligibility": elig}))
        return {"actions": actions}  # return after handling return/refund

    return {"actions": actions}  # default return

def compose(state: GraphState) -> Dict[str, Any]:
    # this function creates the final reply text (uses kb if no order facts)
    user_q = state.get("input", "")  # user question text
    facts = state.get("order_facts")  # selected order facts (may be None)
    hits = state.get("hits", [])  # top order/review chunks
    kb_hits = state.get("kb_hits", [])  # top kb/policy chunks
    timings_prev = state.get("timings", {})  # timings from earlier steps

    # 1) if we have an email query with multiple orders, list a few recent ones
    orders = state.get("orders", [])  # list of order rows when user gave email
    if orders:
        lines = []  # list of lines to print
        first = orders[0] if len(orders) > 0 else {}
        lines.append(
            f"Found {len(orders)} orders for "
            f"{first.get('first_name','')} {first.get('last_name','')} "
            f"<{first.get('customer_email','')}>"
        )  # header line
        lines.append("")  # blank line
        lines.append("Most recent orders:")  # section title
        for r in orders[:5]:  # show top 5
            pid = r.get("purchase_date", "")
            if hasattr(pid, "strftime"):
                pid = pid.strftime("%Y-%m-%d")  # format date if datetime
            lines.append(
                f"- {r.get('order_id','')} | {r.get('order_status','')} | {pid} | "
                f"{r.get('total_payment','')} via {r.get('payment_type','')} | "
                f"items={r.get('num_items','')} | review={r.get('review_score','')}"
            )
        return {"output": "\n".join(lines), "timings": timings_prev}  # return list summary

    # 2) if we don’t have an order but do have KB hits, answer from KB
    if not facts and kb_hits:
        snippets = []  # small list of kb snippets
        for h in kb_hits[:3]:
            txt = (h.get("text") or "").strip().replace("\n", " ")
            if len(txt) > 400:
                txt = txt[:400] + "..."  # trim long text
            src = h.get("source", "")
            page = h.get("page", None)
            tag = f"{src}" + (f" p.{page}" if page else "")
            snippets.append(f"[{tag}] {txt}")  # add labeled snippet
        kb_context = "\n".join(snippets)  # join snippets
        prompt = (
            f"{user_q}\n\n"
            "Use the following knowledge snippets to answer. Do not invent facts.\n"
            f"{kb_context}\n\n"
            "Answer succinctly and cite files inline like [filename p.X] when relevant."
        )  # build a kb prompt
        answer_text, gen_ms = llm.generate_answer_timed(prompt, facts={"order_id": "KB-ONLY"}, hits=[])  # call model
        timings = dict(timings_prev); timings["generation_ms"] = gen_ms  # keep timings
        if "retrieve_ms" in timings:
            try:
                timings["total_ms"] = int(timings.get("retrieve_ms", 0)) + int(gen_ms)  # total latency
            except Exception:
                pass  # ignore if types mismatch
        return {"output": answer_text, "timings": timings}  # return kb answer

    # 3) if still no facts, ask for order id/email
    if not facts:
        return {"output": "I could not find your order. Please share your order id or the email used for purchase."}

    # 4) normal order-answer path (we have facts)
    answer_text, gen_ms = llm.generate_answer_timed(user_q, facts=facts, hits=hits)  # call model
    timings = dict(timings_prev); timings["generation_ms"] = gen_ms  # update timings
    if "retrieve_ms" in timings:
        try:
            timings["total_ms"] = int(timings.get("retrieve_ms", 0)) + int(gen_ms)  # compute total
        except Exception:
            pass  # ignore casting issues

    # 5) append actions (tickets/RMA/cancel/handoff) if any ran
    acts = state.get("actions", [])
    if acts:
        lines = [answer_text, "", "Actions taken:"]
        for a in acts:
            lines.append(str(a))  # show each action as a line
        return {"output": "\n".join(lines), "timings": timings}  # return answer + actions

    # 6) no actions, just return the plain answer
    return {"output": answer_text, "timings": timings}  # final result




# now build the graph with a state schema
graph = StateGraph(GraphState)  # make graph with schema
graph.add_node("classify", classify)  # add classify node
graph.add_node("retrieve", retrieve)  # add retrieve node
graph.add_node("resolve_facts", resolve_facts)  # add resolve facts node
graph.add_node("policy_and_actions", policy_and_actions)  # add policy node
graph.add_node("compose", compose)  # add compose node

graph.set_entry_point("classify")  # set entry node
graph.add_edge("classify", "retrieve")  # after classify go to retrieve
graph.add_edge("retrieve", "resolve_facts")  # after retrieve go to resolve
graph.add_edge("resolve_facts", "policy_and_actions")  # after resolve go to policy
graph.add_edge("policy_and_actions", "compose")  # after policy go to compose
graph.add_edge("compose", END)  # compose is the last step

APP = graph.compile()  # compile the graph into app
