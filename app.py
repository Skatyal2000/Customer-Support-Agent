# app.py
# this script runs a simple streamlit app with one Chat page and keeps memory

import streamlit as st  # for the web ui
import graph  # for the agent app (APP)


def init_chat():
    # this function sets up chat history and the first assistant greeting
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list to store chat messages
        st.session_state.messages.append({  # first message from assistant
            "role": "assistant",
            "content": (
                "Hello, I'm your customer support assistant. "
                "Please tell me your order id (or the email you used) and what you need help with."
            )
        })
    if "graph_state" not in st.session_state:
        st.session_state.graph_state = {"memory": {}}  # dict to keep memory across turns


def render_messages():
    # this function displays all saved messages in order
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def handle_user_message(user_text):
    # this function sends the user text to the agent and keeps the graph state
    st.session_state.messages.append({"role": "user", "content": user_text})  # save user msg
    with st.chat_message("user"):
        st.markdown(user_text)  # show user msg

    try:
        start_state = dict(st.session_state.graph_state)  # copy previous graph state
        start_state["input"] = user_text  # set new input
        result = graph.APP.invoke(start_state)  # run graph

        # persist only memory across turns
        new_mem = result.get("memory", st.session_state.graph_state.get("memory", {}))  # read memory
        st.session_state.graph_state = {"memory": new_mem}  # store back memory

        answer = result.get("output", "no response")  # assistant text
        timings = result.get("timings", {})  # timing dict (retrieve_ms, generation_ms, total_ms)
        acc = result.get("accuracy", None)  # grounding score (float)
        checks = result.get("accuracy_checks", {})  # which fields were present (dict)
    except Exception as e:
        answer = f"Sorry, there was an error: {e}"  # fallback on error
        timings, acc, checks = {}, None, {}

    st.session_state.messages.append({"role": "assistant", "content": answer})  # save reply
    with st.chat_message("assistant"):
        st.markdown(answer)  # show reply

        # details panel (latency + accuracy) under the reply
        if timings or acc is not None or checks:
            st.markdown("**Details (latency and accuracy)**")  # section title
            if timings:
                st.json({"Timings (ms)": timings})  # show timings
            if acc is not None:
                st.write(f"Grounding score: {acc}")  # show score
            if checks:
                st.json({"Checks": checks})  # show checks



def main():
    # this function runs the single chat page
    st.set_page_config(page_title="Customer Support AI", layout="wide")  # set page settings
    st.title("Customer Support AI")  # main title

    init_chat()  # initialize chat with greeting and memory holder
    render_messages()  # show previous messages

    user_text = st.chat_input("Type your message here")  # chat input at bottom
    if user_text:
        handle_user_message(user_text)  # process the new message


if __name__ == "__main__":
    # this is the main entry point
    main()  # run the app
