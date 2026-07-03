"""Chat interface component for the RagFlowPro Streamlit app."""
import streamlit as st

from frontend import api_utils


def display_chat_interface() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        def token_stream():
            model = st.session_state.get("model", "gpt-4o-mini")
            for event in api_utils.stream_chat(prompt, st.session_state.session_id, model):
                if "token" in event:
                    yield event["token"]
                elif event.get("done"):
                    st.session_state.session_id = event.get("session_id")

        answer = st.write_stream(token_stream())

    st.session_state.messages.append({"role": "assistant", "content": answer})
