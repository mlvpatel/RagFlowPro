"""RagFlowPro Streamlit application entry point."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from frontend.chat_interface import display_chat_interface  # noqa: E402
from frontend.sidebar import display_sidebar  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="RagFlowPro", layout="wide")
    st.title("RagFlowPro")
    st.caption("Retrieval augmented chat over your documents")
    display_sidebar()
    display_chat_interface()


if __name__ == "__main__":
    main()
