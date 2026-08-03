import os
import streamlit as st
from rag import build_vectorstore_from_url, get_retriever_chain, get_conversational_rag_chain, get_response
from langchain_core.messages import AIMessage, HumanMessage
from utils import get_index_name_from_url
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['no_proxy'] = 'localhost,127.0.0.1'

st.set_page_config(page_title="Chat With Websites")
st.title(" Chat With Websites")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    st.divider()


if not website_url:
    st.info("Enter a website URL")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! Ask me anything about the website.")
        ]

    if "indexed_url" not in st.session_state:
        st.session_state.indexed_url = None

    if website_url != st.session_state.indexed_url:
        with st.spinner("Building knowledge base..."):
            st.session_state.chat_history = [
                AIMessage(content=f"Hello! Ask me anything about {website_url}")
            ]

            vector_store = build_vectorstore_from_url(website_url)

            retriever_chain = get_retriever_chain(vector_store)
            st.session_state.rag_chain = get_conversational_rag_chain(retriever_chain)

            st.session_state.indexed_url = website_url

    # Chat input
    user_query = st.chat_input("Ask a question...")
    if user_query:
        answer = get_response(
            st.session_state.rag_chain,
            user_query,
            st.session_state.chat_history
        )
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=answer))

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)