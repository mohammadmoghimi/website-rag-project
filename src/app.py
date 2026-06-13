import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

load_dotenv()


# -----------------------
# VECTOR STORE
# -----------------------

def get_vectorstore_from_url(url):

    loader = WebBaseLoader(url)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )   

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore


# -----------------------
# RETRIEVER
# -----------------------

def get_context_retriever_chain(vectorstore):

    llm = ChatOllama(
    model="gemma2:2b",
    temperature=0
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),

        ("user", "{input}"),

        (
            "user",
            """
Given the conversation above,
generate a search query that would help retrieve
relevant website information.
"""
        ),
    ])

    return create_history_aware_retriever(
        llm,
        retriever,
        prompt
    )


# -----------------------
# RAG CHAIN
# -----------------------

def get_conversational_rag_chain(retriever_chain):

    llm = ChatOllama(
    model="gemma2:2b",
    temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
You are a website assistant.

Answer ONLY using the provided context.

If the answer is not contained in the context,
respond:

"I could not find that information on the website."

Context:
{context}
"""
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    return create_retrieval_chain(
        retriever_chain,
        document_chain
    )


# -----------------------
# RESPONSE
# -----------------------

def get_response(user_input):

    retriever_chain = get_context_retriever_chain(
        st.session_state.vector_store
    )

    rag_chain = get_conversational_rag_chain(
        retriever_chain
    )

    response = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response["answer"]


# -----------------------
# STREAMLIT UI
# -----------------------

st.set_page_config(
    page_title="Chat With Websites",
    page_icon="🤖"
)

st.title("🤖 Chat With Websites")

with st.sidebar:

    st.header("Settings")

    website_url = st.text_input(
        "Website URL"
    )


if not website_url:

    st.info(
        "Enter a website URL to begin."
    )

else:

    if "chat_history" not in st.session_state:

        st.session_state.chat_history = [

            AIMessage(
                content="Hello! Ask me anything about the website."
            )
        ]

    if "vector_store" not in st.session_state:

        with st.spinner(
            "Loading website..."
        ):

            st.session_state.vector_store = (
                get_vectorstore_from_url(
                    website_url
                )
            )

    user_query = st.chat_input(
        "Ask a question..."
    )

    if user_query:

        response = get_response(
            user_query
        )

        st.session_state.chat_history.append(
            HumanMessage(content=user_query)
        )

        st.session_state.chat_history.append(
            AIMessage(content=response)
        )

    for message in st.session_state.chat_history:

        if isinstance(
            message,
            AIMessage
        ):

            with st.chat_message("assistant"):

                st.write(
                    message.content
                )

        elif isinstance(
            message,
            HumanMessage
        ):

            with st.chat_message("user"):

                st.write(
                    message.content
                )