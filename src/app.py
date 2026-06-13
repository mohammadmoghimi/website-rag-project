import streamlit as st
import requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from langchain_core.messages import (
    AIMessage,
    HumanMessage
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings
)

from langchain_community.document_loaders import (
    WebBaseLoader
)

from langchain_community.vectorstores import (
    Chroma
)

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)


# -----------------------
# LOGGING
# -----------------------

def log(message):

    print(message)

    if "logs" not in st.session_state:
        st.session_state.logs = []

    st.session_state.logs.append(message)


# -----------------------
# WEBSITE CRAWLER
# -----------------------

def crawl_website(start_url, max_pages=30):

    visited = set()

    queue = [start_url]

    documents = []

    domain = urlparse(start_url).netloc

    while queue and len(visited) < max_pages:

        current_url = queue.pop(0)

        if current_url in visited:
            continue

        try:

            log(f"🌐 Crawling: {current_url}")

            visited.add(current_url)

            loader = WebBaseLoader(current_url)

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = current_url

            documents.extend(docs)

            response = requests.get(
                current_url,
                timeout=10
            )

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            for link in soup.find_all(
                "a",
                href=True
            ):

                href = link["href"]

                full_url = urljoin(
                    current_url,
                    href
                )

                parsed = urlparse(full_url)

                if (
                    parsed.netloc == domain
                    and full_url not in visited
                ):
                    queue.append(full_url)

        except Exception as e:

            log(
                f"❌ Error on {current_url}: {e}"
            )

    log(
        f"✅ Crawled {len(visited)} pages"
    )

    return documents


# -----------------------
# VECTOR STORE
# -----------------------

def get_vectorstore_from_url(url):

    log("STEP 1: Crawling website")

    documents = crawl_website(
        url,
        max_pages=30
    )

    log(
        f"Loaded {len(documents)} documents"
    )

    log("STEP 2: Splitting text")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    log(
        f"Created {len(chunks)} chunks"
    )

    log("STEP 3: Creating embeddings")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    log("✅ Vector store ready")

    return vectorstore


# -----------------------
# RETRIEVER CHAIN
# -----------------------

def get_context_retriever_chain(
    vectorstore
):

    llm = ChatOllama(
        model="gemma2:2b",
        temperature=0
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_messages([

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        ("user", "{input}"),

        (
            "user",
            """
Generate a search query that
would help retrieve relevant
website information.
"""
        )
    ])

    return create_history_aware_retriever(
        llm,
        retriever,
        prompt
    )


# -----------------------
# RAG CHAIN
# -----------------------

def get_conversational_rag_chain(
    retriever_chain
):

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

If the answer is not in the context,
say:

"I could not find that information on the website."

When possible mention the source page.

Context:
{context}
"""
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        ("user", "{input}")
    ])

    document_chain = (
        create_stuff_documents_chain(
            llm,
            prompt
        )
    )

    return create_retrieval_chain(
        retriever_chain,
        document_chain
    )


# -----------------------
# RESPONSE
# -----------------------

def get_response(user_input):

    log("STEP 4: Retrieving")

    response = (
        st.session_state.rag_chain.invoke({

            "chat_history":
                st.session_state.chat_history,

            "input":
                user_input
        })
    )

    log("STEP 5: Answer generated")

    return response["answer"]


# -----------------------
# STREAMLIT
# -----------------------

st.set_page_config(
    page_title="Chat With Websites",
    page_icon="🤖"
)

st.title(
    "🤖 Chat With Websites"
)

with st.sidebar:

    st.header("Settings")

    website_url = st.text_input(
        "Website URL"
    )

    st.divider()

    st.subheader("Logs")

    if "logs" in st.session_state:

        for item in st.session_state.logs[-20:]:

            st.text(item)


if not website_url:

    st.info(
        "Enter a website URL"
    )

else:

    if "chat_history" not in st.session_state:

        st.session_state.chat_history = [

            AIMessage(
                content=(
                    "Hello! Ask me anything "
                    "about the website."
                )
            )
        ]

    if "vector_store" not in st.session_state:

        with st.spinner(
            "Building knowledge base..."
        ):

            st.session_state.vector_store = (
                get_vectorstore_from_url(
                    website_url
                )
            )

            retriever_chain = (
                get_context_retriever_chain(
                    st.session_state.vector_store
                )
            )

            st.session_state.rag_chain = (
                get_conversational_rag_chain(
                    retriever_chain
                )
            )

    user_query = st.chat_input(
        "Ask a question..."
    )

    if user_query:

        answer = get_response(
            user_query
        )

        st.session_state.chat_history.append(
            HumanMessage(
                content=user_query
            )
        )

        st.session_state.chat_history.append(
            AIMessage(
                content=answer
            )
        )

    for message in (
        st.session_state.chat_history
    ):

        if isinstance(
            message,
            AIMessage
        ):

            with st.chat_message(
                "assistant"
            ):

                st.write(
                    message.content
                )

        elif isinstance(
            message,
            HumanMessage
        ):

            with st.chat_message(
                "user"
            ):

                st.write(
                    message.content
                )