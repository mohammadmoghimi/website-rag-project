import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from crawler import crawl_website, log  # import logging from crawler
import time 

def build_vectorstore_from_url(url, persist_dir="D:/chroma_website_db"):
    t0 = time.time()
    log("STEP 1: Crawling website")
    documents = crawl_website(url, max_pages=30)
    log(f"Crawling took {time.time()-t0:.2f}s")

    log("STEP 2: Splitting text")
    t1 = time.time()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    log(f"Splitting took {time.time()-t1:.2f}s")

    if not chunks:
        raise ValueError("No chunks to embed – check crawling step.")

    log("STEP 3: Creating embeddings")
    t2 = time.time()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    log(f"Embedding & storage took {time.time()-t2:.2f}s")
    return vectorstore

def get_retriever_chain(vectorstore):
    """Creates a history-aware retriever chain."""
    llm = ChatOllama(model="gemma2:2b", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query that would help retrieve relevant website information.")
    ])

    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    """Creates the full conversational RAG chain."""
    llm = ChatOllama(model="gemma2:2b", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a website assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say:
"I could not find that information on the website."
When possible mention the source page.
Context:
{context}
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, document_chain)

def get_response(rag_chain, user_input, chat_history):
    t0 = time.time()
    log("STEP 4: Retrieving")
    response = rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    log(f"RAG chain invoke took {time.time()-t0:.2f}s")
    return response["answer"]