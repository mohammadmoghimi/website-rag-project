from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from crawler import crawl_website
import time 
from langchain_elasticsearch import ElasticsearchStore
from retrievers import ElasticsearchHybridRetriever
from elasticsearch import Elasticsearch
from utils import extract_main_text , get_index_name_from_url
import os

os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['no_proxy'] = 'localhost,127.0.0.1'

def build_vectorstore_from_url(url , index_name = None):
    if index_name is None :
        index_name = get_index_name_from_url(url)

    t0 = time.time()
    documents = crawl_website(url, max_pages=15)
    print(f"Crawling took {time.time()-t0:.2f}s")

    for i, doc in enumerate(documents, 1):
        print(f"  Cleaning document {i}/{len(documents)}...")
        doc.page_content = extract_main_text(doc.page_content)

    t1 = time.time()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Splitting took {time.time()-t1:.2f}s")
    print(f"Number of chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks to embed – check crawling step.")

    print("STEP 3: Creating embeddings and indexing")
    t2 = time.time()

    embeddings = OllamaEmbeddings(model="all-minilm") 

    vectorstore = ElasticsearchStore(
        es_url="http://localhost:9200",
        index_name=index_name,
        embedding=embeddings,
    )

    batch_size = 10
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]
        
        texts = [chunk.page_content for chunk in batch]
        metadatas = [chunk.metadata for chunk in batch]
        
        print(f" Indexing batch {batch_idx+1}/{total_batches} ({len(batch)} chunks)...")
        batch_start = time.time()
        
        try:
            vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                refresh=False    
            )
            print(f"    Batch done in {time.time()-batch_start:.2f}s")
        except Exception as e:
            print(f"   Error: {e}")
            raise

    es = Elasticsearch("http://localhost:9200")
    es.indices.refresh(index=index_name)
    print(f" All chunks indexed in {time.time()-t2:.2f}s")


    vectorstore.custom_index_name = index_name
    return vectorstore

def get_retriever_chain(vectorstore):
    llm = ChatOllama(model="llama3.2:1b", temperature=0)

    es_client = vectorstore.client

    retriever = ElasticsearchHybridRetriever(
        es_client=es_client,
        index_name=vectorstore.custom_index_name,
        embedding_model=vectorstore.embeddings,   
        k=4
    )

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query that would help retrieve relevant website information.")
    ])

    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOllama(model="llama3.2:1b", temperature=0)

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
    print("STEP 4: Retrieving")
    response = rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    print(f"RAG chain invoke took {time.time()-t0:.2f}s")
    return response["answer"]

