import time
from elasticsearch import Elasticsearch
from langchain_ollama import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from rag import get_retriever_chain, get_conversational_rag_chain, get_response
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os

# -------------------------------------------------------------
# 1. Connect to existing Elasticsearch index
# -------------------------------------------------------------
print("Connecting to existing Elasticsearch index...")
# embeddings = OllamaEmbeddings(model="all-minilm")
embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY_FINEGRAINED")
)
vectorstore = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="my_rag_index",
    embedding=embeddings,
)
count = vectorstore.client.count(index="my_rag_index")['count']
print(f"✓ Connected. Index has {count} documents.\n")

# -------------------------------------------------------------
# 2. Build the retriever chain and the full RAG chain
# -------------------------------------------------------------
print("Building retriever chain...")
retriever_chain = get_retriever_chain(vectorstore)

print("Building conversational RAG chain...")
rag_chain = get_conversational_rag_chain(retriever_chain)

# -------------------------------------------------------------
# 3. Test queries
# -------------------------------------------------------------
queries = [
    "admissions",
    "research and innovation",
    "What is MIT famous for?",
]

print("\n" + "="*70)
print("FULL RAG RESPONSES (with LLM generation)")
print("="*70)

for q in queries:
    print(f"\n🔍 Query: '{q}'")
    
    # (A) Retrieve documents (as before)
    docs = retriever_chain.invoke({
        "input": q,
        "chat_history": []
    })
    print(f"   Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:200].replace('\n', ' ')
        source = doc.metadata.get("source", "?")
        print(f"   {i}. [{source}] {preview}...")
    
    # (B) Generate final answer using the LLM
    print("\n   🤖 Final answer (LLM):")
    answer = get_response(rag_chain, q, [])
    print(f"   {answer}\n")
    print("-" * 70)

# -------------------------------------------------------------
# 4. (Optional) Pure BM25 and kNN for comparison
# -------------------------------------------------------------
es_client = Elasticsearch("http://localhost:9200")
print("\n" + "="*70)
print("PURE BM25 (first 3 results for 'admissions')")
print("="*70)
bm25_body = {
    "query": {"match": {"text": "admissions"}},
    "size": 3,
    "_source": ["text", "metadata"]
}
bm25_resp = es_client.search(index="my_rag_index", body=bm25_body)
for hit in bm25_resp["hits"]["hits"]:
    print(hit["_source"]["text"][:200])

print("\n" + "="*70)
print("PURE kNN (first 3 results for 'What is MIT famous for?')")
print("="*70)
query_vector = embeddings.embed_query("What is MIT famous for?")
knn_body = {
    "knn": {
        "field": "vector",
        "query_vector": query_vector,
        "k": 3,
        "num_candidates": 10
    },
    "size": 3,
    "_source": ["text", "metadata"]
}
knn_resp = es_client.search(index="my_rag_index", body=knn_body)
for hit in knn_resp["hits"]["hits"]:
    print(hit["_source"]["text"][:200])

print("\n✅ Test complete.")