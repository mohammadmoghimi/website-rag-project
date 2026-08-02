import time
from elasticsearch import Elasticsearch
from langchain_ollama import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from rag import get_retriever_chain   # make sure this imports correctly

# -------------------------------------------------------------
# 1. Connect to existing index
# -------------------------------------------------------------
print("Connecting to existing Elasticsearch index...")
embeddings = OllamaEmbeddings(model="all-minilm")
vectorstore = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="my_rag_index",
    embedding=embeddings,
)
count = vectorstore.client.count(index="my_rag_index")['count']
print(f"✓ Connected. Index has {count} documents.")

# -------------------------------------------------------------
# 2. Build retriever chain
# -------------------------------------------------------------
retriever_chain = get_retriever_chain(vectorstore)

# -------------------------------------------------------------
# 3. Test queries
# -------------------------------------------------------------
queries = [
    "admissions",
    "research and innovation",
    "What is MIT famous for?",
]

print("\n" + "="*60)
print("HYBRID SEARCH RESULTS (Top 3 per query)")
print("="*60)

for q in queries:
    print(f"\n🔍 Query: '{q}'")
    # Invoke retriever – it returns a list of Documents
    docs = retriever_chain.invoke({
        "input": q,
        "chat_history": []
    })
    print(f"   Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:200].replace('\n', ' ')
        source = doc.metadata.get("source", "?")
        print(f"   {i}. [{source}] {preview}...")

# -------------------------------------------------------------
# 4. (Optional) Compare with pure BM25 and kNN
# -------------------------------------------------------------
es_client = Elasticsearch("http://localhost:9200")
print("\n" + "="*60)
print("PURE BM25 (first 3 results for 'admissions')")
print("="*60)
bm25_body = {
    "query": {"match": {"text": "admissions"}},
    "size": 3,
    "_source": ["text", "metadata"]
}
bm25_resp = es_client.search(index="my_rag_index", body=bm25_body)
for hit in bm25_resp["hits"]["hits"]:
    print(hit["_source"]["text"][:200])

print("\n" + "="*60)
print("PURE kNN (first 3 results for 'What is MIT famous for?')")
print("="*60)
query_vector = embeddings.embed_query("What is MIT famous for?")
knn_body = {
    "knn": {
        "field": "embedding",
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