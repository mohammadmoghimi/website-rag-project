import time
from rag import build_vectorstore_from_url  # your function
from elasticsearch import Elasticsearch

# 1. Build index (crawls + embeds + stores)
test_url = "https://www.mit.edu/"
vectorstore = build_vectorstore_from_url(test_url)  # adjust if your function has params

# 2. Check Elasticsearch directly
es = Elasticsearch("http://localhost:9200")
index_name = "my_rag_index"  # must match your code

# Count documents
count_resp = es.count(index=index_name)
print(f"📊 Index '{index_name}' has {count_resp['count']} documents")

# Fetch one sample
sample = es.search(index=index_name, size=1)
if sample["hits"]["hits"]:
    doc = sample["hits"]["hits"][0]["_source"]
    print("Sample document fields:", doc.keys())
    print("Text preview:", doc.get("text", "")[:200])
    print("Metadata:", doc.get("metadata", {}))
    # Check if embedding field exists (should be a list)
    if "embedding" in doc:
        print(f"✅ Embedding vector length: {len(doc['embedding'])}")
else:
    print("❌ No documents found!")

assert count_resp['count'] > 0, "Index is empty!"
print("✅ Indexing test passed!")