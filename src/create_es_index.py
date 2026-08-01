from elasticsearch import Elasticsearch

# Connect to your local Elasticsearch (default port)
es = Elasticsearch("http://localhost:9200")

# Name your index
INDEX_NAME = "my_rag_index"

# Define the mapping
mapping = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},               # chunk content
            "metadata": {"type": "object"},         # source URL, etc.
            "embedding": {
                "type": "dense_vector",
                "dims": 768,        # ⚠️ CHANGE THIS to match your model
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

# Check if index already exists
if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"✅ Index '{INDEX_NAME}' created.")
else:
    print(f"ℹ️ Index '{INDEX_NAME}' already exists.")