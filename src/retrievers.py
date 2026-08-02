from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from elasticsearch import Elasticsearch

class ElasticsearchHybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines BM25 and kNN search manually
    without using Elasticsearch's licensed RRF feature.
    """
    es_client: Elasticsearch
    index_name: str
    embedding_model: any
    text_field: str = "text"
    embedding_field: str = "embedding"
    k: int = 4
    num_candidates: int = 50
    bm25_weight: float = 0.5   # Adjust these weights
    vector_weight: float = 0.5  # Adjust these weights

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Generate embedding
        query_vector = self.embedding_model.embed_query(query)

        # 1. BM25 (text) search
        bm25_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [self.text_field]
                }
            },
            "size": self.num_candidates,  # Fetch more than k for better fusion
            "_source": [self.text_field, "metadata"]
        }
        bm25_response = self.es_client.search(index=self.index_name, body=bm25_body)
        bm25_hits = bm25_response["hits"]["hits"]

        # 2. Vector (kNN) search
        knn_body = {
            "knn": {
                "field": self.embedding_field,
                "query_vector": query_vector,
                "k": self.num_candidates,
                "num_candidates": self.num_candidates
            },
            "size": self.num_candidates,
            "_source": [self.text_field, "metadata"]
        }
        knn_response = self.es_client.search(index=self.index_name, body=knn_body)
        knn_hits = knn_response["hits"]["hits"]

        # 3. Manual score combination
        # Build a dict to store combined scores per document ID
        combined_scores = {}
        doc_store = {}  # Store document content and metadata

        # Process BM25 results
        for hit in bm25_hits:
            doc_id = hit["_id"]
            score = hit["_score"]
            # Normalize BM25 score (range is unpredictable, so we'll use rank position instead)
            # Better approach: use reciprocal rank fusion (RRF) manually
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 / (self.k + 1))
            doc_store[doc_id] = {
                "text": hit["_source"].get(self.text_field, ""),
                "metadata": hit["_source"].get("metadata", {})
            }

        # Process kNN results
        for hit in knn_hits:
            doc_id = hit["_id"]
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 / (self.k + 1))
            if doc_id not in doc_store:
                doc_store[doc_id] = {
                    "text": hit["_source"].get(self.text_field, ""),
                    "metadata": hit["_source"].get("metadata", {})
                }

        # Sort by combined score (higher is better) and take top k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:self.k]

        # Convert to LangChain Documents
        documents = []
        for doc_id, score in sorted_docs:
            doc_data = doc_store[doc_id]
            documents.append(
                Document(
                    page_content=doc_data["text"],
                    metadata=doc_data["metadata"]
                )
            )
        return documents