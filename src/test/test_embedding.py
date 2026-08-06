import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY_FINEGRAINED")
)

text = "This is a test document."

vector = embeddings.embed_query(text)

print("Embedding size:", len(vector))
print("First 5 values:", vector[:5])