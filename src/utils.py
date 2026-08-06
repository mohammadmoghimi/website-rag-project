import re
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urlparse
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY_FINEGRAINED")

def extract_main_text(html: str) -> str:

    try:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False
        )
        if text and len(text.strip()) > 50:
            print("[INFO] trafilatura extracted text successfully.")
            return text
        else:
            print("[WARNING] trafilatura returned empty or too short; falling back to readability.")
    except Exception as e:
        print(f"[WARNING] trafilatura failed with error: {e}; falling back to readability.")

    print("[INFO] Using custom cleaning fallback.")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return text if text else "No content extracted."

def get_index_name_from_url(url):
    domain = urlparse(url).netloc
    safe = re.sub(r'[^a-zA-Z0-9]', '_', domain)
    return f"rag_{safe}"

def get_llm():

    endpoint = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Flash-0731", 
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        task="text-generation",
        max_new_tokens=512,    
        temperature=0,         
        timeout=60,         
    )
    return ChatHuggingFace(llm=endpoint)