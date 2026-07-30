import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
import time

def log(message):
    """Simple logging that appends to Streamlit session state."""
    print(message)
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs.append(message)

def crawl_website(start_url, max_pages=30):
    """
    Crawls a website starting from the given URL, follows internal links,
    and returns a list of LangChain Document objects.
    """
    start_total = time.time()
    visited = set()
    queue = [start_url]
    documents = []
    domain = urlparse(start_url).netloc

    page_count = 0
    while queue and len(visited) < max_pages:
        current_url = queue.pop(0)
        if current_url in visited:
            continue
        try:
            page_start = time.time()
            log(f" Crawling: {current_url}")
            visited.add(current_url)

                        # --- Time the actual request ---
            t0 = time.time()
            response = requests.get(current_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            log(f"   GET request took {time.time()-t0:.2f}s")
            
            # --- WebBaseLoader also does a request! This is the duplicate ---
            t1 = time.time()

            # Load document using WebBaseLoader
            loader = WebBaseLoader(current_url)
            docs = loader.load()
            log(f"   WebBaseLoader took {time.time()-t1:.2f}s")
            log(f"   Loaded {len(docs)} docs from loader")
            for doc in docs:
                doc.metadata["source"] = current_url
            documents.extend(docs)

            # Find internal links for further crawling
            response = requests.get(current_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            t2 = time.time()
            soup = BeautifulSoup(response.text, "html.parser")
            log(f"   Parsing & link extraction took {time.time()-t2:.2f}s")
            log(f"   Total for this page: {time.time()-page_start:.2f}s")
            page_count += 1
            for link in soup.find_all("a", href=True):
                full_url = urljoin(current_url, link["href"])
                parsed = urlparse(full_url)
                if parsed.netloc == domain and full_url not in visited:
                    queue.append(full_url)
        except Exception as e:
            log(f" Error on {current_url}: {e}")
    log(f" Crawled {len(visited)} pages in {time.time()-start_total:.2f}s")
    return documents