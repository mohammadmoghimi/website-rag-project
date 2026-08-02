import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from langchain_core.documents import Document   # <-- CORRECT IMPORT
import streamlit as st

# ------------------------------
# Logging
# ------------------------------
def log(message):
    print(message)
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs.append(message)

# ------------------------------
# URL normalisation
# ------------------------------
def normalize_url(url):
    """Remove fragments, enforce scheme, lower domain, strip trailing slash."""
    url = urldefrag(url)[0]
    parsed = urlparse(url)
    path = parsed.path.rstrip('/') or '/'
    normalized = parsed._replace(
        netloc=parsed.netloc.lower(),
        path=path,
        fragment=''
    ).geturl()
    return normalized

# ------------------------------
# Robots.txt parser (cached per domain)
# ------------------------------
_robot_cache = {}

def can_fetch(url, user_agent="Mozilla/5.0"):
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    if domain not in _robot_cache:
        rp = RobotFileParser()
        rp.set_url(urljoin(domain, "/robots.txt"))
        try:
            rp.read()
        except Exception:
            rp = None
        _robot_cache[domain] = rp
    rp = _robot_cache[domain]
    if rp is None:
        return True
    return rp.can_fetch(user_agent, url)

# ------------------------------
# Crawler class
# ------------------------------
class WebsiteCrawler:
    def __init__(
        self,
        start_url,
        max_pages=30,
        concurrent_workers=5,
        delay=0.5,
        timeout=10,
        max_retries=2,
        user_agent="Mozilla/5.0"
    ):
        self.start_url = normalize_url(start_url)
        self.domain = urlparse(self.start_url).netloc
        self.max_pages = max_pages
        self.concurrent_workers = concurrent_workers
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        retry_strategy = requests.adapters.Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.visited = set()
        self.queue = [self.start_url]
        self.documents = []
        self.page_count = 0

    def fetch_and_parse(self, url):
        try:
            if not can_fetch(url):
                log(f"Skipped (robots.txt): {url}")
                return None, []

            time.sleep(self.delay)

            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            html = response.text

            doc = Document(page_content=html, metadata={"source": url})
            log(f"Fetched: {url} (size: {len(html)} bytes)")

            # Use lxml if available, else fallback to html.parser
            try:
                soup = BeautifulSoup(html, "lxml")
            except:
                soup = BeautifulSoup(html, "html.parser")

            links = []
            for a in soup.find_all("a", href=True):
                full_url = urljoin(url, a["href"])
                normalized = normalize_url(full_url)
                parsed = urlparse(normalized)
                if (
                    parsed.netloc == self.domain
                    and normalized not in self.visited
                    and normalized not in self.queue
                    and parsed.scheme in ("http", "https")
                ):
                    links.append(normalized)

            return doc, links

        except Exception as e:
            log(f"Error fetching {url}: {e}")
            return None, []

    def crawl(self):
        log(f"Starting crawl of {self.start_url} (max {self.max_pages} pages)")

        with ThreadPoolExecutor(max_workers=self.concurrent_workers) as executor:
            futures = {executor.submit(self.fetch_and_parse, self.start_url): self.start_url}
            while futures and self.page_count < self.max_pages:
                for future in as_completed(futures):
                    url = futures.pop(future)
                    doc, links = future.result()

                    if doc is not None:
                        self.documents.append(doc)
                        self.page_count += 1
                        log(f"Page {self.page_count}/{self.max_pages} processed: {url}")

                        for link in links:
                            if link not in self.visited:
                                self.queue.append(link)
                                if self.page_count < self.max_pages:
                                    futures[executor.submit(self.fetch_and_parse, link)] = link
                                else:
                                    break

                    self.visited.add(url)

                    if self.page_count >= self.max_pages:
                        for f in futures:
                            f.cancel()
                        break

        log(f"Crawled {self.page_count} pages. Total documents: {len(self.documents)}")
        return self.documents

# ------------------------------
# Backwards‑compatible function
# ------------------------------
def crawl_website(start_url, max_pages=30, **kwargs):
    crawler = WebsiteCrawler(start_url, max_pages=max_pages, **kwargs)
    return crawler.crawl()