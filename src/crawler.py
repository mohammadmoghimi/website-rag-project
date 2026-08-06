import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from langchain_core.documents import Document 
import streamlit as st

def normalize_url(url):
    url = urldefrag(url)[0]
    parsed = urlparse(url)
    path = parsed.path.rstrip('/') or '/'
    normalized = parsed._replace(
        netloc=parsed.netloc.lower(),
        path=path,
        fragment=''
    ).geturl()
    return normalized

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

class WebsiteCrawler:
    def __init__(
        self,
        start_url,
        max_pages=15,
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
                print(f"Blocked by robots.txt: {url}")
                return None, []

            time.sleep(self.delay)

            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                print(f" HTTP {response.status_code} for {url}")
                return None, []

            html = response.text

            try:
                soup = BeautifulSoup(html, "lxml")
            except:
                soup = BeautifulSoup(html, "html.parser")

            doc = Document(page_content=html, metadata={"source": url})
            print(f"Fetched: {url} (HTML: {len(html)} bytes)")

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

            if not links:
                print(f" No internal links found on {url}")

            return doc, links

        except requests.exceptions.Timeout:
            print(f"Timeout fetching {url}")
            return None, []
        except requests.exceptions.ConnectionError:
            print(f" Connection error for {url}")
            return None, []
        except Exception as e:
            print(f" Error fetching {url}: {e}")
            return None, []

    def crawl(self):
        print(f"Starting crawl of {self.start_url} (max {self.max_pages} pages)")

        if not can_fetch(self.start_url):
            print(f" Start URL blocked by robots.txt: {self.start_url}")
            return []

        with ThreadPoolExecutor(max_workers=self.concurrent_workers) as executor:
            futures = {executor.submit(self.fetch_and_parse, self.start_url): self.start_url}
            while futures and self.page_count < self.max_pages:
                for future in as_completed(futures):
                    url = futures.pop(future)
                    doc, links = future.result()

                    if doc is not None:
                        self.documents.append(doc)
                        self.page_count += 1
                        print(f" Page {self.page_count}/{self.max_pages} processed: {url}")
                    else:
                        print(f" No document returned for {url}")

                    self.visited.add(url)

                    if doc is not None: 
                        for link in links:
                            if link not in self.visited:
                                self.queue.append(link)
                                if self.page_count < self.max_pages:
                                    futures[executor.submit(self.fetch_and_parse, link)] = link
                                else:
                                    break

                    if self.page_count >= self.max_pages:
                        for f in futures:
                            f.cancel()
                        break

        if self.page_count == 0:
            print(" No pages fetched")

        return self.documents


def crawl_website(start_url, max_pages=15, **kwargs):
    crawler = WebsiteCrawler(start_url, max_pages=max_pages, **kwargs)
    return crawler.crawl()