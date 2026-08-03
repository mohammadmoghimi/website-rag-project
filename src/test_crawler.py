from crawler import crawl_website

url = "https://www.roadto1000goals.com/" 
# url = "https://www.mit.edu/" 
docs = crawl_website(url, max_pages=15)

print(f"✅ Crawled {len(docs)} documents")
for i, doc in enumerate(docs[:3]):  # show first 3
    print(f"Doc {i+1}: source={doc.metadata.get('source')}, content length={len(doc.page_content)}")
assert len(docs) > 0, "No documents crawled"
assert all(doc.page_content.strip() for doc in docs), "Empty content found"
print("✅ Crawler test passed!")