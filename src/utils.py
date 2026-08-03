from bs4 import BeautifulSoup
import trafilatura

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