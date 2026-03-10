"""
Download Doodle Labs Technical Library documentation into the RAG data folder.

Fetches all documentation pages from https://techlibrary.doodlelabs.com/,
extracts main text, and saves as .txt files so they can be indexed by the RAG.

Usage:
    python scripts/download_doodle_labs.py

Output:
    data/doodle_labs/*.txt  (one file per page)
"""

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Configuration
BASE_URL = "https://techlibrary.doodlelabs.com"
START_URL = "https://techlibrary.doodlelabs.com/doodle-labs-technical-library"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "doodle_labs"
REQUEST_DELAY = 1.5  # seconds between requests (be polite to the server)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (compatible; RAG-doc-ingest/1.0)"
TIMEOUT = 30
MAX_PAGES = 200  # safety limit

# Skip URLs that are not documentation (anchors, external, assets)
SKIP_PATTERNS = (
    r"#",
    r"mailto:",
    r"\.zip\b",
    r"\.(jpg|jpeg|png|gif|css|js)\b",
    r"youtube\.com",
    r"facebook|twitter|linkedin",
    r"/tag/",
    r"/author/",
    r"wp-login",
    r"wp-admin",
    r"search\?",
    r"\.xml",
    r"feed",
    r"cart|checkout|account",
)


def is_same_domain(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == "techlibrary.doodlelabs.com" or (
        parsed.netloc == "" and url.startswith("/")
    )


def normalize_url(url: str) -> str:
    url = url.split("#")[0].strip()
    if not url:
        return ""
    return urljoin(BASE_URL + "/", url)


def should_skip(url: str) -> bool:
    for pat in SKIP_PATTERNS:
        if re.search(pat, url, re.I):
            return True
    return False


def url_to_filename(url: str) -> str:
    """Convert URL path to a safe filename."""
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    path = path.replace("/", "_").replace(" ", "_")
    path = re.sub(r"[^\w\-_.]", "_", path)
    if len(path) > 120:
        path = path[:120]
    return path + ".txt"


def fetch_page(url: str) -> str | None:
    """Fetch HTML of a page. Returns None on failure."""
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
            allow_redirects=True,
        )
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return None


def extract_text(html: str, url: str) -> str:
    """Extract main readable text from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav clutter
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "iframe"]):
        tag.decompose()

    # Prefer main content area if present
    main = soup.find("main") or soup.find("article") or soup.find(class_=re.compile(r"content|post|entry|document", re.I))
    root = main if main else soup.body
    if not root:
        root = soup

    text = root.get_text(separator="\n", strip=True)
    # Collapse multiple newlines and trim
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Prepend URL as source
    return f"Source: {url}\n\n{text}" if text else ""


def get_links(html: str, current_url: str):
    """Extract same-domain doc links and PDF links from HTML. Returns (pages, pdf_urls)."""
    soup = BeautifulSoup(html, "html.parser")
    pages = set()
    pdf_urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = normalize_url(href)
        if not full:
            continue
        if full.lower().endswith(".pdf"):
            if is_same_domain(full) or "doodlelabs.com" in full:
                pdf_urls.add(full)
            continue
        if not is_same_domain(full):
            continue
        if should_skip(full):
            continue
        if "techlibrary.doodlelabs.com" not in full and full.startswith("http"):
            continue
        pages.add(full)
    return list(pages), list(pdf_urls)


def download_pdf(url: str) -> Path | None:
    """Download a PDF and save to OUTPUT_DIR. Returns path or None."""
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        name = Path(urlparse(url).path).name or "document.pdf"
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        out = OUTPUT_DIR / name
        out.write_bytes(r.content)
        return out
    except Exception as e:
        print(f"  PDF failed {url}: {e}")
        return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Start URL: {START_URL}")
    print()

    to_visit = [normalize_url(START_URL)]
    visited = set()
    saved_pages = 0
    saved_pdfs = 0
    pdfs_seen = set()

    while to_visit and len(visited) < MAX_PAGES:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"[{len(visited)}] {url}")
        html = fetch_page(url)
        time.sleep(REQUEST_DELAY)

        if not html:
            continue

        # Discover new links and PDFs
        page_links, pdf_links = get_links(html, url)
        for link in page_links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)
        for pdf_url in pdf_links:
            if pdf_url not in pdfs_seen:
                pdfs_seen.add(pdf_url)
                print(f"  Downloading PDF: {pdf_url[:60]}...")
                time.sleep(REQUEST_DELAY)
                if download_pdf(pdf_url):
                    saved_pdfs += 1

        # Extract and save page text
        text = extract_text(html, url)
        if not text or len(text) < 100:
            print("  -> Skipped (too little text)")
            continue

        fname = url_to_filename(url)
        out_path = OUTPUT_DIR / fname
        out_path.write_text(text, encoding="utf-8")
        saved_pages += 1
        print(f"  -> Saved {fname} ({len(text)} chars)")

    print()
    print(f"Done. Visited {len(visited)} pages, saved {saved_pages} .txt + {saved_pdfs} PDFs to {OUTPUT_DIR}")
    print("Next: In the chatbot, click 'Update with New Documents' to index these files.")


if __name__ == "__main__":
    main()
